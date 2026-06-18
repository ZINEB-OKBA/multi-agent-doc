"""
services/chat_service.py — VERSION AVEC MÉMOIRE CONVERSATIONNELLE + HYBRID SEARCH
─────────────────────────────────────────────────────────────────────────────────────
Ajouts par rapport à la version précédente :
  1. stream_response accepte conversation_history (liste de messages)
  2. Le prompt RAG inclut l'historique pour que le LLM comprenne le contexte
  3. hybrid_search boost les chunks contenant des mots-clés de la question
  4. Les sources sont retournées dans le message "done"
"""

import json
import logging
import asyncio
import re
from typing import Optional, Dict, List
import pandas as pd

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm_factory import get_llm
from utils.orchestrator import classify_intent

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

class WebSocketStreamHandler(AsyncCallbackHandler):
    def __init__(self, send_fn):
        self.send_fn = send_fn
        self.tokens: list[str] = []

    async def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)
        await self.send_fn(json.dumps({"type": "token", "token": token}))


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT AVEC MÉMOIRE CONVERSATIONNELLE
# ══════════════════════════════════════════════════════════════════════════════

RAG_PROMPT_WITH_MEMORY = ChatPromptTemplate.from_template(
    """Tu es un assistant expert en analyse documentaire. Tu travailles EXCLUSIVEMENT
à partir des documents fournis dans le contexte ci-dessous.

══════════════════════════════════════════════
RÈGLES ABSOLUES :
══════════════════════════════════════════════
1. Ne fabrique JAMAIS d'information absente du contexte.
2. Cite les valeurs exactes (chiffres, dates, noms, règles).
3. Si la réponse est incomplète, dis-le clairement.
4. Si l'information demandée (nom, titre, poste) est présente dans le contexte,
   tu DOIS la fournir — cherche toutes les occurrences de noms propres et titres.

══════════════════════════════════════════════
MISE EN FORME OBLIGATOIRE :
══════════════════════════════════════════════
- Commence par un résumé direct en 1-2 phrases.
- Utilise ## pour les sections principales.
- Utilise des listes à tirets (- item) pour les énumérations.
- Mets en **gras** les mots-clés, règles et valeurs importantes.
- Pour les tableaux de données, utilise le format Markdown.
- Termine TOUJOURS par une section ## Sources consultées
  listant chaque fichier utilisé sous la forme :
  - 📄 **NomDuFichier** (page X) : [phrase résumant ce qui y a été trouvé]

══════════════════════════════════════════════
HISTORIQUE DE LA CONVERSATION :
══════════════════════════════════════════════
{history}

══════════════════════════════════════════════
CONTEXTE DOCUMENTAIRE :
══════════════════════════════════════════════
{context}

══════════════════════════════════════════════
QUESTION ACTUELLE :
══════════════════════════════════════════════
{question}

Réponds en français. Sois précis, structuré et lisible.

RÉPONSE :"""
)


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID SEARCH — sémantique + boost par mots-clés
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_search(vectorstore: FAISS, question: str, k: int = 20) -> list:
    """
    Combine recherche sémantique (FAISS) + filtre par mots-clés critiques.
    Les chunks contenant les mots-clés de la question passent en tête de liste.
    """
    # 1. Recherche sémantique large
    semantic_docs = vectorstore.similarity_search(question, k=k)

    # 2. Extraire les mots-clés : noms propres + mots en majuscules
    keywords = re.findall(
        r'\b[A-ZÀ-Ÿ][a-zà-ÿ]{2,}\b|\b[A-ZÀ-Ÿ]{2,}\b',
        question
    )
    # Ajouter aussi les mots courants liés aux personnes/rôles
    role_keywords = re.findall(
        r'\b(directeur|gérant|président|fondateur|PDG|DG|responsable|chef|directrice)\b',
        question,
        re.IGNORECASE
    )
    keywords = list(set(keywords + role_keywords))

    if not keywords:
        return semantic_docs

    # 3. Trier : docs avec mots-clés en premier
    keyword_pattern = '|'.join(re.escape(kw) for kw in keywords)
    boosted = []
    others = []
    for doc in semantic_docs:
        if re.search(keyword_pattern, doc.page_content, re.IGNORECASE):
            boosted.append(doc)
        else:
            others.append(doc)

    logger.info(
        f"🔍 Hybrid search : {len(boosted)} docs boostés / {len(others)} autres "
        f"(mots-clés: {keywords})"
    )
    return boosted + others


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — formate l'historique pour le prompt
# ══════════════════════════════════════════════════════════════════════════════

def _format_history(conversation_history: List[Dict]) -> str:
    """
    Transforme la liste [{role, content}, ...] en texte lisible pour le LLM.
    On garde les N derniers échanges pour ne pas dépasser la fenêtre de contexte.
    """
    MAX_TURNS = 6  # 3 échanges Q/R = 6 messages
    recent = conversation_history[-MAX_TURNS:] if len(conversation_history) > MAX_TURNS else conversation_history

    if not recent:
        return "Aucun échange précédent."

    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = "👤 Utilisateur" if role == "user" else "🤖 Assistant"
        lines.append(f"{prefix} : {content}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — construit la liste des sources depuis les docs FAISS
# ══════════════════════════════════════════════════════════════════════════════

def _build_sources(docs: list) -> list:
    """
    Regroupe les chunks FAISS par fichier source.
    Retourne une liste de dicts sérialisables en JSON.
    """
    from collections import defaultdict
    groups = defaultdict(lambda: {"pages": set(), "count": 0})

    for doc in docs:
        src  = doc.metadata.get("source", "Document inconnu")
        page = doc.metadata.get("page")
        groups[src]["count"] += 1
        if page is not None:
            groups[src]["pages"].add(int(page) + 1)

    result = []
    for file_name, info in groups.items():
        pages = sorted(info["pages"])
        if len(pages) >= 2:
            pages_str = f"{pages[0]}–{pages[-1]}"
        elif len(pages) == 1:
            pages_str = str(pages[0])
        else:
            pages_str = None

        result.append({
            "fileName":     file_name,
            "pages":        pages_str,
            "extractCount": info["count"],
        })

    return sorted(result, key=lambda s: s["extractCount"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# SERVICE CHAT
# ══════════════════════════════════════════════════════════════════════════════

class ChatService:

    async def stream_response(
        self,
        question:             str,
        project:              str,
        vectorstore:          Optional[FAISS],
        dataframes:           Optional[Dict[str, pd.DataFrame]],
        send_fn,
        force_agent:          Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,  # ← NOUVEAU
    ):
        """
        Point d'entrée principal du chat.

        :param conversation_history: Liste de messages précédents au format
               [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
               Le message courant (question) NE doit PAS être dans cette liste.
        """
        full_text  = ""
        agent_used = ""
        sources    = []

        # Valeur par défaut si non fourni
        if conversation_history is None:
            conversation_history = []

        try:
            intent = force_agent if force_agent in ("pdf", "excel") \
                     else classify_intent(question)
            agent_label = "Agent PDF (RAG)" if intent == "pdf" else "Agent Excel (Pandas)"
            agent_used  = agent_label

            await send_fn(json.dumps({
                "type":   "intent",
                "intent": intent,
                "agent":  agent_label,
            }))

            if intent == "pdf":
                full_text, sources = await self._stream_pdf(
                    question, vectorstore, send_fn, conversation_history
                )
            elif intent == "excel":
                full_text = await self._run_excel(question, dataframes, send_fn)

            await send_fn(json.dumps({
                "type":       "done",
                "full_text":  full_text,
                "agent_used": agent_used,
                "intent":     intent,
                "sources":    sources,
            }))

        except Exception as e:
            logger.error(f"Erreur ChatService: {e}", exc_info=True)
            await send_fn(json.dumps({"type": "error", "message": str(e)}))


    async def _stream_pdf(
        self,
        question:             str,
        vectorstore,
        send_fn,
        conversation_history: List[Dict],
    ):
        """Retourne (full_text, sources). Intègre l'historique dans le prompt."""
        if vectorstore is None:
            msg = "⚠️ Aucun document indexé pour ce projet."
            await send_fn(json.dumps({"type": "token", "token": msg}))
            return msg, []

        logger.info(f"🔍 Hybrid search dans le vectorstore (k=20) pour : {question}")
        docs = hybrid_search(vectorstore, question, k=20)

        if not docs:
            msg = "❌ Aucun document pertinent trouvé."
            await send_fn(json.dumps({"type": "token", "token": msg}))
            return msg, []

        sources = _build_sources(docs)

        context = "\n\n---\n\n".join(
            f"📄 Source : {d.metadata.get('source', 'Inconnu')}\n{d.page_content}"
            for d in docs
        )

        # Formatage de l'historique
        history_text = _format_history(conversation_history)

        handler = WebSocketStreamHandler(send_fn)
        llm     = get_llm(temperature=0, max_tokens=2048, streaming=True, callbacks=[handler])
        chain   = RAG_PROMPT_WITH_MEMORY | llm | StrOutputParser()

        result = await chain.ainvoke({
            "context":  context,
            "question": question,
            "history":  history_text,   # ← injecté dans le prompt
        })

        return result or "".join(handler.tokens), sources


    async def _run_excel(self, question: str, dataframes, send_fn) -> str:
        if not dataframes:
            msg = "⚠️ Aucun tableau chargé pour ce projet."
            await send_fn(json.dumps({"type": "token", "token": msg}))
            return msg

        from utils.loader_excel import run_excel_agent
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_excel_agent, question, dataframes)

        words = result.split(" ")
        chunk = []
        for w in words:
            chunk.append(w)
            if len(chunk) == 4:
                await send_fn(json.dumps({"type": "token", "token": " ".join(chunk) + " "}))
                chunk = []
                await asyncio.sleep(0.02)
        if chunk:
            await send_fn(json.dumps({"type": "token", "token": " ".join(chunk)}))

        return result