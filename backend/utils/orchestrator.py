"""
utils/orchestrator.py — CLASSIFICATEUR AVEC PRIORITÉ SÉMANTIQUE CORRECTE
──────────────────────────────────────────────────────────────────────────
Problème corrigé :
  - "Décrire l'algorithme des stands" allait vers Excel car "stand" était
    dans les excel_keywords du fallback → réponse inventée depuis le CSV.
  - Le mot "stand" (ou tout autre mot métier) ne suffit PAS à forcer Excel.
  - La priorité correcte est : PDF_FORCE > STAFFING > EXCEL > PDF_DEFAULT

LOGIQUE FINALE :
  1. Questions CONCEPTUELLES (décrire, expliquer, comment, algorithme...) → PDF FORCÉ
  2. Sinon : keyword_fallback par colonnes + mots métier SAUF si contexte sémantique
  3. Sinon : LLM classificateur avec résumé colonnes
"""

import logging
import os
import base64
import tempfile
import re
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Literal, Optional, List, Any

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings

from utils.llm_factory import get_llm
from utils.loader_excel import load_dataframe, run_excel_agent
from utils.rag_cache import RAG_ANSWER_CACHE

logger = logging.getLogger(__name__)

IntentType = Literal["pdf", "excel", "staffing", "unknown"]
RAM_PROJECTS_CACHE: Dict[int, Dict] = {}

DB_HOST     = "localhost"
DB_NAME     = "Motuldb"
DB_USER     = "postgres"
DB_PASSWORD = "postgres"


# ══════════════════════════════════════════════════════════════════════════════
# LISTES DE MOTS-CLÉS (centralisées ici pour maintenance facile)
# ══════════════════════════════════════════════════════════════════════════════

# Ces mots signalent une question CONCEPTUELLE → réponse dans les PDF/Word
# Peu importe ce que contient le CSV, un algorithme s'explique dans un document.
PDF_FORCE_KEYWORDS = [
    "décrire", "descrire", "décris", "description",
    "expliquer", "explication", "explique",
    "algorithme", "algo",
    "comment fonctionne", "comment est",
    "architecture", "contexte", "définition", "définir",
    "qu'est-ce que", "c'est quoi", "kesako",
    "procédure", "méthode", "méthodologie",
    "fonctionnement", "principe", "concept",
    "pourquoi", "objectif", "but", "utilité",
    "historique", "origine", "présentation",
    "sfd", "dossier", "documentation", "rapport",
    "qui est", "quel est le directeur", "quel est le responsable",
]

# Ces mots signalent une question sur des DONNÉES CHIFFRÉES → Excel
# MAIS seulement si aucun mot PDF_FORCE n'est présent.
EXCEL_DATA_KEYWORDS = [
    "combien", "nombre", "total", "somme", "moyenne",
    "maximum", "minimum", "max", "min",
    "fréquence", "plus fréquent", "top", "classement",
    "heure de pointe", "peak", "taux",
    "liste des", "liste complète",
    "statistiques", "statistique",
    "calculer", "calcule", "calcul",
    "quel est le chiffre", "donnez-moi les chiffres",
    "analyse des données",
]

# Mots-clés RH/comptables → Staffing (uniquement si calcul explicite demandé)
STAFFING_FORCE_KEYWORDS = [
    "tjm", "taux journalier", "jours travaillés", "jours travaillee",
    "rentabilité", "rentabilite", "profitabilité",
    "coût salarial", "cout salarial", "charge salariale",
    "occupation", "taux d'occupation",
    "gain", "perte", "bénéfice net",
    "freelancer", "prestataire",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER : résumé des DataFrames pour le LLM
# ══════════════════════════════════════════════════════════════════════════════

def _summarize_dataframes(dataframes: Dict[str, pd.DataFrame]) -> str:
    if not dataframes:
        return "Aucun tableau Excel/CSV disponible pour ce projet."
    lines = []
    for name, df in dataframes.items():
        cols = ", ".join(df.columns.tolist()[:15])
        lines.append(f"  - Fichier '{name}' : {len(df)} lignes, colonnes : [{cols}]")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATEUR PRINCIPAL — logique en couches
# ══════════════════════════════════════════════════════════════════════════════

def _classify_by_keywords(
    question: str,
    dataframes: Dict[str, pd.DataFrame],
) -> Optional[IntentType]:
    """
    Classification rapide SANS appel LLM.
    Retourne l'intent détecté ou None si non concluant.

    PRIORITÉ (ordre strict) :
      1. PDF_FORCE  → question conceptuelle, explicative, descriptive
      2. STAFFING   → calcul RH explicite
      3. EXCEL      → donnée chiffrée ET colonne correspondante disponible
      4. None       → laisser le LLM décider
    """
    q = question.lower()

    # ── PRIORITÉ 1 : PDF forcé (conceptuel/sémantique) ────────────────────
    for kw in PDF_FORCE_KEYWORDS:
        if kw in q:
            logger.info(f"📚 PDF forcé par mot-clé conceptuel : '{kw}'")
            return "pdf"

    # ── PRIORITÉ 2 : Staffing ─────────────────────────────────────────────
    for kw in STAFFING_FORCE_KEYWORDS:
        if kw in q:
            logger.info(f"👥 Staffing forcé par mot-clé RH : '{kw}'")
            return "staffing"

    # ── PRIORITÉ 3 : Excel — SEULEMENT si données chiffrées demandées ─────
    # Condition A : un mot Excel_DATA est présent
    has_data_kw = any(kw in q for kw in EXCEL_DATA_KEYWORDS)

    # Condition B : un nom de colonne du DataFrame apparaît dans la question
    col_match = None
    if dataframes:
        all_cols = set()
        for df in dataframes.values():
            for col in df.columns:
                normalized = col.lower().replace("_", " ")
                all_cols.add(normalized)
                all_cols.add(col.lower())
        for col in all_cols:
            if len(col) > 3 and col in q:
                col_match = col
                break

    if has_data_kw and (col_match or dataframes):
        logger.info(
            f"📊 Excel par mots-clés données "
            f"(data_kw=True, col_match={col_match})"
        )
        return "excel"

    if col_match and not has_data_kw:
        # Colonne trouvée mais pas de mot chiffré → ambigu, laisser le LLM décider
        logger.info(f"⚠️ Colonne '{col_match}' trouvée mais question ambiguë → LLM")
        return None

    return None


# ── Prompt LLM classificateur (fallback si keyword insuffisant) ───────────────

INTENT_PROMPT_LLM = ChatPromptTemplate.from_template(
    """Tu es l'orchestrateur d'un système IA multi-agent.
Ta mission : router la question vers le bon agent.

════════════════════════════════════════
DONNÉES EXCEL/CSV DISPONIBLES :
{dataframes_summary}
════════════════════════════════════════

RÈGLES (ordre de priorité strict) :

1. **pdf** : Question conceptuelle, explicative, procédurale, historique, ou administrative.
   La réponse se trouve dans un document texte (PDF, Word).
   → "décrire", "expliquer", "algorithme", "comment fonctionne", "qui est", "procédure",
     "architecture", "qu'est-ce que", "présentation", "documentation".
   → MÊME SI la question mentionne un mot qui existe dans le tableau Excel (ex: "stand",
     "terminal"), si elle demande une EXPLICATION ou une DESCRIPTION → **pdf**.

2. **excel** : Question sur des données quantitatives, statistiques ou des listes de données.
   La réponse est un calcul ou une extraction depuis les colonnes du tableau.
   → "combien", "total", "moyenne", "heure de pointe", "liste des", "fréquence", "maximum".

3. **staffing** : Calcul RH explicite (TJM, rentabilité, jours travaillés, coût salarial).

EXEMPLES :
- "Décrire l'algorithme de planification des stands" → **pdf** (question explicative)
- "Quels stands acceptent les A380 ?" → **excel** (extraction de données)
- "Heure de pointe par terminal" → **excel** (statistique)
- "Quelle est la procédure d'attribution des stands ?" → **pdf** (procédure)
- "Combien de stands sont au terminal T2 ?" → **excel** (comptage)
- "Qui est le directeur ?" → **pdf** (information administrative)

Réponds UNIQUEMENT par : `pdf`, `excel` ou `staffing`.

Question : {question}
Réponse :"""
)


def classify_intent(
    question:   str,
    dataframes: Optional[Dict[str, pd.DataFrame]] = None,
) -> IntentType:
    """
    Classificateur complet en deux passes :
    1. Keyword rapide (sans LLM)
    2. LLM avec contexte colonnes (si keyword non concluant)
    """
    logger.info(f"🧭 Classification : '{question[:80]}'")

    # Passe 1 : mots-clés (gratuit, instantané)
    intent = _classify_by_keywords(question, dataframes or {})
    if intent is not None:
        logger.info(f"   → Intent (keywords) : {intent}")
        return intent

    # Passe 2 : LLM avec contexte des colonnes
    logger.info("   → Aucun keyword concluant, appel LLM classificateur...")
    df_summary = _summarize_dataframes(dataframes or {})
    try:
        llm = get_llm(temperature=0.0, max_tokens=5)
        chain = INTENT_PROMPT_LLM | llm | StrOutputParser()
        result_str = chain.invoke({
            "question":           question,
            "dataframes_summary": df_summary,
        }).strip().lower()

        if "staffing" in result_str:
            intent = "staffing"
        elif "excel" in result_str:
            intent = "excel"
        else:
            intent = "pdf"

    except Exception as e:
        logger.error(f"❌ LLM classificateur échoué, repli PDF : {e}")
        intent = "pdf"

    logger.info(f"   → Intent (LLM) : {intent}")
    return intent


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTEUR POSTGRESQL → RAM
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_resources_from_postgres(project_id: int):
    logger.info(f"🔄 [BDD ➔ RAM] Projet {project_id}")

    conn   = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        "SELECT file_name, content FROM documents WHERE project_id = %s AND is_indexed = TRUE;",
        (project_id,)
    )
    db_docs = cursor.fetchall()
    cursor.close()
    conn.close()

    if not db_docs:
        logger.warning(f"⚠️ Aucun document indexé pour le projet {project_id}")
        return None, {}, []

    all_chunks:  List[Document]       = []
    dataframes:  Dict[str, pd.DataFrame] = {}
    docs_raw:    List[Dict[str, Any]] = []

    for doc in db_docs:
        file_name      = doc['file_name']
        base64_content = doc['content']
        suffix         = os.path.splitext(file_name)[1].lower()

        if not base64_content:
            continue

        if ";base64," in base64_content:
            base64_content = re.sub(r'^data:.*?;base64,', '', base64_content)
        elif "," in base64_content:
            base64_content = base64_content.split(",")[1]
        base64_content = base64_content.strip()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            try:
                file_bytes_decoded = base64.b64decode(base64_content)
                tmp.write(file_bytes_decoded)
                temp_filepath = tmp.name
                docs_raw.append({
                    "file_name":  file_name,
                    "file_bytes": file_bytes_decoded,
                    "suffix":     suffix,
                })
            except Exception as decode_err:
                logger.error(f"❌ Décodage Base64 : '{file_name}' : {decode_err}")
                continue

        try:
            if suffix in (".pdf", ".docx", ".doc"):
                from utils.loader_pdf import load_uploaded_files_from_paths
                chunks = load_uploaded_files_from_paths([temp_filepath])
                if chunks:
                    for chunk in chunks:
                        chunk.metadata["source"] = file_name
                    all_chunks.extend(chunks)
                    logger.info(f"✅ Chunks PDF/Word : {file_name}")
                else:
                    logger.warning(f"⚠️ Aucun contenu extrait de {file_name}")

            elif suffix in (".csv", ".xlsx", ".xls", ".xlsm"):
                dataframes[os.path.splitext(file_name)[0]] = load_dataframe(temp_filepath)
                logger.info(f"✅ DataFrame Excel/CSV : {file_name}")

        except Exception as process_err:
            logger.error(f"❌ Traitement {file_name} : {process_err}")
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)

    vectorstore = None
    if all_chunks:
        logger.info(f"🔢 Vectorisation {len(all_chunks)} chunks...")
        embeddings  = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        logger.info("✅ Index FAISS créé en RAM.")

    RAM_PROJECTS_CACHE[project_id] = {
        "vectorstore": vectorstore,
        "dataframes":  dataframes,
        "docs_raw":    docs_raw,
        "updated_at":  datetime.now(),
    }
    return vectorstore, dataframes, docs_raw


def get_project_resources(project_id: int):
    if project_id in RAM_PROJECTS_CACHE:
        logger.info(f"⚡ [RAM HIT] Projet {project_id}")
        res = RAM_PROJECTS_CACHE[project_id]
        return res["vectorstore"], res["dataframes"], res.get("docs_raw", [])
    logger.info(f"📡 [RAM MISS] Init projet {project_id}")
    return rebuild_resources_from_postgres(project_id)


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un assistant expert en extraction documentaire factuelle.

⚠️ RÈGLES STRICTES :
1. Si la question demande un NOM DE PERSONNE (directeur, responsable, fondateur, président...),
   cherche EXPLICITEMENT dans le contexte toute occurrence de nom propre et cite-le EXACTEMENT.
2. Si l'information est présente dans le contexte, tu DOIS la fournir.
3. Si l'information est ABSENTE du contexte, réponds :
   "❌ L'information demandée n'est pas accessible dans les extraits actuels du document."
4. Ne fabrique jamais de nom ou d'information.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE DIRECTE :"""
)


def hybrid_search(vectorstore, question: str, k: int = 20) -> list:
    semantic_docs = vectorstore.similarity_search(question, k=k)
    keywords  = re.findall(r'\b[A-ZÀ-Ÿ][a-zà-ÿ]{2,}\b|\b[A-ZÀ-Ÿ]{2,}\b', question)
    role_kws  = re.findall(
        r'\b(directeur|gérant|président|fondateur|PDG|DG|responsable|chef|directrice)\b',
        question, re.IGNORECASE
    )
    keywords = list(set(keywords + role_kws))
    if not keywords:
        return semantic_docs
    pattern = '|'.join(re.escape(kw) for kw in keywords)
    boosted, others = [], []
    for doc in semantic_docs:
        (boosted if re.search(pattern, doc.page_content, re.IGNORECASE) else others).append(doc)
    logger.info(f"🔍 Hybrid search : {len(boosted)} boostés / {len(others)} autres")
    return boosted + others


def run_pdf_agent(question: str, docs: List[Document]) -> str:
    logger.info(f"📄 Agent PDF : {len(docs)} docs")
    context = "\n\n---\n\n".join(
        [f"📄 Source : {d.metadata.get('source', 'Inconnu')}\n{d.page_content}" for d in docs]
    )
    llm   = get_llm(temperature=0, max_tokens=2048)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    logger.info(f"✅ Réponse PDF ({len(answer)} chars)")
    return answer


def get_staffing_docs_raw_from_postgres(project_id: int) -> list:
    logger.info(f"📂 [Staffing] Projet {project_id}")
    conn   = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        """SELECT file_name, content FROM documents
           WHERE project_id = %s
             AND (file_name ILIKE '%%staffing%%'
                  OR file_name ILIKE '%%freelancer%%'
                  OR file_name LIKE '%%.csv'
                  OR file_name LIKE '%%.xlsx');""",
        (project_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    docs_raw = []
    for row in rows:
        file_name = row["file_name"]
        content   = row["content"] or ""
        suffix    = os.path.splitext(file_name)[1].lower()
        if ";base64," in content:
            pure = re.sub(r'^data:.*?;base64,', '', content)
        elif "," in content:
            pure = content.split(",")[1]
        else:
            pure = content
        pure = pure.strip().replace(" ", "").replace("\n", "")
        try:
            file_bytes = base64.b64decode(pure)
            docs_raw.append({"file_name": file_name, "file_bytes": file_bytes, "suffix": suffix})
            logger.info(f"✅ Décodé : {file_name} ({len(file_bytes):,} bytes)")
        except Exception as e:
            logger.error(f"❌ Décodage '{file_name}': {e}")

    logger.info(f"📊 {len(docs_raw)} fichier(s) staffing en RAM")
    return docs_raw


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def orchestrate(
    question:    str,
    project_id:  int,
    force_agent: Optional[str] = None,
) -> Dict[str, Any]:

    result = {
        "agent_used": None,
        "intent":     None,
        "answer":     "",
        "error":      None,
        "docs":       [],
        "charts":     [],
        "sources":    [],
        "from_cache": False,
    }

    # 1. Cache
    cached = RAG_ANSWER_CACHE.get(question=question, project_id=project_id)
    if cached:
        logger.info(f"⚡ Cache hit (projet {project_id})")
        result.update(cached)
        return result

    try:
        # 2. Ressources RAM
        vectorstore, dataframes, docs_raw = get_project_resources(project_id)

        # 3. Routage
        if force_agent and force_agent.strip().lower() in ("pdf", "excel", "staffing"):
            intent = force_agent.strip().lower()
            logger.info(f"🎯 Agent forcé : {intent}")
        else:
            # classify_intent reçoit les dataframes pour le contexte colonnes
            intent = classify_intent(question, dataframes)

        result["intent"] = intent
        logger.info(f"🚦 Routage → Agent : {intent.upper()}")

        # 4. Exécution agent
        if intent == "pdf":
            if vectorstore is None:
                result["error"] = "⚠️ Aucun document PDF/Word indexé pour ce projet."
                return result
            result["agent_used"] = "Agent PDF (RAG FAISS en RAM)"
            docs = hybrid_search(vectorstore, question, k=20)
            result["docs"]   = docs
            result["answer"] = run_pdf_agent(question, docs) if docs else \
                               "Aucun document pertinent trouvé dans l'index."

        elif intent == "excel":
            if not dataframes:
                result["error"] = "⚠️ Aucun fichier Excel/CSV indexé pour ce projet."
                return result
            result["agent_used"] = "Agent Excel (Pandas en RAM)"
            result["answer"]     = run_excel_agent(question, dataframes)
            result["docs"]       = list(dataframes.keys())

        elif intent == "staffing":
            from agents.staffing_agent import run_staffing_agent
            docs_raw_staffing = get_staffing_docs_raw_from_postgres(project_id)
            if not docs_raw_staffing:
                result["error"]  = "⚠️ Aucun fichier staffing trouvé."
                result["answer"] = result["error"]
                return result
            staffing_result = run_staffing_agent(question=question, docs_raw=docs_raw_staffing)
            if staffing_result.get("error"):
                result["error"]  = staffing_result["error"]
                result["answer"] = staffing_result["error"]
            else:
                result["agent_used"] = "Agent Staffing (PostgreSQL → RAM → Calculs)"
                result["answer"]     = staffing_result["answer"]
                result["charts"]     = staffing_result.get("charts", [])
                result["sources"]    = staffing_result.get("sources", [])
        else:
            result["error"] = "❌ Intention non reconnue."

        # 5. Mise en cache si réponse valide
        if result["answer"] and not result.get("error"):
            ok = RAG_ANSWER_CACHE.set(
                question=question, project_id=project_id,
                answer=result["answer"], sources=result.get("sources", []),
                charts=result.get("charts", []),
                agent_used=result.get("agent_used", ""), intent=intent,
            )
            logger.info(f"{'💾 Mis en cache' if ok else '🚫 Non mis en cache'}")

    except Exception as e:
        logger.error(f"❌ Erreur orchestrateur : {e}", exc_info=True)
        result["error"] = f"❌ Erreur interne : {str(e)}"

    return result