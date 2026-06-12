"""
utils/orchestrator.py
---------------------
Orchestrateur Multi-Agent 100% Stateless (RAM) connecté à PostgreSQL.

1. Extrait et décode à la volée les fichiers Base64 depuis PostgreSQL.
2. Génère un index FAISS, des DataFrames Pandas, ou conserve les bytes bruts (docs_raw) en RAM.
3. Conserve les instances chaudes dans un cache global applicatif.
4. Classifie l'intention de la question via Llama-3.3-70B.
5. Route vers l'agent approprié (PDF RAG vs Excel Analyst vs Staffing Agent).
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

logger = logging.getLogger(__name__)

# ── 5. MISE À JOUR DE INTENTTYPE ──────────────────────────────────────────────
IntentType = Literal["pdf", "excel", "staffing", "unknown"]

# ── REGISTRE GLOBAL EN RAM (CACHE DES PROJETS ACTIFS) ─────────────────────────
RAM_PROJECTS_CACHE: Dict[int, Dict] = {}
 
# ⚙️ Paramètres d'accès PostgreSQL
DB_HOST = "localhost"
DB_NAME = "Motuldb"
DB_USER = "postgres"
DB_PASSWORD = "postgres"


# ── 1. CLASSIFICATEUR D'INTENTION MODIFIÉ ─────────────────────────────────────

INTENT_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un orchestrateur expert pour un système RAG multi-agent.
Ta mission est de router la question vers l'agent spécialisé le plus pertinent.

### CRITÈRES DE DÉCISION :
- **staffing** : questions sur des employés, TJM, jours travaillés, coûts RH,
                 rentabilité d'une ressource humaine, occupation, gain/perte employé.
                 Exemples : "combien a travaillé Jean ?", "TJM de Marie",
                 "est-ce que cet employé est rentable ?", "occupation en mars".
- **excel** : données structurées, tableaux de chiffres, calculs financiers
                 NON liés aux ressources humaines.
- **pdf** : explications de concepts, procédures, résumés de textes longs.

Réponds UNIQUEMENT par le mot : `pdf`, `excel` ou `staffing`.
Question : {question}"""
)

def classify_intent(question: str) -> IntentType:
    """Utilise Llama-3.3-70B pour router la demande vers l'agent PDF, Excel ou Staffing."""
    logger.info(f"🧭 Classification de l'intention : '{question[:80]}'")
    llm = get_llm(temperature=0.0, max_tokens=10)
    chain = INTENT_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().lower()
    
    if "staffing" in result:
        intent = "staffing"
    elif "excel" in result:
        intent = "excel"
    else:
        intent = "pdf"
        
    logger.info(f"   → Intention retenue : {intent}")
    return intent


# ── 2. EXTRACTEUR STATELESS DIRECT DEPUIS POSTGRESQL MODIFIÉ ──────────────────

def rebuild_resources_from_postgres(project_id: int):
    """
    Se connecte à PostgreSQL, télécharge les chaînes Base64 du projet,
    nettoie les en-têtes Data URL (data:...;base64,), décode le flux en mémoire
    et initialise les structures de données volatiles (RAM), y compris docs_raw.
    """
    logger.info(f"🔄 [BDD ➔ RAM] Extraction et reconstruction de la base documentaire pour le projet ID: {project_id}")
    
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Récupération des documents indexés du projet
    cursor.execute(
        "SELECT file_name, content FROM documents WHERE project_id = %s AND is_indexed = TRUE;",
        (project_id,)
    )
    db_docs = cursor.fetchall()
    cursor.close()
    conn.close()

    if not db_docs:
        logger.warning(f"⚠️ Aucun document marqué 'is_indexed = TRUE' en BDD pour le projet ID {project_id}")
        return None, {}, []

    all_chunks: List[Document] = []
    dataframes: Dict[str, pd.DataFrame] = {}
    docs_raw: List[Dict[str, Any]] = []

    for doc in db_docs:
        file_name = doc['file_name']
        base64_content = doc['content']
        suffix = os.path.splitext(file_name)[1].lower()

        if not base64_content:
            continue

        # Nettoyage des headers "data:...;base64," via Regex
        if ";base64," in base64_content:
            base64_content = re.sub(r'^data:.*?;base64,', '', base64_content)
        elif "," in base64_content:
            base64_content = base64_content.split(",")[1]

        # Nettoyage des espaces ou retours à la ligne parasites
        base64_content = base64_content.strip()

        # Création du fichier temporaire volatile
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            try:
                file_bytes_decoded = base64.b64decode(base64_content)
                tmp.write(file_bytes_decoded)
                temp_filepath = tmp.name
                
                # Enregistrement des bytes bruts pour le Staffing
                docs_raw.append({
                    "file_name": file_name,
                    "file_bytes": file_bytes_decoded,
                    "suffix": suffix,
                })
                
            except Exception as decode_err:
                logger.error(f"❌ Échec du décodage Base64 pour '{file_name}' : {decode_err}")
                continue

        try:
            if suffix in (".pdf", ".docx", ".doc"):
                from utils.loader_pdf import load_uploaded_files_from_paths
                chunks = load_uploaded_files_from_paths([temp_filepath])
                if chunks:
                    for chunk in chunks:
                        chunk.metadata["source"] = file_name
                    all_chunks.extend(chunks)
                    logger.info(f"✅ Chunks RAM extraits pour le document textuel : {file_name}")
                else:
                    logger.warning(f"⚠️ Le fichier {file_name} n'a généré aucun contenu exploitable.")
                
            elif suffix in (".csv", ".xlsx", ".xls", ".xlsm"):
                dataframes[os.path.splitext(file_name)[0]] = load_dataframe(temp_filepath)
                logger.info(f"✅ DataFrame RAM initialisé pour le tableau : {file_name}")
        except Exception as process_err:
            logger.error(f"❌ Erreur lors du traitement du fichier temporaire {file_name} : {process_err}")
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)  # 🗑️ Suppression immédiate

    # Montage du VectorStore FAISS éphémère en RAM
    vectorstore = None
    if all_chunks:
        logger.info(f"🔢 Vectorisation en RAM de {len(all_chunks)} chunks via Ollama...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        logger.info("✅ Index FAISS temporaire créé avec succès en RAM.")

    # Enregistrement dans le cache RAM
    RAM_PROJECTS_CACHE[project_id] = {
        "vectorstore": vectorstore,
        "dataframes": dataframes,
        "docs_raw": docs_raw,
        "updated_at": datetime.now()
    }
    return vectorstore, dataframes, docs_raw


def get_project_resources(project_id: int):
    """
    Fonction ultra-rapide pour le chat.
    Récupère directement depuis la RAM sans toucher à PostgreSQL ni recalculer les embeddings.
    """
    if project_id in RAM_PROJECTS_CACHE:
        logger.info(f"⚡ [CACHE RAM HIT] Récupération instantanée du contexte projet {project_id}")
        res = RAM_PROJECTS_CACHE[project_id]
        return res["vectorstore"], res["dataframes"], res.get("docs_raw", [])
    
    logger.info(f"📡 [CACHE RAM MISS] Première initialisation requise pour le projet {project_id}")
    return rebuild_resources_from_postgres(project_id)


# ── 3. AGENT PDF (RAG EN MÉMOIRE) ─────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un assistant expert en extraction et analyse documentaire.
Tu doit extraire les informations demandées avec une fidélité absolue, sans rien inventer ni généraliser.

⚠️ DIRECTIVES STRICTES DE MISE EN FORME :
1. Présente TOUJOURS tes réponses de manière aérée et hautement lisible.
2. Utilise des titres Markdown clairs (## pour les sections principales, ### pour les sous-sections).
3. Structure tes explications avec des listes à puces aérées ou des tableaux Markdown.
4. Intègre des émojis contextuels et pertinents au début des titres et des points clés pour guider l'œil.
5. Mets en gras (**Texte**) les concepts, règles de gestion et champs obligatoires critiques.

CONTEXTE DE SPÉCIFICATION :
{context}

QUESTION :
{question}

RÉPONSE FORMATEE EN MARKDOWN CORRETE :"""
)

def run_pdf_agent(question: str, docs: List[Document]) -> str:
    """Reçoit les documents pré-extraits par l'orchestrateur et génère la réponse finale via le LLM."""
    logger.info(f"📄 Agent PDF activé avec {len(docs)} documents sources.")
    
    context = "\n\n---\n\n".join(
        [f"📄 Source : {d.metadata.get('source', 'Inconnu')}\n{d.page_content}" for d in docs]
    )
    logger.info(f"   → Contexte assemblé : {len(context)} caractères")

    llm = get_llm(temperature=0.1, max_tokens=2048)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    logger.info(f"✅ Réponse PDF générée ({len(answer)} caractères)")
    return answer


def get_staffing_docs_raw_from_postgres(project_id: int) -> list:
    """
    Lit UNIQUEMENT les fichiers staffing depuis la table staffing_documents.
    Décode le base64 en bytes directement en RAM — rien sur le disque.
    """
    logger.info(f"📂 [Staffing BDD→RAM] Lecture des fichiers staffing projet {project_id}")
    
    conn   = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute(
        """SELECT file_name, content, extension
           FROM staffing_documents
           WHERE project_id = %s
           ORDER BY uploaded_at DESC;""",
        (project_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    docs_raw = []
    for row in rows:
        file_name = row["file_name"]
        content   = row["content"] or ""
        suffix    = row["extension"] or os.path.splitext(file_name)[1].lower()
        if not suffix.startswith("."): suffix = f".{suffix}"

        # Nettoyer le préfixe data-URL
        if ";base64," in content:
            pure = re.sub(r'^data:.*?;base64,', '', content)
        elif "," in content:
            pure = content.split(",")[1]
        else:
            pure = content

        pure = pure.strip().replace(" ", "").replace("\n", "")

        try:
            file_bytes = base64.b64decode(pure)
            docs_raw.append({
                "file_name":  file_name,
                "file_bytes": file_bytes,   # bytes en RAM, jamais écrits sur disque
                "suffix":     suffix,
            })
            logger.info(f"✅ Décodé en RAM : {file_name} ({len(file_bytes):,} bytes)")
        except Exception as e:
            logger.error(f"❌ Erreur décodage '{file_name}': {e}")

    logger.info(f"📊 {len(docs_raw)} fichier(s) staffing chargés en RAM")
    return docs_raw


# ── 4. ORCHESTRATEUR PRINCIPAL UNIFIÉ AVEC LE ROUTAGE STAFFING ────────────────

def orchestrate(
    question: str,
    project_id: int,
    force_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Point d'entrée de l'orchestrateur. Consomme les structures en RAM, route
    vers le bon agent (PDF, Excel, Staffing) et extrait les sources et graphiques nécessaires.
    """
    result = {
        "agent_used": None,
        "intent": None,
        "answer": "",
        "error": None,
        "docs": [],
        "charts": [],   # Initialisation pour accueillir les graphiques de l'agent staffing
        "sources": []   # Initialisation pour accueillir les métadonnées de sources du staffing
    }

    try:
        # 1. Récupération immédiate depuis le registre RAM ou reconstruction automatique
        vectorstore, dataframes, docs_raw = get_project_resources(project_id)

        # 2. Routage intelligent ou forcé
        intent = force_agent if force_agent in ("pdf", "excel", "staffing") else classify_intent(question)
        result["intent"] = intent

        # 3. Traitement selon l'intention identifiée
        if intent == "pdf":
            if vectorstore is None:
                result["error"] = "⚠️ Aucun document textuel (PDF/Word) indexé trouvé en BDD pour ce projet."
                return result
            
            result["agent_used"] = "Agent PDF (RAG Stateless en RAM)"
            docs = vectorstore.similarity_search(question, k=5)
            result["docs"] = docs
            
            if not docs:
                result["answer"] = "Aucun document pertinent trouvé dans l'index FAISS éphémère."
            else:
                result["answer"] = run_pdf_agent(question, docs)

        elif intent == "excel":
            if not dataframes:
                result["error"] = "⚠️ Aucun tableau de données (Excel/CSV) indexé trouvé en BDD pour ce projet."
                return result
            
            result["agent_used"] = "Agent Excel (Pandas en RAM)"
            result["answer"] = run_excel_agent(question, dataframes)
            result["docs"] = list(dataframes.keys()) 

        # ── 4. BLOC DE ROUTAGE CORRIGÉ POUR L'AGENT STAFFING ───────────────────
        elif intent == "staffing":
            from agents.staffing_agent import run_staffing_agent

            # Lire depuis staffing_documents (pas documents) — tout en RAM depuis PostgreSQL
            docs_raw = get_staffing_docs_raw_from_postgres(project_id)

            if not docs_raw:
                result["error"] = "⚠️ Aucun fichier dans staffing_documents pour ce projet."
                result["answer"] = result["error"]
                return result

            staffing_result = run_staffing_agent(
                question=question,
                docs_raw=docs_raw,
            )

            if staffing_result.get("error"):
                result["error"] = staffing_result["error"]
                result["answer"] = staffing_result["error"]
            else:
                result["agent_used"] = "Agent Staffing (Base64 PostgreSQL → RAM → Calculs → Graphiques)"
                result["answer"] = staffing_result["answer"]
                result["docs"] = []
                result["charts"] = staffing_result.get("charts", [])
                result["sources"] = staffing_result.get("sources", [])

        else:
            result["error"] = "❌ Intention non reconnue."

    except Exception as e:
        logger.error(f"❌ Erreur critique dans l'orchestrateur : {e}", exc_info=True)
        result["error"] = f"❌ Erreur interne : {str(e)}"

    return result