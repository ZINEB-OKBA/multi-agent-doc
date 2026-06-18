"""
routers/documents_router.py — VERSION AVEC INVALIDATION CACHE APRÈS UPLOAD
───────────────────────────────────────────────────────────────────────────
Modifications vs version précédente :
  1. Après receive_and_index_synchronous() → appel à RAG_ANSWER_CACHE.invalidate_project()
  2. Nouveau endpoint GET /cache/stats pour monitoring
  3. Nouveau endpoint DELETE /cache/{project_id} pour purge manuelle (admin)
"""

import logging
import re
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import psycopg2

from utils.orchestrator import rebuild_resources_from_postgres
from utils.rag_cache import RAG_ANSWER_CACHE      # ← NOUVEAU

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents (Stateless IA)"])

DB_HOST     = "localhost"
DB_NAME     = "Motuldb"
DB_USER     = "postgres"
DB_PASSWORD = "postgres"


# ── SCHÉMAS ──────────────────────────────────────────────────────────────────

class SpringBootUploadPayload(BaseModel):
    documentId: int
    fileName:   str
    content:    str
    projectId:  int


class IndexSyncResponse(BaseModel):
    status:       str
    projectId:    int
    documentId:   int
    fileName:     str
    fileType:     str
    chunks_count: int
    message:      str


# ── HELPERS BDD ───────────────────────────────────────────────────────────────

def _update_postgres_status(doc_id: int, is_indexed: bool, indexing: bool, error_msg: Optional[str] = None):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE documents
               SET is_indexed = %s, indexing = %s, index_error = %s, indexed_at = %s
               WHERE id = %s;""",
            (is_indexed, indexing, error_msg, datetime.now() if is_indexed else None, doc_id)
        )
        cursor.close()
        conn.close()
        logger.info(f"💾 [PG] Doc {doc_id} → is_indexed={is_indexed}, indexing={indexing}")
    except Exception as db_err:
        logger.error(f"❌ Écriture PostgreSQL échouée : {db_err}")


# ── ENDPOINT : INDEXATION SYNCHRONE ──────────────────────────────────────────

@router.post("/backoffice/receive-and-index", response_model=IndexSyncResponse, status_code=200)
async def receive_and_index_synchronous(payload: SpringBootUploadPayload):
    """
    Reçoit un fichier de Spring Boot, l'indexe en RAM (FAISS/Pandas),
    puis invalide le cache RAG du projet pour forcer le recalcul des réponses.
    """
    filename   = payload.fileName
    project_id = payload.projectId
    doc_id     = payload.documentId

    logger.info(f"⚡ [Upload] {filename} → projet {project_id}")

    if filename.lower().endswith(('.pdf', '.docx', '.doc')):
        file_type = "pdf"
    elif filename.lower().endswith(('.csv', '.xlsx', '.xls', '.xlsm')):
        file_type = "excel"
    else:
        raise HTTPException(status_code=400, detail=f"Extension non supportée : '{filename}'")

    _update_postgres_status(doc_id=doc_id, is_indexed=False, indexing=True)

    try:
        vectorstore, dataframes, *_ = rebuild_resources_from_postgres(project_id)

        chunks_count = 0
        if file_type == "pdf" and vectorstore:
            chunks_count = vectorstore.index.ntotal
        elif file_type == "excel" and dataframes:
            chunks_count = len(dataframes)

        _update_postgres_status(doc_id=doc_id, is_indexed=True, indexing=False)

        # ── INVALIDATION DU CACHE RAG ─────────────────────────────────────
        # Après un nouvel upload/indexation, le fingerprint documentaire change
        # automatiquement → toutes les futures requêtes feront un cache miss.
        # On appelle invalidate_project() pour mettre à jour le fingerprint
        # en mémoire immédiatement (sans attendre la prochaine question).
        RAG_ANSWER_CACHE.invalidate_project(project_id)
        logger.info(f"🔄 Cache RAG invalidé pour le projet {project_id} après indexation de '{filename}'")

        return IndexSyncResponse(
            status       = "READY",
            projectId    = project_id,
            documentId   = doc_id,
            fileName     = filename,
            fileType     = file_type,
            chunks_count = chunks_count,
            message      = f"'{filename}' indexé avec succès. Cache RAG invalidé → prochaines réponses recalculées."
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Échec indexation {filename} : {error_msg}", exc_info=True)
        _update_postgres_status(doc_id=doc_id, is_indexed=False, indexing=False, error_msg=error_msg)
        raise HTTPException(status_code=500, detail=f"Erreur IA : {error_msg}")


# ── ENDPOINT DEBUG : SEARCH ───────────────────────────────────────────────────

@router.post("/debug/search")
async def debug_search(body: dict):
    from utils.orchestrator import get_project_resources
    project_id = body.get("project_id")
    query      = body.get("query", "directeur général")
    vs, _, _   = get_project_resources(project_id)
    if vs:
        docs = vs.similarity_search(query, k=10)
        return [{"source": d.metadata.get("source"), "content": d.page_content[:200]} for d in docs]
    return {"error": "Pas de vectorstore disponible"}


# ── ENDPOINT : STATS DU CACHE ─────────────────────────────────────────────────

@router.get("/cache/stats")
async def get_cache_stats():
    """
    Retourne les statistiques du cache RAG.
    Utile pour monitorer l'efficacité du cache en développement.
    """
    return RAG_ANSWER_CACHE.stats()


# ── ENDPOINT : PURGE MANUELLE DU CACHE (ADMIN) ───────────────────────────────

@router.delete("/cache/{project_id}")
async def invalidate_project_cache(project_id: int):
    """
    Purge manuelle du cache pour un projet donné.
    Utile si les documents ont été modifiés côté BDD sans passer par l'API d'indexation.
    """
    RAG_ANSWER_CACHE.invalidate_project(project_id)
    return {
        "message":    f"Cache invalidé pour le projet {project_id}.",
        "cache_stats": RAG_ANSWER_CACHE.stats(),
    }


@router.delete("/cache")
async def clear_all_cache():
    """Vide tout le cache RAG (admin uniquement)."""
    count = RAG_ANSWER_CACHE.clear_all()
    return {"message": f"{count} entrées supprimées du cache."}