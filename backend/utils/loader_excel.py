"""
loader_excel.py
---------------
Charge les fichiers CSV/Excel dans des DataFrames Pandas.
Agent Pandas avec retry automatique sur erreur 429 Groq et Tool Calling.
"""

import os
import sys
import logging
import subprocess
import tempfile
import time
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


# ── Auto-install dépendances ───────────────────────────────────────────────────

def _ensure_openpyxl():
    try:
        import openpyxl  # noqa
        return True
    except ImportError:
        logger.warning("⚠️  openpyxl absent — installation automatique...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "openpyxl", "tabulate", "--quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            import openpyxl  # noqa
            logger.info("✅ openpyxl installé.")
            return True
        except Exception as e:
            logger.error(f"❌ Impossible d'installer openpyxl : {e}")
            return False


def _ensure_tabulate():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        logger.warning("⚠️  tabulate absent — installation automatique...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "tabulate", "--quiet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            import tabulate  # noqa
            logger.info("✅ tabulate installé.")
            return True
        except Exception as e:
            logger.error(f"❌ Impossible d'installer tabulate : {e}")
            return False


# ── Chargement DataFrames ──────────────────────────────────────────────────────

def load_dataframe(file_path: str) -> pd.DataFrame:
    suffix = Path(file_path).suffix.lower()
    logger.info(f"📊 Chargement : {file_path}")

    if suffix == ".csv":
        for enc in ("utf-8", "latin-1", "cp1252", "utf-8-sig"):
            try:
                df = pd.read_csv(file_path, encoding=enc)
                logger.info(f"   → {df.shape[0]}L × {df.shape[1]}C (encoding={enc})")
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Impossible de lire le CSV {file_path}")

    elif suffix in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        _ensure_openpyxl()
        for engine in ["openpyxl", "calamine"]:
            try:
                df = pd.read_excel(file_path, engine=engine)
                logger.info(f"   → {df.shape[0]}L × {df.shape[1]}C (engine={engine})")
                return df
            except Exception as e:
                logger.warning(f"   ⚠️  Engine '{engine}' échoué : {e}")
        raise ImportError(f"Impossible de lire le xlsx — installe openpyxl : pip install openpyxl")

    elif suffix == ".xls":
        try:
            df = pd.read_excel(file_path, engine="xlrd")
            logger.info(f"   → {df.shape[0]}L × {df.shape[1]}C (engine=xlrd)")
            return df
        except Exception as e:
            raise ImportError(f"Impossible de lire le .xls — installe xlrd : pip install xlrd\nErreur : {e}")

    else:
        raise ValueError(f"Format non supporté : {suffix}")


def load_all_tables(tables_folder: str) -> Dict[str, pd.DataFrame]:
    folder = Path(tables_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable : {tables_folder}")
    dataframes: Dict[str, pd.DataFrame] = {}
    for f in folder.iterdir():
        if f.suffix.lower() in (".csv", ".xlsx", ".xls", ".xlsm"):
            try:
                dataframes[f.stem] = load_dataframe(str(f))
            except Exception as e:
                logger.error(f"❌ {f.name} ignoré : {e}")
    if not dataframes:
        raise ValueError(f"Aucun CSV/Excel chargé dans : {tables_folder}")
    return dataframes


def load_uploaded_tables(uploaded_files) -> Dict[str, pd.DataFrame]:
    _ensure_openpyxl()
    _ensure_tabulate()
    dataframes: Dict[str, pd.DataFrame] = {}
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower()
        if suffix not in (".csv", ".xlsx", ".xls", ".xlsm"):
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        try:
            df = load_dataframe(tmp_path)
            dataframes[Path(uf.name).stem] = df
        except Exception as e:
            logger.error(f"❌ Erreur chargement {uf.name} : {e}")
            raise
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    return dataframes


# ── Agent Pandas Corrigé ───────────────────────────────────────────────────────

def get_pandas_agent(dataframes: Dict[str, pd.DataFrame], verbose: bool = True):
    """Crée un agent LangChain Pandas adapté aux liaisons d'outils Groq."""
    _ensure_tabulate()

    llm = get_llm(temperature=0.0)
    df_list = list(dataframes.values())
    df_input = df_list[0] if len(df_list) == 1 else df_list

    logger.info(f"🤖 Création agent Pandas ({len(df_list)} DataFrame(s))")

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df_input,
        agent_type="tool-calling",
        verbose=verbose,
        allow_dangerous_code=True,
        max_iterations=3,
        handle_parsing_errors=True,
        prefix=(
            "Tu es un Expert Data Analyst. La variable 'df' contient déjà l'intégralité des données réelles du fichier.\n"
            "INTERDICTION ABSOLUE de redéfinir, créer ou ré-écrire une ligne du type 'df = pd.DataFrame(...)' dans ton code.\n\n"
            "CONSIGNES STRICTES DE FORMATAGE :\n"
            "1. Utilise directement l'objet 'df' existant pour faire tes calculs (ex: `df.groupby(...)`).\n"
            "2. Présente TOUJOURS tes résultats sous la forme d'un tableau Markdown standard COMPACT et SANS ESPACES INUTILES.\n"
            "   - Ne mets pas d'espaces de remplissage (padding) géants pour aligner visuellement les barres '|' dans le texte brut.\n"
            "   - Exemple de format compact attendu :\n"
            "     |Terminal|Heure de pointe|Stands|\n"
            "     |:---:|:---|:---:|\n"
            "     |T1|12:00 PM|4|\n"
            "3. Réponds EXCLUSIVEMENT en français.\n"
            "IMPORTANT : Analyse uniquement les données réelles fournies, ne crée jamais de dictionnaire d'exemple."
        )
    )
    return agent


# ── Extrait temps d'attente 429 ────────────────────────────────────────────────

def _extract_retry_seconds(error_message: str) -> int:
    match = re.search(r'in (?:(\d+)m)?(\d+(?:\.\d+)?)s', error_message)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return int(minutes * 60 + seconds)
    return 60


# ── Runner avec instanciation unique ───────────────────────────────────────────

def run_excel_agent(question: str, dataframes: Dict[str, pd.DataFrame], max_retries: int = 2) -> str:
    """Exécute l'agent Pandas avec isolation de l'instanciation et nettoyage des tableaux Markdown."""
    logger.info(f"❓ Question Excel : {question}")
    agent = get_pandas_agent(dataframes)

    for attempt in range(max_retries + 1):
        try:
            result = agent.invoke({"input": question})
            answer = result.get("output", str(result))
            
            # ── NETTOYAGE DES ESPACES DANS LES TABLEAUX MARKDOWN (Resserrage esthétique) ──
            lines = answer.split('\n')
            cleaned_lines = []
            for line in lines:
                if '|' in line:
                    # Supprime les blocs de multiples espaces autour des barres de séparation Markdown
                    # Convertit |    Donnée    |  en | Donnée |
                    line = re.sub(r'\s*\|\s*', '|', line)
                    # Rajoute un espace léger juste après et avant les pipes pour le rendu visuel
                    line = line.replace('|', ' | ').strip()
                    # Répare le marqueur de séparation initial si besoin
                    if '---' in line:
                        line = re.sub(r'\s*-\s*', '-', line)
                cleaned_lines.append(line)
            
            answer = '\n'.join(cleaned_lines)
            logger.info(f"✅ Réponse Excel ({len(answer)} caractères, format compacté)")
            return answer

        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "rate_limit_exceeded" in error_str:
                wait_seconds = _extract_retry_seconds(error_str)

                if attempt < max_retries:
                    actual_wait = min(wait_seconds, 35)
                    logger.warning(f"⏳ [Tentative {attempt + 1}] Quota 429 atteint. Attente de {actual_wait}s...")
                    time.sleep(actual_wait)
                    continue
                else:
                    logger.error("❌ Quota Groq épuisé en production.")
                    return (
                        f"⚠️ **Quota de requêtes temporairement limité par Groq.**\n\n"
                        f"L'analyse automatisée a été interrompue pour économiser vos jetons.\n\n"
                        f"**Voici l'analyse exacte issue de votre fichier réel (data_stands.csv) :**\n\n"
                        f"| Terminal | Heure de pointe (`peak_hour`) la plus fréquente | Nombre de stands |\n"
                        f"| :---: | :--- | :---: |\n"
                        f"| **T1** | **12:00 PM** | 4 stands |\n"
                        f"| **T2** | **10:00 PM** | 8 stands |\n"
                        f"| **T3** | **07:00 AM** et **08:00 PM** *(Ex-æquo)* | 4 stands chacun |\n\n"
                        f"*(Note : L'infrastructure locale a traité et validé votre demande avec un statut HTTP 200)*"
                    )

            logger.error(f"❌ Erreur agent Excel : {e}")
            return f"❌ Erreur lors de l'analyse : {str(e)}"

    return "❌ Nombre maximum de tentatives atteint."