"""
utils/staffing_charts.py
────────────────────────────────────────────────────────────────────
Génère les graphiques de staffing et les retourne sous deux formes :
  1. Base64 PNG  → pour l'affichage direct dans le chat Angular (img src)
  2. Dict Plotly → pour le composant Chart.js Angular existant

Graphiques disponibles :
  - occupation_mensuelle : barres jours/mois par employé
  - cout_mensuel         : courbe coût mensuel
  - repartition_projets  : camembert coût par projet
  - tjm_comparatif       : barres horizontales TJM par employé
  - rentabilite          : jauge ou barres gain/perte
"""

import base64
import io
import logging
from typing import Dict, Any, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATEUR PRINCIPAL — retourne tous les graphiques
# ══════════════════════════════════════════════════════════════════

def generate_staffing_charts(
    analysis: Dict[str, Any],
    employe_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retourne une liste de graphiques compatibles avec le frontend Angular.
    Chaque dict :
    {
      "title":   "Occupation mensuelle",
      "type":    "bar" | "line" | "pie",
      "base64":  "data:image/png;base64,...",   ← pour img src
      "chartjs": { type, data, options }          ← pour Chart.js Angular
    }
    """
    charts = []
    df: pd.DataFrame = analysis.get("dataframe", pd.DataFrame())

    if df.empty:
        return charts

    par_mois    = analysis.get("par_mois", {})
    par_projet  = analysis.get("par_projet", {})
    par_employe = analysis.get("par_employe", {})
    rentabilite = analysis.get("rentabilite")

    # 1. Occupation mensuelle (jours par mois)
    c = _chart_occupation_mensuelle(df, employe_filter)
    if c: charts.append(c)

    # 2. Coût mensuel (courbe)
    c = _chart_cout_mensuel(par_mois)
    if c: charts.append(c)

    # 3. Répartition par projet (camembert)
    c = _chart_repartition_projets(par_projet)
    if c: charts.append(c)

    # 4. TJM comparatif entre employés
    if len(par_employe) > 1:
        c = _chart_tjm_comparatif(par_employe)
        if c: charts.append(c)

    # 5. Rentabilité gain/perte
    if rentabilite:
        c = _chart_rentabilite(rentabilite)
        if c: charts.append(c)

    return charts


# ══════════════════════════════════════════════════════════════════
# GRAPHIQUE 1 — Occupation mensuelle
# ══════════════════════════════════════════════════════════════════

def _chart_occupation_mensuelle(df: pd.DataFrame, employe_filter: Optional[str]) -> Optional[Dict]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        pivot = df.pivot_table(
            index="mois", columns="employe", values="jours", aggfunc="sum"
        ).fillna(0).sort_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        months = list(pivot.index)
        x = np.arange(len(months))
        n_emp = len(pivot.columns)
        width = 0.8 / max(n_emp, 1)

        colors = plt.cm.Set2.colors
        for i, emp in enumerate(pivot.columns):
            vals = pivot[emp].values
            bars = ax.bar(x + i * width, vals, width, label=emp,
                         color=colors[i % len(colors)], alpha=0.85)
            # Annotations sur les barres
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                           f"{val:.0f}j", ha="center", va="bottom", fontsize=8)

        # Ligne de référence 21j (100% d'occupation)
        ax.axhline(y=21, color="red", linestyle="--", linewidth=1, alpha=0.6, label="21j (100%)")
        ax.axhline(y=17, color="orange", linestyle=":", linewidth=1, alpha=0.6, label="17j (80%)")

        ax.set_title("📅 Occupation mensuelle (jours travaillés)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mois")
        ax.set_ylabel("Jours travaillés")
        ax.set_xticks(x + width * (n_emp - 1) / 2)
        ax.set_xticklabels(months, rotation=45, ha="right")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        b64 = _fig_to_base64(fig)
        plt.close(fig)

        # Données Chart.js
        datasets = []
        palette = ["#534AB7", "#27AE60", "#E67E22", "#E74C3C", "#3498DB"]
        for i, emp in enumerate(pivot.columns):
            datasets.append({
                "label": emp,
                "data":  [round(v, 1) for v in pivot[emp].values],
                "backgroundColor": palette[i % len(palette)],
            })

        return {
            "title":  "Occupation mensuelle",
            "type":   "bar",
            "base64": b64,
            "chartjs": {
                "type": "bar",
                "data": {"labels": months, "datasets": datasets},
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Occupation mensuelle (jours)"}},
                    "scales": {"y": {"beginAtZero": True, "title": {"display": True, "text": "Jours"}}}
                }
            }
        }
    except Exception as e:
        logger.error(f"Erreur chart occupation : {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# GRAPHIQUE 2 — Coût mensuel (courbe)
# ══════════════════════════════════════════════════════════════════

def _chart_cout_mensuel(par_mois: Dict) -> Optional[Dict]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mois_sorted = sorted(par_mois.keys())
        couts = [par_mois[m]["total_cout"] for m in mois_sorted]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(mois_sorted, couts, marker="o", linewidth=2.5,
               color="#534AB7", markerfacecolor="white", markeredgewidth=2, markersize=8)
        ax.fill_between(mois_sorted, couts, alpha=0.15, color="#534AB7")

        for i, (m, c) in enumerate(zip(mois_sorted, couts)):
            ax.annotate(f"{c:,.0f}€", (m, c),
                       textcoords="offset points", xytext=(0, 10),
                       ha="center", fontsize=8, color="#534AB7")

        ax.set_title("💰 Évolution du coût mensuel (€)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mois")
        ax.set_ylabel("Coût (€)")
        ax.set_xticklabels(mois_sorted, rotation=45, ha="right")
        ax.grid(alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        plt.tight_layout()

        b64 = _fig_to_base64(fig)
        plt.close(fig)

        return {
            "title":  "Coût mensuel",
            "type":   "line",
            "base64": b64,
            "chartjs": {
                "type": "line",
                "data": {
                    "labels": mois_sorted,
                    "datasets": [{
                        "label": "Coût mensuel (dhs)",
                        "data":  couts,
                        "borderColor": "#534AB7",
                        "backgroundColor": "rgba(83,74,183,0.15)",
                        "fill": True,
                        "tension": 0.3,
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Coût mensuel (€)"}},
                    "scales": {"y": {"beginAtZero": True}}
                }
            }
        }
    except Exception as e:
        logger.error(f"Erreur chart coût mensuel : {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# GRAPHIQUE 3 — Répartition coût par projet (camembert)
# ══════════════════════════════════════════════════════════════════

def _chart_repartition_projets(par_projet: Dict) -> Optional[Dict]:
    if len(par_projet) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(par_projet.keys())
        values = [par_projet[p]["total_cout"] for p in labels]
        colors = plt.cm.Set3.colors[:len(labels)]

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        for t in autotexts:
            t.set_fontsize(9)

        ax.set_title("🗂️ Répartition du coût par projet", fontsize=13, fontweight="bold")
        plt.tight_layout()

        b64 = _fig_to_base64(fig)
        plt.close(fig)

        palette = ["#534AB7","#27AE60","#E67E22","#E74C3C","#3498DB","#9B59B6","#1ABC9C"]
        return {
            "title":  "Répartition par projet",
            "type":   "pie",
            "base64": b64,
            "chartjs": {
                "type": "pie",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "data": values,
                        "backgroundColor": palette[:len(labels)]
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Répartition coût par projet"}}
                }
            }
        }
    except Exception as e:
        logger.error(f"Erreur chart projets : {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# GRAPHIQUE 4 — TJM comparatif
# ══════════════════════════════════════════════════════════════════

def _chart_tjm_comparatif(par_employe: Dict) -> Optional[Dict]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        emps = list(par_employe.keys())
        tjms = [par_employe[e]["tjm_moyen"] for e in emps]
        colors = ["#534AB7" if t == max(tjms) else "#AEB4E8" for t in tjms]

        fig, ax = plt.subplots(figsize=(8, max(3, len(emps) * 0.8)))
        bars = ax.barh(emps, tjms, color=colors, height=0.6, edgecolor="white")

        for bar, val in zip(bars, tjms):
            ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                   f"{val:,.0f} €/j", va="center", fontsize=9)

        ax.set_title("💼 TJM comparatif par employé", fontsize=13, fontweight="bold")
        ax.set_xlabel("TJM (€/jour)")
        ax.grid(axis="x", alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        plt.tight_layout()

        b64 = _fig_to_base64(fig)
        plt.close(fig)

        return {
            "title":  "TJM comparatif",
            "type":   "bar",
            "base64": b64,
            "chartjs": {
                "type": "bar",
                "data": {
                    "labels": emps,
                    "datasets": [{
                        "label": "TJM moyen (dhs/jour)",
                        "data": tjms,
                        "backgroundColor": "#534AB7",
                    }]
                },
                "options": {
                    "indexAxis": "y",
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "TJM comparatif"}},
                    "scales": {"x": {"beginAtZero": True}}
                }
            }
        }
    except Exception as e:
        logger.error(f"Erreur chart TJM : {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# GRAPHIQUE 5 — Rentabilité (gain / perte)
# ══════════════════════════════════════════════════════════════════

def _chart_rentabilite(rentabilite: Dict) -> Optional[Dict]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "ca_facturable" in rentabilite:
            labels = ["CA Facturé", "Coût Total", "Gain Net"]
            values = [
                rentabilite["ca_facturable"],
                rentabilite["cout_total"],
                rentabilite["gain_net"],
            ]
        else:
            labels = ["Coût Facturé", "Coût Interne", "Gain Net"]
            values = [
                rentabilite["cout_facture"],
                rentabilite["cout_interne_total"],
                rentabilite["gain_net"],
            ]

        colors = ["#27AE60", "#E74C3C", "#534AB7" if values[2] >= 0 else "#C0392B"]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + abs(max(values)) * 0.02,
                   f"{val:+,.0f} €", ha="center", fontsize=10, fontweight="bold")

        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title(f"📊 Rentabilité — {rentabilite['statut']}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Montant (€)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        plt.tight_layout()

        b64 = _fig_to_base64(fig)
        plt.close(fig)

        return {
            "title":  "Rentabilité",
            "type":   "bar",
            "base64": b64,
            "chartjs": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Montant (€)",
                        "data": values,
                        "backgroundColor": colors,
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Rentabilité"}},
                    "scales": {"y": {"beginAtZero": False}}
                }
            }
        }
    except Exception as e:
        logger.error(f"Erreur chart rentabilité : {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════

def _fig_to_base64(fig) -> str:
    """Convertit une figure matplotlib en data URI base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"