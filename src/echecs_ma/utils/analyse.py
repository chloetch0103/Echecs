"""Analyse du journal d'un entraînement : résumé chiffré, alertes et tableau de bord.

Le journal (journal.csv) contient une ligne toutes les `journal.frequence` parties,
avec les colonnes écrites par scripts/entrainer.py. La config copiée dans le dossier
de l'entraînement (config.yaml) fournit les seuils (budget de danger d, seuil de
montée du curriculum).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

COLONNES = ["parties", "z_moyen", "echecs_subis", "lambda", "niveau", "epsilon", "erreur_r", "erreur_c",
            "pourcentage_victoire", "pourcentage_nulle", "pourcentage_defaite", "longueur_moy_partie", "duree"]

# Couleurs (palette de référence, mode clair) : identité des séries, encres et repères
BLEU, ORANGE, GRIS_NULLE = "#2a78d6", "#eb6834", "#b5b3ad"
ENCRE, ENCRE_SECONDAIRE, REPERE, GRILLE, FOND = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

EPSILON_EXPLORATION = 0.2     # au-dessus, les coups sont surtout tirés au hasard


# --- Chargement ---------------------------------------------------------------

@dataclass
class Entrainement:
    dossier: Path
    journal: pd.DataFrame
    config: dict = field(default_factory=dict)

    def reglage(self, *cles, defaut=None):
        """Lit une valeur imbriquée de la config, ex. reglage("contrainte", "lagrangien", "seuil_danger")."""
        valeur = self.config
        for cle in cles:
            if not isinstance(valeur, dict) or cle not in valeur:
                return defaut
            valeur = valeur[cle]
        return valeur


def dernier_entrainement(dossier_runs: Path) -> Path:
    """Le dossier d'entraînement le plus récent dans runs/."""
    candidats = [d for d in Path(dossier_runs).iterdir() if d.is_dir() and any(d.glob("*.csv"))]
    if not candidats:
        raise FileNotFoundError(f"Aucun entraînement (dossier contenant un .csv) dans {dossier_runs}")
    return max(candidats, key=lambda d: d.stat().st_mtime)


def charger(dossier: Path) -> Entrainement:
    dossier = Path(dossier)
    config = {}
    if (dossier / "config.yaml").is_file():
        with open(dossier / "config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    nom_csv = config.get("journal", {}).get("csv_file_name", "journal.csv")
    chemin_csv = dossier / nom_csv
    if not chemin_csv.is_file():
        csvs = sorted(dossier.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"Aucun journal .csv dans {dossier}")
        chemin_csv = csvs[0]
    journal = pd.read_csv(chemin_csv)
    manquantes = [c for c in COLONNES if c not in journal.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes du journal : {', '.join(manquantes)}")
    if journal.empty:
        raise ValueError("Le journal est vide : l'entraînement s'est arrêté avant la première écriture.")
    return Entrainement(dossier, journal, config)


# --- Calculs ------------------------------------------------------------------

def fenetre_lissage(nb_lignes: int, demande: int | None = None) -> int:
    """Nombre de lignes de la moyenne glissante : environ 1/20 du journal, au moins 1."""
    if demande:
        return max(1, demande)
    return max(1, nb_lignes // 20)


def lisser(serie: pd.Series, fenetre: int) -> pd.Series:
    return serie.rolling(fenetre, min_periods=1).mean()


def fin_du_journal(journal: pd.DataFrame) -> pd.DataFrame:
    """Les 10 % dernières lignes (au moins une) : l'état « final » de l'entraînement."""
    return journal.tail(max(1, len(journal) // 10))


def parties_par_niveau(journal: pd.DataFrame) -> dict[int, int]:
    """Nombre de parties jouées à chaque niveau de l'adversaire."""
    taille_blocs = journal["parties"].diff().fillna(journal["parties"].iloc[0])
    return {int(n): int(t) for n, t in taille_blocs.groupby(journal["niveau"]).sum().items()}


def pente_par_100_parties(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x.to_numpy(float), y.to_numpy(float), 1)[0] * 100)


def resume(e: Entrainement, parties_a_estimer: int | None = None) -> dict:
    j = e.journal
    fin = fin_du_journal(j)
    w = fenetre_lissage(len(j))
    parties = int(j["parties"].iloc[-1])
    duree = float(j["duree"].sum())
    estimees = parties_a_estimer or e.reglage("entrainement", "nb_parties")
    res = {
        "parties": parties,
        "duree_totale_s": duree,
        "secondes_par_partie": duree / parties if parties else float("nan"),
        "z_final": float(fin["z_moyen"].mean()),
        "meilleur_z_lisse": float(lisser(j["z_moyen"], w).max()),
        "victoires_final": float(fin["pourcentage_victoire"].mean()),
        "nulles_final": float(fin["pourcentage_nulle"].mean()),
        "defaites_final": float(fin["pourcentage_defaite"].mean()),
        "echecs_final": float(fin["echecs_subis"].mean()),
        "budget_d": e.reglage("contrainte", "lagrangien", "seuil_danger"),
        "longueur_finale": float(fin["longueur_moy_partie"].mean()),
        "niveau_final": int(j["niveau"].iloc[-1]),
        "parties_par_niveau": parties_par_niveau(j),
        "lambda_debut": float(j["lambda"].iloc[0]),
        "lambda_final": float(j["lambda"].iloc[-1]),
        "pente_lambda_100": pente_par_100_parties(fin["parties"], fin["lambda"]),
        "epsilon_final": float(j["epsilon"].iloc[-1]),
        "erreur_r_finale": float(fin["erreur_r"].mean()),
        "erreur_c_finale": float(fin["erreur_c"].mean()),
    }
    if estimees and parties:
        res["parties_estimees"] = int(estimees)
        res["duree_estimee_s"] = res["secondes_par_partie"] * int(estimees)
    return res


def alertes(e: Entrainement) -> list[str]:
    j = e.journal
    messages = []
    erreurs = j[["erreur_r", "erreur_c"]].to_numpy(float)

    if not np.isfinite(erreurs).all():
        messages.append("Erreurs non finies (nan ou inf) dans le journal : l'apprentissage a divergé.")

    if np.isfinite(erreurs).all() and (erreurs == 0).all():
        messages.append("Aucune correction des réseaux : l'apprentissage n'a pas commencé "
                        "(debut_apprentissage trop grand pour cet essai ?).")

    exploration = j[j["epsilon"] > EPSILON_EXPLORATION]
    if len(exploration) >= 3:
        hausse = exploration["lambda"].iloc[-1] - exploration["lambda"].iloc[0]
        if hausse > 0 and exploration["lambda"].is_monotonic_increasing:
            messages.append(f"λ a augmenté de {hausse:.2f} pendant l'exploration (ε > {EPSILON_EXPLORATION}) "
                            "alors que les coups y sont surtout aléatoires : il risque d'écraser le retour "
                            "quand l'exploitation commencera.")

    for col, nom in (("erreur_r", "retour"), ("erreur_c", "danger")):
        lisse = lisser(j[col], fenetre_lissage(len(j))).replace(0, np.nan).dropna()
        if len(lisse) >= 5 and np.isfinite(lisse).all():
            mediane = float(lisse.median())
            if mediane > 0 and lisse.iloc[-1] > 10 * mediane:
                messages.append(f"L'erreur de {nom} finit à plus de 10 fois sa valeur médiane "
                                f"({lisse.iloc[-1]:.3g} contre {mediane:.3g}) : risque de divergence.")

    d = e.reglage("contrainte", "lagrangien", "seuil_danger")
    fin = fin_du_journal(j)
    if d is not None and fin["epsilon"].mean() <= EPSILON_EXPLORATION and fin["echecs_subis"].mean() > d:
        messages.append(f"Budget de danger dépassé en fin d'entraînement : {fin['echecs_subis'].mean():.1f} "
                        f"échecs subis par partie pour d = {d}.")

    if fin["pourcentage_nulle"].mean() > 0.9:
        messages.append("Plus de 90 % de nulles en fin d'entraînement : les agents ne parviennent pas à conclure.")

    return messages


# --- Texte --------------------------------------------------------------------

def duree_lisible(secondes: float) -> str:
    secondes = round(secondes)
    h, reste = divmod(secondes, 3600)
    m, s = divmod(reste, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


def resume_en_texte(e: Entrainement, res: dict, liste_alertes: list[str]) -> str:
    pct = lambda x: f"{100 * x:.0f} %"
    lignes = [
        f"# Analyse de l'entraînement {e.dossier.name}",
        "",
        "## Déroulement",
        f"- Parties jouées : {res['parties']}",
        f"- Durée : {duree_lisible(res['duree_totale_s'])} ({res['secondes_par_partie']:.2f} s par partie)",
    ]
    if "duree_estimee_s" in res:
        lignes.append(f"- Estimation pour {res['parties_estimees']} parties : {duree_lisible(res['duree_estimee_s'])}")
    niveaux = ", ".join(f"niveau {n} : {p} parties" for n, p in res["parties_par_niveau"].items())
    lignes += [
        f"- Niveau final de l'adversaire : {res['niveau_final']} ({niveaux})",
        f"- ε final : {res['epsilon_final']:.3f}",
        "",
        "## Fin de l'entraînement (10 % dernières lignes du journal)",
        f"- Résultat moyen z : {res['z_final']:+.3f} (meilleur z lissé : {res['meilleur_z_lisse']:+.3f})",
        (f"- Victoires / nulles / défaites : {pct(res['victoires_final'])} / {pct(res['nulles_final'])} / "
        f"{pct(res['defaites_final'])}"),
        f"- Longueur moyenne des parties : {res['longueur_finale']:.1f} coups",
    ]
    budget = f" pour un budget d = {res['budget_d']}" if res["budget_d"] is not None else ""
    lignes += [
        f"- Échecs subis par partie : {res['echecs_final']:.1f}{budget}",
        (f"- λ : {res['lambda_debut']:.3f} au début, {res['lambda_final']:.3f} à la fin "
        f"({res['pente_lambda_100']:+.3f} par 100 parties sur la fin)"),
        f"- Erreurs : retour {res['erreur_r_finale']:.4g}, danger {res['erreur_c_finale']:.4g}",
        "",
        "## Alertes",
    ]
    lignes += [f"- {a}" for a in liste_alertes] if liste_alertes else ["- Aucune."]
    return "\n".join(lignes) + "\n"


# --- Graphiques ---------------------------------------------------------------

def tableau_de_bord(e: Entrainement, chemin_image: Path, lissage: int | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    j = e.journal
    x = j["parties"]
    w = fenetre_lissage(len(j), lissage)
    d = e.reglage("contrainte", "lagrangien", "seuil_danger")
    seuil = e.reglage("stockfish", "curriculum", "seuil_montee")
    montees = x[j["niveau"].diff().fillna(0) > 0]

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRILLE, "axes.labelcolor": ENCRE_SECONDAIRE,
        "xtick.color": ENCRE_SECONDAIRE, "ytick.color": ENCRE_SECONDAIRE, "axes.titlecolor": ENCRE,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.6,
        "figure.facecolor": FOND, "axes.facecolor": FOND, "legend.frameon": False,
    })
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    fig.suptitle(f"Entraînement {e.dossier.name}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=ENCRE)

    def courbe(ax, col, titre, couleur=BLEU, etiquette=None):
        ax.plot(x, j[col], color=couleur, linewidth=1, alpha=0.3)
        ax.plot(x, lisser(j[col], w), color=couleur, linewidth=2, label=etiquette)
        ax.set_title(titre)

    def repere(ax, y, texte):
        ax.axhline(y, color=REPERE, linewidth=1, linestyle="--")
        ax.annotate(texte, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8, color=ENCRE_SECONDAIRE)

    def montees_de_niveau(ax):
        for xm in montees:
            ax.axvline(xm, color=REPERE, linewidth=0.8, linestyle=":")

    ax = axes[0, 0]
    courbe(ax, "z_moyen", "Résultat moyen z")
    repere(ax, 0, "0")
    if seuil not in (None, 0):
        repere(ax, seuil, f"seuil de montée ({seuil})")
    montees_de_niveau(ax)
    ax.set_ylim(-1.05, 1.05)

    ax = axes[0, 1]
    ax.stackplot(x, 100 * j["pourcentage_victoire"], 100 * j["pourcentage_nulle"], 100 * j["pourcentage_defaite"],
                 colors=[BLEU, GRIS_NULLE, ORANGE], labels=["victoires", "nulles", "défaites"],
                 edgecolor=FOND, linewidth=1)
    ax.set_title("Victoires, nulles, défaites (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=8, borderaxespad=0.2)
    montees_de_niveau(ax)

    ax = axes[1, 0]
    courbe(ax, "echecs_subis", "Échecs subis par partie")
    if d is not None:
        repere(ax, d, f"budget d = {d}")
    ax.set_ylim(bottom=0)

    ax = axes[1, 1]
    courbe(ax, "lambda", "λ (prix du danger)")
    ax.set_ylim(bottom=0)

    ax = axes[2, 0]
    ax.plot(x, j["epsilon"], color=BLEU, linewidth=2)
    ax.set_title("ε (exploration)")
    repere(ax, EPSILON_EXPLORATION, f"ε = {EPSILON_EXPLORATION}")
    ax.set_ylim(0, 1.05)

    ax = axes[2, 1]
    for col, couleur, nom in (("erreur_r", BLEU, "retour (Q_R)"), ("erreur_c", ORANGE, "danger (Q_C)")):
        valeurs = j[col].where(j[col] > 0)
        ax.plot(x, valeurs, color=couleur, linewidth=1, alpha=0.3)
        ax.plot(x, lisser(valeurs, w), color=couleur, linewidth=2, label=nom)
    ax.set_yscale("log")
    ax.set_title("Erreurs d'apprentissage (échelle log)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=8, borderaxespad=0.2)

    ax = axes[3, 0]
    courbe(ax, "longueur_moy_partie", "Longueur moyenne des parties (coups de nos agents)")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Parties jouées")

    ax = axes[3, 1]
    ax.step(x, j["niveau"], where="post", color=BLEU, linewidth=2)
    ax.set_title("Niveau de l'adversaire (Skill Level)")
    ax.set_ylim(-0.5, max(1, int(j["niveau"].max())) + 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Parties jouées")
    ax.annotate(f"durée totale : {duree_lisible(float(j['duree'].sum()))}", xy=(0.01, 0.92),
                xycoords="axes fraction", fontsize=8, color=ENCRE_SECONDAIRE)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    chemin_image = Path(chemin_image)
    chemin_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin_image, dpi=130, facecolor=FOND)
    plt.close(fig)
    return chemin_image