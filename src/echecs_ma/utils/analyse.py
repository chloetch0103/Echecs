"""Analyse du journal d'un entraînement : résumé chiffré, alertes et tableau de bord.

Le journal (journal.csv) contient une ligne toutes les `journal.frequence` parties,
avec les colonnes écrites par scripts/entrainer.py. La config copiée dans le dossier
de l'entraînement (config.yaml) fournit les seuils (budget de danger d, seuil de
montée du curriculum).

Le fichier par partie (parties.csv), quand il existe, sert aux statistiques de la
phase d'exploitation (ε au plancher) : une partie par ligne, donc des moyennes sur
toutes les parties de la phase et des intervalles de confiance par bootstrap.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

COLONNES = ["parties", "z_moyen", "echecs_subis", "lambda", "niveau", "epsilon", "erreur_r", "erreur_c",
            "pourcentage_victoire", "pourcentage_nulle", "pourcentage_defaite", "longueur_moy_partie", "duree",
            "bilan_mat_moy"]

# Couleurs (palette de référence, mode clair) : identité des séries, encres et repères
BLEU, ORANGE, GRIS_NULLE = "#2a78d6", "#eb6834", "#b5b3ad"
ENCRE, ENCRE_SECONDAIRE, REPERE, GRILLE, FOND = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

EPSILON_EXPLORATION = 0.2     # au-dessus, les coups sont surtout tirés au hasard

# Raisons de fin de partie : le journal a une colonne fin_<raison> par membre de RaisonFinPartie
# (proportion des parties du bloc). Pour la lecture, on regroupe les raisons de même nature ;
# une colonne fin_* qui n'est dans aucun groupe (variantes d'échecs...) tombe dans « autre ».
PREFIXE_FIN = "fin_"
GROUPES_FIN = [  # (libellé, colonnes, couleur), dans l'ordre d'empilement
    ("mat", ["fin_checkmate"], BLEU),
    ("pat", ["fin_stalemate"], "#8a5cd0"),
    ("matériel insuffisant", ["fin_insufficient_material"], GRIS_NULLE),
    ("répétition", ["fin_threefold_repetition", "fin_fivefold_repetition"], ORANGE),
    ("règle des 50 coups", ["fin_fifty_moves", "fin_seventyfive_moves"], "#1f9e89"),
    ("partie trop longue", ["fin_partie_trop_longue"], ENCRE_SECONDAIRE),
]
COULEUR_AUTRE_FIN = "#dcdad3"
SEUIL_REPETITIONS = 0.5       # au-delà, les agents tournent en rond

# Fichier par partie (écrit par scripts/entrainer.py)
COLONNES_PARTIES = ["partie", "z", "raison_fin", "nb_coups", "bilan_mat", "echecs_subis", "epsilon"]
NB_TIRAGES_BOOTSTRAP = 2000
NIVEAU_CONFIANCE = 0.95


# --- Chargement ---------------------------------------------------------------

@dataclass
class Entrainement:
    dossier: Path
    journal: pd.DataFrame
    config: dict = field(default_factory=dict)
    parties: pd.DataFrame | None = None     # parties.csv, absent pour les entraînements plus anciens

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
        nom_parties = config.get("journal", {}).get("csv_parties", "parties.csv")
        csvs = sorted(c for c in dossier.glob("*.csv") if c.name != nom_parties)
        if not csvs:
            raise FileNotFoundError(f"Aucun journal .csv dans {dossier}")
        chemin_csv = csvs[0]
    journal = pd.read_csv(chemin_csv)
    manquantes = [c for c in COLONNES if c not in journal.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes du journal : {', '.join(manquantes)}")
    if journal.empty:
        raise ValueError("Le journal est vide : l'entraînement s'est arrêté avant la première écriture.")
    return Entrainement(dossier, journal, config, charger_parties(dossier, config))


def charger_parties(dossier: Path, config: dict) -> pd.DataFrame | None:
    """Le fichier par partie s'il existe et contient les colonnes utiles, sinon None (ancien entraînement)."""
    chemin = Path(dossier) / config.get("journal", {}).get("csv_parties", "parties.csv")
    if not chemin.is_file():
        return None
    parties = pd.read_csv(chemin)
    if parties.empty or any(c not in parties.columns for c in COLONNES_PARTIES):
        return None
    return parties


# --- Calculs ------------------------------------------------------------------

def fenetre_lissage(nb_lignes: int, demande: int | None = None) -> int:
    """Nombre de lignes de la moyenne glissante : environ 1/20 du journal, au moins 1."""
    if demande:
        return max(1, demande)
    return max(1, nb_lignes // 20)


def lisser(serie: pd.Series, fenetre: int) -> pd.Series:
    return serie.rolling(fenetre, min_periods=1).mean()


def lisser_mediane(serie: pd.Series, fenetre: int) -> pd.Series:
    """Médiane glissante : insensible à une valeur aberrante isolée (ex. une mise en veille du PC)."""
    return serie.rolling(fenetre, min_periods=1).median()


def fins_regroupees(journal: pd.DataFrame) -> pd.DataFrame:
    """
    Proportion des parties de chaque bloc par raison de fin, regroupées pour la lecture.
    :return: une colonne par groupe présent dans le journal (libellés de GROUPES_FIN, plus « autre »),
             DataFrame sans colonnes si le journal est antérieur à ce suivi
    """
    colonnes_fin = [c for c in journal.columns if c.startswith(PREFIXE_FIN)]
    groupes = {}
    rangees = set()
    for libelle, colonnes, _ in GROUPES_FIN:
        presentes = [c for c in colonnes if c in colonnes_fin]
        if presentes:
            groupes[libelle] = journal[presentes].sum(axis=1)
            rangees.update(presentes)
    autres = [c for c in colonnes_fin if c not in rangees]
    if autres:
        groupes["autre"] = journal[autres].sum(axis=1)
    return pd.DataFrame(groupes, index=journal.index)


def couleur_fin(libelle: str) -> str:
    return next((couleur for nom, _, couleur in GROUPES_FIN if nom == libelle), COULEUR_AUTRE_FIN)


def fin_du_journal(journal: pd.DataFrame) -> pd.DataFrame:
    """Les 10 % dernières lignes (au moins une) : l'état « final » de l'entraînement."""
    return journal.tail(max(1, len(journal) // 10))


def debut_du_journal(journal: pd.DataFrame) -> pd.DataFrame:
    """Les 10 % premières lignes (au moins une) : le point de départ, pour comparer."""
    return journal.head(max(1, len(journal) // 10))


def lagrangien_actif(e: Entrainement) -> bool:
    """Le lagrangien est considéré actif si la config ne dit pas le contraire."""
    return bool(e.reglage("contrainte", "lagrangien", "actif", defaut=True))


def parties_par_niveau(journal: pd.DataFrame) -> dict[int, int]:
    """Nombre de parties jouées à chaque niveau de l'adversaire."""
    taille_blocs = journal["parties"].diff().fillna(journal["parties"].iloc[0])
    return {int(n): int(t) for n, t in taille_blocs.groupby(journal["niveau"]).sum().items()}


def pente_par_100_parties(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x.to_numpy(float), y.to_numpy(float), 1)[0] * 100)


# --- Statistiques par partie (parties.csv) ------------------------------------

@dataclass(frozen=True)
class Estimation:
    """Une moyenne sur des parties et son intervalle de confiance."""
    valeur: float
    bas: float
    haut: float

    def contient(self, x: float) -> bool:
        return self.bas <= x <= self.haut


def intervalle_bootstrap(valeurs, rng: np.random.Generator, nb_tirages: int = NB_TIRAGES_BOOTSTRAP,
                         niveau: float = NIVEAU_CONFIANCE) -> Estimation:
    """
    Moyenne d'un échantillon et son intervalle de confiance par bootstrap (méthode des percentiles).
    Pour une proportion, passer un tableau de 0 et de 1 (ex. 1 si la partie est gagnée).
    :param valeurs: (n,) une valeur par partie
    :return: Estimation(moyenne, borne basse, borne haute) ; nan si l'échantillon est vide
    """
    valeurs = np.asarray(valeurs, dtype=float)
    n = len(valeurs)
    if n == 0:
        return Estimation(float("nan"), float("nan"), float("nan"))
    # Chaque tirage : n parties tirées avec remise, dont on prend la moyenne (une moyenne par ligne)
    # Par paquets pour limiter la mémoire : nb_tirages x n peut être grand.
    moyennes = []
    for debut in range(0, nb_tirages, 500):
        k = min(500, nb_tirages - debut)
        indices = rng.integers(0, n, size=(k, n))
        moyennes.append(valeurs[indices].mean(axis=1))
    moyennes = np.concatenate(moyennes)
    alpha = (1 - niveau) / 2
    return Estimation(float(valeurs.mean()), float(np.quantile(moyennes, alpha)),
                      float(np.quantile(moyennes, 1 - alpha)))


def intervalle_difference(avant, apres, rng: np.random.Generator, nb_tirages: int = NB_TIRAGES_BOOTSTRAP,
                          niveau: float = NIVEAU_CONFIANCE) -> Estimation:
    """Différence des moyennes (après − avant) de deux échantillons indépendants, avec son intervalle."""
    avant, apres = np.asarray(avant, dtype=float), np.asarray(apres, dtype=float)
    if len(avant) == 0 or len(apres) == 0:
        return Estimation(float("nan"), float("nan"), float("nan"))
    differences = []
    for debut in range(0, nb_tirages, 500):
        k = min(500, nb_tirages - debut)
        m_avant = avant[rng.integers(0, len(avant), size=(k, len(avant)))].mean(axis=1)
        m_apres = apres[rng.integers(0, len(apres), size=(k, len(apres)))].mean(axis=1)
        differences.append(m_apres - m_avant)
    differences = np.concatenate(differences)
    alpha = (1 - niveau) / 2
    return Estimation(float(apres.mean() - avant.mean()), float(np.quantile(differences, alpha)),
                      float(np.quantile(differences, 1 - alpha)))


def phase_exploitation(parties: pd.DataFrame, epsilon_final: float | None = None) -> pd.DataFrame:
    """
    Les parties jouées une fois ε arrivé à son plancher : celles qui décrivent la politique apprise,
    et non le hasard de l'exploration.
    :param epsilon_final: le plancher (agents.exploration.epsilon_final) ; par défaut le plus petit ε du fichier
    """
    plancher = parties["epsilon"].min() if epsilon_final is None else epsilon_final
    return parties[parties["epsilon"] <= plancher + 1e-9]


def groupe_de_fin(raison: str) -> str:
    """Nom d'une RaisonFinPartie (ex. "THREEFOLD_REPETITION") → libellé de son groupe (ex. « répétition »)."""
    colonne = PREFIXE_FIN + str(raison).lower()
    return next((libelle for libelle, colonnes, _ in GROUPES_FIN if colonne in colonnes), "autre")


def statistiques_parties(parties: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Résultats, matériel, échecs, longueur et fins de partie d'un ensemble de parties, avec intervalles."""
    z = parties["z"].to_numpy(float)
    groupes = parties["raison_fin"].map(groupe_de_fin)
    fins = groupes.value_counts(normalize=True)
    return {
        "nb_parties": len(parties),
        "premiere": int(parties["partie"].min()),
        "derniere": int(parties["partie"].max()),
        "victoires": intervalle_bootstrap(z == 1, rng),
        "nulles": intervalle_bootstrap((z != 1) & (z != -1), rng),
        "defaites": intervalle_bootstrap(z == -1, rng),
        "z": intervalle_bootstrap(z, rng),
        "bilan": intervalle_bootstrap(parties["bilan_mat"], rng),
        "echecs": intervalle_bootstrap(parties["echecs_subis"], rng),
        "longueur": intervalle_bootstrap(parties["nb_coups"], rng),
        "fins": {nom: float(p) for nom, p in fins.items()},
    }


def resume_exploitation(e: Entrainement) -> dict | None:
    """
    Statistiques de la phase d'exploitation : sur l'ensemble, et évolution entre sa première
    et sa seconde moitié (la politique progresse-t-elle encore ?). None sans parties.csv.
    """
    if e.parties is None:
        return None
    phase = phase_exploitation(e.parties, e.reglage("agents", "exploration", "epsilon_final"))
    if phase.empty:
        return None
    rng = np.random.default_rng(e.reglage("experience", "graine", defaut=0))   # résultats reproductibles
    res = {"epsilon": float(phase["epsilon"].iloc[0]), "ensemble": statistiques_parties(phase, rng)}
    moitie = len(phase) // 2
    if moitie >= 10:
        avant, apres = phase.iloc[:moitie], phase.iloc[moitie:]
        res["evolution"] = {
            "bilan": intervalle_difference(avant["bilan_mat"], apres["bilan_mat"], rng),
            "z": intervalle_difference(avant["z"], apres["z"], rng),
            "victoires": intervalle_difference(avant["z"] == 1, apres["z"] == 1, rng),
        }
    return res


# --- Comparaison de deux entraînements ---------------------------------------

# (clé, libellé, unité, fonction qui donne une valeur par partie, nombre de décimales)
INDICATEURS_COMPARAISON = [
    ("bilan", "Bilan matériel en fin de partie", "points", lambda p: p["bilan_mat"], 2),
    ("victoires", "Victoires", "%", lambda p: p["z"] == 1, 1),
    ("nulles", "Nulles", "%", lambda p: (p["z"] != 1) & (p["z"] != -1), 1),
    ("defaites", "Défaites", "%", lambda p: p["z"] == -1, 1),
    ("z", "Résultat moyen z", "", lambda p: p["z"], 3),
    ("echecs", "Échecs subis par partie", "", lambda p: p["echecs_subis"], 2),
    ("longueur", "Longueur des parties", "coups", lambda p: p["nb_coups"], 1),
]


def parties_d_exploitation(e: Entrainement) -> pd.DataFrame:
    if e.parties is None:
        raise ValueError(f"Pas de parties.csv dans {e.dossier} : comparaison impossible (entraînement trop ancien ?).")
    phase = phase_exploitation(e.parties, e.reglage("agents", "exploration", "epsilon_final"))
    if phase.empty:
        raise ValueError(f"Aucune partie jouée avec ε au plancher dans {e.dossier}.")
    return phase


def comparer(essai: Entrainement, temoin: Entrainement, graine: int = 0) -> dict:
    """
    Compare la phase d'exploitation de deux entraînements, indicateur par indicateur.
    Différence = essai − témoin, avec son intervalle de confiance par bootstrap (parties indépendantes).
    """
    rng = np.random.default_rng(graine)
    p_essai, p_temoin = parties_d_exploitation(essai), parties_d_exploitation(temoin)
    indicateurs = {}
    for cle, _, _, valeurs, _ in INDICATEURS_COMPARAISON:
        a, b = np.asarray(valeurs(p_essai), dtype=float), np.asarray(valeurs(p_temoin), dtype=float)
        indicateurs[cle] = {"essai": float(a.mean()), "temoin": float(b.mean()),
                            "difference": intervalle_difference(b, a, rng)}     # a − b
    return {"essai": essai.dossier.name, "temoin": temoin.dossier.name,
            "nb_essai": len(p_essai), "nb_temoin": len(p_temoin), "indicateurs": indicateurs}



def formater(x: float, echelle: float, chiffres: int, signe: bool = False) -> str:
    return f"{echelle * x:{'+' if signe else ''}.{chiffres}f}"


def comparaison_en_texte(comp: dict) -> str:
    lignes = [
        f"# Comparaison : {comp['essai']} (essai) contre {comp['temoin']} (témoin)",
        "",
        (f"Phase d'exploitation (ε au plancher) : {comp['nb_essai']} parties pour l'essai, "
         f"{comp['nb_temoin']} pour le témoin."),
        (f"Différence = essai − témoin, intervalle de confiance à {100 * NIVEAU_CONFIANCE:.0f} % par bootstrap. "
         "« Significatif » : l'intervalle exclut 0. Pour les %, l'écart est en points de pourcentage (pts)."),
        "",
        "| Indicateur | Essai | Témoin | Différence [IC] | |",
        "|---|---|---|---|---|",
    ]
    for cle, libelle, unite, _, chiffres in INDICATEURS_COMPARAISON:
        ind = comp["indicateurs"][cle]
        echelle = 100 if unite == "%" else 1
        d = ind["difference"]

        suffixe = " pts" if unite == "%" else (f" {unite}" if unite else "")   # écart de % = points de %
        verdict = "non significatif" if d.contient(0) else "**significatif**"

        lignes.append(f"| {libelle}{f' ({unite})' if unite else ''} | "
                      f"{formater(ind['essai'], echelle, chiffres)} | {formater(ind['temoin'], echelle, chiffres)} | "
                      f"{formater(d.valeur, echelle, chiffres, True)}{suffixe} "
                      f"[{formater(d.bas, echelle, chiffres, True)} ; {formater(d.haut, echelle, chiffres, True)}] "
                      f"| {verdict} |")
    lignes += [
        "",
        ("Limite : l'intervalle mesure le hasard d'une partie à l'autre, pas celui d'un entraînement à l'autre. "
         "Pour généraliser, répéter la comparaison sur plusieurs graines."),
    ]
    return "\n".join(lignes) + "\n"


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
        "bilan_materiel_fin_partie": float(fin["bilan_mat_moy"].mean()),
        "bilan_materiel_debut": float(debut_du_journal(j)["bilan_mat_moy"].mean()),
        "meilleur_z_lisse": float(lisser(j["z_moyen"], w).max()),
        "victoires_final": float(fin["pourcentage_victoire"].mean()),
        "nulles_final": float(fin["pourcentage_nulle"].mean()),
        "defaites_final": float(fin["pourcentage_defaite"].mean()),
        "echecs_final": float(fin["echecs_subis"].mean()),
        "budget_d": e.reglage("contrainte", "lagrangien", "seuil_danger"),
        "lagrangien_actif": lagrangien_actif(e),
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
    fins = fins_regroupees(fin)
    if not fins.empty:
        moyennes = fins.mean()
        res["fins_final"] = {nom: float(v) for nom, v in moyennes.sort_values(ascending=False).items() if v > 0}
    res["parties_fin_journal"] = int(fin["parties"].iloc[-1] - (j["parties"].iloc[-len(fin) - 1]
                                                               if len(fin) < len(j) else 0))
    exploitation = resume_exploitation(e)
    if exploitation is not None:
        res["exploitation"] = exploitation
    if estimees and parties and int(estimees) != parties:     # inutile une fois l'entraînement terminé
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
    if lagrangien_actif(e) and len(exploration) >= 3:
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
    if (lagrangien_actif(e) and d is not None and fin["epsilon"].mean() <= EPSILON_EXPLORATION
            and fin["echecs_subis"].mean() > d):
        messages.append(f"Budget de danger dépassé en fin d'entraînement : {fin['echecs_subis'].mean():.1f} "
                        f"échecs subis par partie pour d = {d}.")

    if fin["pourcentage_nulle"].mean() > 0.9:
        messages.append("Plus de 90 % de nulles en fin d'entraînement : les agents ne parviennent pas à conclure.")

    fins = fins_regroupees(fin)
    if "répétition" in fins and fins["répétition"].mean() > SEUIL_REPETITIONS:
        messages.append(f"{100 * fins['répétition'].mean():.0f} % des parties finissent par répétition en fin "
                        "d'entraînement : les agents tournent en rond au lieu de chercher à conclure.")

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


def pourcentage_avec_intervalle(est: Estimation) -> str:
    return f"{100 * est.valeur:.0f} % [{100 * est.bas:.0f} ; {100 * est.haut:.0f}]"


def nombre_avec_intervalle(est: Estimation, chiffres: int = 2, signe: bool = True) -> str:
    f = f"{'+' if signe else ''}.{chiffres}f"
    return f"{est.valeur:{f}} [{est.bas:{f}} ; {est.haut:{f}}]"


def verdict_evolution(est: Estimation, unite: str) -> str:
    if est.contient(0):
        return "pas d'évolution démontrée (l'intervalle contient 0)"
    sens = "hausse" if est.valeur > 0 else "baisse"
    return f"{sens} significative de {abs(est.valeur):.2f} {unite}".rstrip()


def texte_exploitation(ex: dict) -> list[str]:
    s = ex["ensemble"]
    pct_ic = pourcentage_avec_intervalle
    lignes = [
        (f"## Phase d'exploitation (ε = {ex['epsilon']:.3g} : {s['nb_parties']} parties, "
         f"n° {s['premiere']} à {s['derniere']})"),
        (f"Valeur moyenne [intervalle de confiance à {100 * NIVEAU_CONFIANCE:.0f} %, bootstrap], "
         "sur toutes les parties de la phase."),
        "",
        f"- Victoires : {pct_ic(s['victoires'])}",
        f"- Nulles : {pct_ic(s['nulles'])}",
        f"- Défaites : {pct_ic(s['defaites'])}",
        f"- Résultat moyen z : {nombre_avec_intervalle(s['z'], 3)}",
        f"- Bilan matériel en fin de partie : {nombre_avec_intervalle(s['bilan'])} points",
        f"- Échecs subis par partie : {nombre_avec_intervalle(s['echecs'], 1, signe=False)}",
        f"- Longueur des parties : {nombre_avec_intervalle(s['longueur'], 0, signe=False)} coups",
    ]
    if s["fins"]:
        lignes.append("- Fins de partie : " + ", ".join(f"{nom} {100 * p:.0f} %" for nom, p in s["fins"].items()))
    if "evolution" in ex:
        ev = ex["evolution"]
        lignes += [
            "",
            "Évolution entre la première et la seconde moitié de la phase (seconde − première) :",
            (f"- Bilan matériel : {nombre_avec_intervalle(ev['bilan'])} points, "
             f"{verdict_evolution(ev['bilan'], 'points')}"),
            f"- Résultat moyen z : {nombre_avec_intervalle(ev['z'], 3)}, {verdict_evolution(ev['z'], '')}",
            (f"- Victoires : {100 * ev['victoires'].valeur:+.0f} points de % "
             f"[{100 * ev['victoires'].bas:+.0f} ; {100 * ev['victoires'].haut:+.0f}]"),
        ]
    return lignes


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
    ]
    if "exploitation" in res:
        lignes += texte_exploitation(res["exploitation"]) + [""]
    lignes += [
        f"## Fin de l'entraînement, d'après le journal ({res['parties_fin_journal']} dernières parties)",
        f"- Résultat moyen z : {res['z_final']:+.3f} (meilleur z lissé : {res['meilleur_z_lisse']:+.3f})",
        (f"- Bilan matériel moyen en fin de partie : {res['bilan_materiel_fin_partie']:+.2f} points "
         f"(contre {res['bilan_materiel_debut']:+.2f} au début de l'entraînement ; 1 point = un pion)"),
        (f"- Victoires / nulles / défaites : {pct(res['victoires_final'])} / {pct(res['nulles_final'])} / "
        f"{pct(res['defaites_final'])}"),
        f"- Longueur moyenne des parties : {res['longueur_finale']:.1f} coups",
    ]
    if res.get("fins_final"):
        lignes.append("- Fins de partie : " + ", ".join(f"{nom} {pct(v)}" for nom, v in res["fins_final"].items()))
    if not res["lagrangien_actif"]:
        budget = " (lagrangien désactivé)"
    elif res["budget_d"] is not None:
        budget = f" pour un budget d = {res['budget_d']}"
    else:
        budget = ""
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
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    j = e.journal
    x = j["parties"]
    w = fenetre_lissage(len(j), lissage)
    d = e.reglage("contrainte", "lagrangien", "seuil_danger") if lagrangien_actif(e) else None
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
    fins = fins_regroupees(j)
    fins = fins.loc[:, fins.sum() > 0]          # seulement les raisons rencontrées au moins une fois
    nb_rangees = 6 if not fins.empty else 5     # une rangée pleine largeur pour les fins de partie
    fig, axes = plt.subplots(nb_rangees, 2, figsize=(13, 3.4 * nb_rangees), sharex=True)
    fig.suptitle(f"Entraînement {e.dossier.name}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=ENCRE)

    # Légende commune : le trait clair est la valeur brute de chaque ligne du journal,
    # le trait foncé sa moyenne glissante (même couleur, la série reste identifiable)
    frequence = e.reglage("journal", "frequence")
    brute = "valeur brute" + (f" (toutes les {frequence} parties)" if frequence else "")
    fig.legend(handles=[Line2D([], [], color=BLEU, linewidth=1, alpha=0.3, label=brute),
                        Line2D([], [], color=BLEU, linewidth=2, label=f"moyenne glissante sur {w} lignes")],
               loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=2, fontsize=9, frameon=False)

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
    courbe(ax, "bilan_mat_moy", "Bilan matériel en fin de partie (points, 1 = un pion)")
    repere(ax, 0, "égalité")

    ax = axes[1, 1]
    courbe(ax, "longueur_moy_partie", "Longueur moyenne des parties (coups de nos agents)")
    ax.set_ylim(bottom=0)

    ax = axes[2, 0]
    courbe(ax, "echecs_subis", "Échecs subis par partie")
    if d is not None:
        repere(ax, d, f"budget d = {d}")
    ax.set_ylim(bottom=0)

    ax = axes[2, 1]
    courbe(ax, "lambda", "λ (prix du danger)" + ("" if lagrangien_actif(e) else " — lagrangien désactivé"))
    ax.set_ylim(bottom=0)

    ax = axes[3, 0]
    ax.plot(x, j["epsilon"], color=BLEU, linewidth=2)
    ax.set_title("ε (exploration)")
    repere(ax, EPSILON_EXPLORATION, f"ε = {EPSILON_EXPLORATION}")
    ax.set_ylim(0, 1.05)

    ax = axes[3, 1]
    for col, couleur, nom in (("erreur_r", BLEU, "retour (Q_R)"), ("erreur_c", ORANGE, "danger (Q_C)")):
        valeurs = j[col].where(j[col] > 0)
        ax.plot(x, valeurs, color=couleur, linewidth=1, alpha=0.3)
        ax.plot(x, lisser(valeurs, w), color=couleur, linewidth=2, label=nom)
    ax.set_yscale("log")
    ax.set_title("Erreurs d'apprentissage (échelle log)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=8, borderaxespad=0.2)

    ax = axes[4, 0]
    ax.step(x, j["niveau"], where="post", color=BLEU, linewidth=2)
    ax.set_title("Niveau de l'adversaire (Skill Level)")
    ax.set_ylim(-0.5, max(1, int(j["niveau"].max())) + 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax = axes[4, 1]
    parties_du_bloc = j["parties"].diff().fillna(j["parties"].iloc[0])
    secondes_par_partie = j["duree"] / parties_du_bloc
    ax.plot(x, secondes_par_partie, color=BLEU, linewidth=1, alpha=0.3)
    # médiane plutôt que moyenne : une mise en veille du PC ne déforme pas la courbe sur toute la fenêtre
    ax.plot(x, lisser_mediane(secondes_par_partie, w), color=BLEU, linewidth=2)
    ax.set_title("Durée par partie (s, médiane glissante)")
    ax.set_ylim(bottom=0)
    ax.annotate(f"durée totale : {duree_lisible(float(j['duree'].sum()))}", xy=(0.01, 0.92),
                xycoords="axes fraction", fontsize=8, color=ENCRE_SECONDAIRE)

    if not fins.empty:
        # Dernière rangée sur toute la largeur : les raisons de fin, empilées comme V/N/D
        grille = axes[5, 0].get_gridspec()
        for a in axes[5]:
            a.remove()
        ax = fig.add_subplot(grille[5, :], sharex=axes[0, 0])
        ax.stackplot(x, *(100 * fins[nom] for nom in fins.columns), colors=[couleur_fin(n) for n in fins.columns],
                     labels=list(fins.columns), edgecolor=FOND, linewidth=1)
        ax.set_title("Raisons de fin de partie (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(fins.columns), fontsize=8,
                  borderaxespad=0.2)
        montees_de_niveau(ax)
        ax.set_xlabel("Parties jouées")
    else:
        axes[4, 0].set_xlabel("Parties jouées")
        axes[4, 1].set_xlabel("Parties jouées")

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    chemin_image = Path(chemin_image)
    chemin_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin_image, dpi=130, facecolor=FOND)
    plt.close(fig)
    return chemin_image