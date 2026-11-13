"""Tests du module d'analyse des journaux d'entraînement."""
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from echecs_ma.utils.analyse import (
    COLONNES,
    Estimation,
    alertes,
    charger,
    comparaison_en_texte,
    comparer,
    dernier_entrainement,
    duree_lisible,
    fins_regroupees,
    groupe_de_fin,
    intervalle_bootstrap,
    intervalle_difference,
    lisser_mediane,
    parties_par_niveau,
    phase_exploitation,
    resume,
    resume_en_texte,
    resume_exploitation,
    tableau_de_bord,
)


def ligne(parties, **valeurs):
    base = dict(parties=parties, z_moyen=0.0, echecs_subis=5.0, **{"lambda": 0.0}, niveau=0, epsilon=0.05,
                erreur_r=0.01, erreur_c=0.02, pourcentage_victoire=0.3, pourcentage_nulle=0.4,
                pourcentage_defaite=0.3, longueur_moy_partie=40.0, duree=10.0, bilan_mat_moy=0.0)
    base.update(valeurs)
    return base


def ecrire_entrainement(dossier, lignes, config=None):
    dossier.mkdir(parents=True, exist_ok=True)
    # colonnes obligatoires d'abord, puis les éventuelles colonnes en plus (fin_*), dans l'ordre d'apparition
    en_plus = [c for valeurs in lignes for c in valeurs if c not in COLONNES]
    colonnes = COLONNES + list(dict.fromkeys(en_plus))
    pd.DataFrame(lignes, columns=colonnes).to_csv(dossier / "journal.csv", index=False)
    if config is not None:
        (dossier / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return dossier


CONFIG = {"contrainte": {"lagrangien": {"seuil_danger": 5}}, "entrainement": {"nb_parties": 1000},
          "stockfish": {"curriculum": {"seuil_montee": 0.0}}, "journal": {"csv_file_name": "journal.csv"}}


# --- Chargement ---------------------------------------------------------------

def test_charger_lit_journal_et_config(tmp_path):
    e = charger(ecrire_entrainement(tmp_path / "essai", [ligne(10), ligne(20)], CONFIG))
    assert len(e.journal) == 2
    assert e.reglage("contrainte", "lagrangien", "seuil_danger") == 5
    assert e.reglage("cle", "absente", defaut="x") == "x"


def test_charger_sans_config(tmp_path):
    e = charger(ecrire_entrainement(tmp_path / "essai", [ligne(10)]))
    assert e.config == {}


def test_colonne_manquante(tmp_path):
    dossier = tmp_path / "essai"
    dossier.mkdir()
    pd.DataFrame({"parties": [10]}).to_csv(dossier / "journal.csv", index=False)
    with pytest.raises(ValueError, match="Colonnes absentes"):
        charger(dossier)


def test_journal_vide(tmp_path):
    with pytest.raises(ValueError, match="vide"):
        charger(ecrire_entrainement(tmp_path / "essai", []))


def test_dernier_entrainement(tmp_path):
    import os
    import time
    ancien = ecrire_entrainement(tmp_path / "runs" / "essai_a", [ligne(10)])
    recent = ecrire_entrainement(tmp_path / "runs" / "essai_b", [ligne(10)])
    os.utime(ancien, (time.time() - 100, time.time() - 100))
    assert dernier_entrainement(tmp_path / "runs") == recent


# --- Résumé -------------------------------------------------------------------

def test_parties_par_niveau():
    j = pd.DataFrame([ligne(10), ligne(20), ligne(30, niveau=1), ligne(40, niveau=1), ligne(50, niveau=1)])
    assert parties_par_niveau(j) == {0: 20, 1: 30}


def test_resume_duree_et_estimation(tmp_path):
    lignes = [ligne(10 * k, duree=20.0) for k in range(1, 11)]          # 100 parties en 200 s
    res = resume(charger(ecrire_entrainement(tmp_path / "e", lignes, CONFIG)))
    assert res["parties"] == 100
    assert res["secondes_par_partie"] == pytest.approx(2.0)
    assert res["parties_estimees"] == 1000                               # lu dans la config
    assert res["duree_estimee_s"] == pytest.approx(2000)


def test_resume_fin_d_entrainement(tmp_path):
    # 20 lignes : la « fin » correspond aux 2 dernières
    lignes = [ligne(10 * k) for k in range(1, 19)] + [ligne(190, z_moyen=0.4), ligne(200, z_moyen=0.6)]
    res = resume(charger(ecrire_entrainement(tmp_path / "e", lignes, CONFIG)))
    assert res["z_final"] == pytest.approx(0.5)


def test_resume_en_texte(tmp_path):
    e = charger(ecrire_entrainement(tmp_path / "e", [ligne(10), ligne(20)], CONFIG))
    texte = resume_en_texte(e, resume(e), ["Une alerte."])
    assert "Parties jouées : 20" in texte
    assert "- Une alerte." in texte


@pytest.mark.parametrize("secondes, attendu", [(45, "45 s"), (125, "2 min 05 s"), (3725, "1 h 02 min")])
def test_duree_lisible(secondes, attendu):
    assert duree_lisible(secondes) == attendu


# --- Alertes ------------------------------------------------------------------

def alertes_de(tmp_path, lignes, config=CONFIG):
    return alertes(charger(ecrire_entrainement(tmp_path / "e", lignes, config)))


def test_aucune_alerte_sur_un_entrainement_sain(tmp_path):
    assert alertes_de(tmp_path, [ligne(10 * k) for k in range(1, 11)]) == []


def test_alerte_nan(tmp_path):
    assert any("non finies" in a for a in alertes_de(tmp_path, [ligne(10), ligne(20, erreur_r=math.nan)]))


def test_alerte_apprentissage_pas_commence(tmp_path):
    lignes = [ligne(10 * k, erreur_r=0.0, erreur_c=0.0) for k in range(1, 5)]
    assert any("pas commencé" in a for a in alertes_de(tmp_path, lignes))


def test_alerte_lambda_pendant_l_exploration(tmp_path):
    lignes = [ligne(10 * k, epsilon=0.9, **{"lambda": 0.5 * k}) for k in range(1, 6)]
    assert any("pendant l'exploration" in a for a in alertes_de(tmp_path, lignes))


def test_pas_d_alerte_lambda_si_lambda_stable(tmp_path):
    lignes = [ligne(10 * k, epsilon=0.9, **{"lambda": 0.5}) for k in range(1, 6)]
    assert not any("pendant l'exploration" in a for a in alertes_de(tmp_path, lignes))


def test_alerte_erreur_qui_explose(tmp_path):
    lignes = [ligne(10 * k) for k in range(1, 20)] + [ligne(200, erreur_r=5.0)]
    assert any("retour finit à plus de 10 fois" in a for a in alertes_de(tmp_path, lignes))


def test_alerte_budget_depasse_seulement_hors_exploration(tmp_path):
    exploitation = [ligne(10 * k, echecs_subis=9.0, epsilon=0.05) for k in range(1, 6)]
    exploration = [ligne(10 * k, echecs_subis=9.0, epsilon=0.9) for k in range(1, 6)]
    assert any("Budget de danger" in a for a in alertes_de(tmp_path, exploitation))
    assert not any("Budget de danger" in a for a in alertes_de(tmp_path, exploration))


def test_pas_d_alerte_budget_si_lagrangien_desactive(tmp_path):
    config = {**CONFIG, "contrainte": {"lagrangien": {"seuil_danger": 5, "actif": False}}}
    lignes = [ligne(10 * k, echecs_subis=9.0, epsilon=0.05) for k in range(1, 6)]
    assert not any("Budget de danger" in a for a in alertes_de(tmp_path, lignes, config))


def test_resume_bilan_materiel_debut_et_fin(tmp_path):
    # 20 lignes : début = 2 premières, fin = 2 dernières
    lignes = ([ligne(10, bilan_mat_moy=-1.0), ligne(20, bilan_mat_moy=0.0)]
              + [ligne(10 * k) for k in range(3, 19)]
              + [ligne(190, bilan_mat_moy=3.0), ligne(200, bilan_mat_moy=5.0)])
    res = resume(charger(ecrire_entrainement(tmp_path / "e", lignes, CONFIG)))
    assert res["bilan_materiel_debut"] == pytest.approx(-0.5)
    assert res["bilan_materiel_fin_partie"] == pytest.approx(4.0)


def test_resume_en_texte_lagrangien_desactive(tmp_path):
    config = {**CONFIG, "contrainte": {"lagrangien": {"seuil_danger": 5, "actif": False}}}
    e = charger(ecrire_entrainement(tmp_path / "e", [ligne(10), ligne(20)], config))
    assert "lagrangien désactivé" in resume_en_texte(e, resume(e), [])


def test_alerte_trop_de_nulles(tmp_path):
    lignes = [ligne(10 * k, pourcentage_victoire=0.0, pourcentage_nulle=1.0, pourcentage_defaite=0.0)
              for k in range(1, 6)]
    assert any("90 % de nulles" in a for a in alertes_de(tmp_path, lignes))


# --- Raisons de fin de partie -------------------------------------------------

def fins(**proportions):
    """Les 11 colonnes fin_* écrites par entrainer.py, à 0 sauf celles données."""
    noms = ["checkmate", "stalemate", "insufficient_material", "seventyfive_moves", "fivefold_repetition",
            "fifty_moves", "threefold_repetition", "variant_win", "variant_loss", "variant_draw",
            "partie_trop_longue"]
    return {f"fin_{n}": proportions.get(n, 0.0) for n in noms}


def test_fins_regroupees_additionne_les_raisons_de_meme_nature():
    j = pd.DataFrame([ligne(10, **fins(threefold_repetition=0.3, fivefold_repetition=0.1, fifty_moves=0.2,
                                       seventyfive_moves=0.1, checkmate=0.3))])
    g = fins_regroupees(j)
    assert g["répétition"].iloc[0] == pytest.approx(0.4)
    assert g["règle des 50 coups"].iloc[0] == pytest.approx(0.3)
    assert g["mat"].iloc[0] == pytest.approx(0.3)
    assert g.sum(axis=1).iloc[0] == pytest.approx(1.0)   # aucune partie perdue au regroupement


def test_fins_regroupees_range_l_inconnu_dans_autre():
    g = fins_regroupees(pd.DataFrame([ligne(10, **fins(variant_draw=0.2, checkmate=0.8))]))
    assert g["autre"].iloc[0] == pytest.approx(0.2)


def test_fins_regroupees_journal_ancien():
    # journal d'avant le suivi des fins : rien à regrouper, et le reste de l'analyse fonctionne
    assert fins_regroupees(pd.DataFrame([ligne(10)])).empty


def test_resume_sans_suivi_des_fins(tmp_path):
    e = charger(ecrire_entrainement(tmp_path / "e", [ligne(10), ligne(20)], CONFIG))
    assert "fins_final" not in resume(e)
    assert "Fins de partie" not in resume_en_texte(e, resume(e), [])


def test_resume_fins_triees_et_sans_zero(tmp_path):
    lignes = [ligne(10 * k, **fins(insufficient_material=0.5, checkmate=0.2, threefold_repetition=0.3))
              for k in range(1, 3)]
    e = charger(ecrire_entrainement(tmp_path / "e", lignes, CONFIG))
    res = resume(e)
    assert list(res["fins_final"]) == ["matériel insuffisant", "répétition", "mat"]
    assert "- Fins de partie : matériel insuffisant 50 %, répétition 30 %, mat 20 %" in resume_en_texte(e, res, [])


def test_alerte_repetitions(tmp_path):
    lignes = [ligne(10 * k, **fins(threefold_repetition=0.6, checkmate=0.4)) for k in range(1, 6)]
    assert any("tournent en rond" in a for a in alertes_de(tmp_path, lignes))


def test_pas_d_alerte_repetitions_minoritaires(tmp_path):
    lignes = [ligne(10 * k, **fins(threefold_repetition=0.3, checkmate=0.7)) for k in range(1, 6)]
    assert not any("tournent en rond" in a for a in alertes_de(tmp_path, lignes))


def test_lisser_mediane_ignore_une_valeur_aberrante():
    serie = pd.Series([2.0, 2.0, 3600.0, 2.0, 2.0])
    assert lisser_mediane(serie, 3).iloc[3] == pytest.approx(2.0)


# --- Tableau de bord ----------------------------------------------------------

def test_tableau_de_bord_cree_une_image(tmp_path):
    lignes = [ligne(10 * k, niveau=0 if k < 6 else 1) for k in range(1, 11)]
    e = charger(ecrire_entrainement(tmp_path / "e", lignes, CONFIG))
    image = tableau_de_bord(e, tmp_path / "e" / "analyse" / "tableau.png")
    assert image.is_file() and image.stat().st_size > 10_000


def test_tableau_de_bord_avec_les_fins_de_partie(tmp_path):
    lignes = [ligne(10 * k, **fins(checkmate=0.2, insufficient_material=0.5, threefold_repetition=0.3))
              for k in range(1, 11)]
    e = charger(ecrire_entrainement(tmp_path / "e", lignes, CONFIG))
    assert tableau_de_bord(e, tmp_path / "t.png").stat().st_size > 10_000


def test_tableau_de_bord_avant_tout_apprentissage(tmp_path):
    # Erreurs toutes nulles (échelle log) et une seule ligne : pas de plantage
    e = charger(ecrire_entrainement(tmp_path / "e", [ligne(10, erreur_r=0.0, erreur_c=0.0)], CONFIG))
    assert tableau_de_bord(e, tmp_path / "t.png").is_file()


# --- Phase d'exploitation (parties.csv) --------------------------------------

def partie(numero, z=0, epsilon=0.05, bilan=0.0, raison="INSUFFICIENT_MATERIAL", nb_coups=100, echecs=5):
    return {"partie": numero, "couleur": "blancs", "z": z, "raison_fin": raison, "nb_coups": nb_coups, "bilan_mat": bilan,
                "echecs_subis": echecs, "epsilon": epsilon, "niveau": 0}


def ecrire_parties(dossier, parties):
    pd.DataFrame(parties).to_csv(dossier / "parties.csv", index=False)


CONFIG_EXPLOITATION = {**CONFIG, "agents": {"exploration": {"epsilon_final": 0.05}}, "experience": {"graine": 1}}


def entrainement_avec_parties(tmp_path, parties, config=CONFIG_EXPLOITATION):
    dossier = ecrire_entrainement(tmp_path / "e", [ligne(10 * k) for k in range(1, 11)], config)
    ecrire_parties(dossier, parties)
    return charger(dossier)


def test_bootstrap_echantillon_constant():
    est = intervalle_bootstrap(np.full(50, 3.0), np.random.default_rng(0))
    assert est == Estimation(3.0, 3.0, 3.0)


def test_bootstrap_proportion_largeur_attendue():
    # 400 parties dont la moitié gagnées : écart-type ≈ 0,025, donc intervalle ≈ ±0,05
    est = intervalle_bootstrap(np.arange(400) % 2 == 0, np.random.default_rng(0))
    assert est.valeur == pytest.approx(0.5)
    assert est.contient(0.5)
    assert 0.07 < est.haut - est.bas < 0.13


def test_bootstrap_reproductible_avec_la_meme_graine():
    valeurs = np.random.default_rng(3).normal(size=100)
    assert intervalle_bootstrap(valeurs, np.random.default_rng(7)) == intervalle_bootstrap(valeurs,
                                                                                           np.random.default_rng(7))


def test_bootstrap_echantillon_vide():
    assert math.isnan(intervalle_bootstrap([], np.random.default_rng(0)).valeur)


def test_difference_nette_exclut_zero():
    rng = np.random.default_rng(0)
    est = intervalle_difference(rng.normal(0, 1, 200), rng.normal(2, 1, 200), rng)
    assert est.valeur == pytest.approx(2, abs=0.3)
    assert not est.contient(0)


def test_difference_nulle_contient_zero():
    rng = np.random.default_rng(0)
    est = intervalle_difference(rng.normal(0, 1, 200), rng.normal(0, 1, 200), rng)
    assert est.contient(0)


def test_phase_exploitation_ne_garde_que_le_plancher():
    parties = pd.DataFrame([partie(1, epsilon=1.0), partie(2, epsilon=0.5), partie(3), partie(4)])
    assert phase_exploitation(parties)["partie"].tolist() == [3, 4]
    assert phase_exploitation(parties, epsilon_final=0.05)["partie"].tolist() == [3, 4]


@pytest.mark.parametrize("raison, groupe", [("CHECKMATE", "mat"), ("THREEFOLD_REPETITION", "répétition"),
                                            ("FIVEFOLD_REPETITION", "répétition"),
                                            ("SEVENTYFIVE_MOVES", "règle des 50 coups"),
                                            ("PARTIE_TROP_LONGUE", "partie trop longue"), ("VARIANT_WIN", "autre")])
def test_groupe_de_fin(raison, groupe):
    assert groupe_de_fin(raison) == groupe


def test_charger_lit_parties_csv(tmp_path):
    e = entrainement_avec_parties(tmp_path, [partie(1), partie(2)])
    assert len(e.parties) == 2


def test_charger_sans_parties_csv(tmp_path):
    e = charger(ecrire_entrainement(tmp_path / "e", [ligne(10)], CONFIG))
    assert e.parties is None
    assert resume_exploitation(e) is None
    assert "exploitation" not in resume(e)


def test_exploitation_ignore_les_parties_d_exploration(tmp_path):
    # 100 parties d'exploration toutes gagnées, puis 200 parties d'exploitation dont 20 gagnées :
    # seules les secondes comptent
    exploration = [partie(n, z=1, epsilon=1.0) for n in range(1, 101)]
    exploitation = [partie(n, z=1 if n % 10 == 0 else 0) for n in range(101, 301)]
    ex = resume_exploitation(entrainement_avec_parties(tmp_path, exploration + exploitation))
    s = ex["ensemble"]
    assert s["nb_parties"] == 200 and (s["premiere"], s["derniere"]) == (101, 300)
    assert s["victoires"].valeur == pytest.approx(0.10)
    assert s["victoires"].contient(0.10) and s["victoires"].bas > 0.04
    assert s["nulles"].valeur == pytest.approx(0.90)
    assert s["defaites"].valeur == 0


def test_exploitation_compte_une_partie_coupee_comme_nulle(tmp_path):
    # Une partie trop longue a un z non entier (tanh du bilan) : ni victoire, ni défaite
    parties = [partie(1, z=0.3, raison="PARTIE_TROP_LONGUE"), partie(2, z=-1, raison="CHECKMATE")]
    s = resume_exploitation(entrainement_avec_parties(tmp_path, parties))["ensemble"]
    assert s["nulles"].valeur == pytest.approx(0.5) and s["defaites"].valeur == pytest.approx(0.5)
    assert s["fins"] == {"partie trop longue": 0.5, "mat": 0.5}


def test_evolution_detecte_une_progression_du_bilan(tmp_path):
    rng = np.random.default_rng(0)
    parties = ([partie(n, bilan=rng.normal(-3, 2)) for n in range(1, 101)]
               + [partie(n, bilan=rng.normal(1, 2)) for n in range(101, 201)])
    ev = resume_exploitation(entrainement_avec_parties(tmp_path, parties))["evolution"]
    assert ev["bilan"].valeur == pytest.approx(4, abs=0.8)
    assert not ev["bilan"].contient(0)


def test_pas_d_evolution_sur_trop_peu_de_parties(tmp_path):
    ex = resume_exploitation(entrainement_avec_parties(tmp_path, [partie(n) for n in range(1, 11)]))
    assert "evolution" not in ex


def test_texte_phase_exploitation(tmp_path):
    parties = [partie(n, z=1 if n % 10 == 0 else 0, raison="CHECKMATE" if n % 10 == 0 else "FIFTY_MOVES")
               for n in range(1, 201)]
    e = entrainement_avec_parties(tmp_path, parties)
    texte = resume_en_texte(e, resume(e), [])
    assert "## Phase d'exploitation (ε = 0.05 : 200 parties, n° 1 à 200)" in texte
    assert "- Victoires : 10 % [" in texte
    assert "- Fins de partie : règle des 50 coups 90 %, mat 10 %" in texte
    assert "pas d'évolution démontrée" in texte


def test_texte_titre_fin_du_journal_en_parties(tmp_path):
    e = charger(ecrire_entrainement(tmp_path / "e", [ligne(10 * k) for k in range(1, 31)], CONFIG))
    assert "d'après le journal (30 dernières parties)" in resume_en_texte(e, resume(e), [])


def test_pas_d_estimation_quand_l_entrainement_est_fini(tmp_path):
    config = {**CONFIG, "entrainement": {"nb_parties": 100}}
    e = charger(ecrire_entrainement(tmp_path / "e", [ligne(10 * k) for k in range(1, 11)], config))
    assert "Estimation" not in resume_en_texte(e, resume(e), [])


# --- Comparaison de deux entraînements ---------------------------------------

def run(dossier, parties):
    dossier = ecrire_entrainement(dossier, [ligne(10 * k) for k in range(1, 11)], CONFIG_EXPLOITATION)
    ecrire_parties(dossier, parties)
    return charger(dossier)


def test_comparer_detecte_un_ecart_de_bilan(tmp_path):
    rng = np.random.default_rng(0)
    essai = run(tmp_path / "essai", [partie(n, bilan=rng.normal(1, 3)) for n in range(1, 201)])
    temoin = run(tmp_path / "temoin", [partie(n, bilan=rng.normal(-3, 3)) for n in range(1, 201)])
    d = comparer(essai, temoin)["indicateurs"]["bilan"]["difference"]
    assert d.valeur == pytest.approx(4, abs=0.8)            # essai − témoin, donc positif
    assert not d.contient(0)


def test_comparer_deux_runs_identiques(tmp_path):
    parties = [partie(n, z=[1, 0, -1][n % 3], bilan=n % 5) for n in range(1, 101)]
    comp = comparer(run(tmp_path / "a", parties), run(tmp_path / "b", parties))
    for ind in comp["indicateurs"].values():
        assert ind["difference"].valeur == pytest.approx(0) and ind["difference"].contient(0)


def test_comparer_ignore_l_exploration(tmp_path):
    # Les parties d'exploration de l'essai sont toutes gagnées : elles ne doivent pas compter
    essai = run(tmp_path / "essai", [partie(n, z=1, epsilon=1.0) for n in range(1, 101)]
                + [partie(n) for n in range(101, 201)])
    temoin = run(tmp_path / "temoin", [partie(n) for n in range(1, 101)])
    comp = comparer(essai, temoin)
    assert comp["nb_essai"] == 100
    assert comp["indicateurs"]["victoires"]["essai"] == 0


def test_comparer_sans_parties_csv(tmp_path):
    ancien = charger(ecrire_entrainement(tmp_path / "ancien", [ligne(10)], CONFIG))
    recent = run(tmp_path / "recent", [partie(1)])
    with pytest.raises(ValueError, match="parties.csv"):
        comparer(recent, ancien)


def test_comparaison_en_texte(tmp_path):
    essai = run(tmp_path / "essai", [partie(n, z=-1 if n % 10 == 0 else 0, bilan=2.0) for n in range(1, 201)])
    temoin = run(tmp_path / "temoin", [partie(n, z=-1 if n % 5 == 0 else 0, bilan=-2.0) for n in range(1, 201)])
    texte = comparaison_en_texte(comparer(essai, temoin))
    assert "| Indicateur | Essai | Témoin | Différence [IC] | |" in texte
    assert ("| Bilan matériel en fin de partie (points) | 2.00 | -2.00 | +4.00 points [+4.00 ; +4.00] "
            "| **significatif** |") in texte
    assert "| Défaites (%) | 10.0 | 20.0 | -10.0 pts" in texte