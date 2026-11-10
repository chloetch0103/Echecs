"""Tests du module d'analyse des journaux d'entraînement."""
import math

import pandas as pd
import pytest
import yaml

from echecs_ma.utils.analyse import (
    COLONNES,
    alertes,
    charger,
    dernier_entrainement,
    duree_lisible,
    parties_par_niveau,
    resume,
    resume_en_texte,
    tableau_de_bord,
)


def ligne(parties, **valeurs):
    base = dict(parties=parties, z_moyen=0.0, echecs_subis=5.0, **{"lambda": 0.0}, niveau=0, epsilon=0.05,
                erreur_r=0.01, erreur_c=0.02, pourcentage_victoire=0.3, pourcentage_nulle=0.4,
                pourcentage_defaite=0.3, longueur_moy_partie=40.0, duree=10.0)
    base.update(valeurs)
    return base


def ecrire_entrainement(dossier, lignes, config=None):
    dossier.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lignes, columns=COLONNES).to_csv(dossier / "journal.csv", index=False)
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


def test_alerte_trop_de_nulles(tmp_path):
    lignes = [ligne(10 * k, pourcentage_victoire=0.0, pourcentage_nulle=1.0, pourcentage_defaite=0.0)
              for k in range(1, 6)]
    assert any("90 % de nulles" in a for a in alertes_de(tmp_path, lignes))


# --- Tableau de bord ----------------------------------------------------------

def test_tableau_de_bord_cree_une_image(tmp_path):
    lignes = [ligne(10 * k, niveau=0 if k < 6 else 1) for k in range(1, 11)]
    e = charger(ecrire_entrainement(tmp_path / "e", lignes, CONFIG))
    image = tableau_de_bord(e, tmp_path / "e" / "analyse" / "tableau.png")
    assert image.is_file() and image.stat().st_size > 10_000


def test_tableau_de_bord_avant_tout_apprentissage(tmp_path):
    # Erreurs toutes nulles (échelle log) et une seule ligne : pas de plantage
    e = charger(ecrire_entrainement(tmp_path / "e", [ligne(10, erreur_r=0.0, erreur_c=0.0)], CONFIG))
    assert tableau_de_bord(e, tmp_path / "t.png").is_file()