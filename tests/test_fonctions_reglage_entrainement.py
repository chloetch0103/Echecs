"""Tests des fonctions de réglage de l'entraînement : ε, λ et curriculum."""
import itertools

import pytest

from echecs_ma.entrainement.fonctions_utiles import calcul_epsilon, calcul_lambda, decision_curriculum

# =============================================================================
# calcul_epsilon : décroissance linéaire de 1,0 à 0,05 sur 5 000 parties
# =============================================================================

def epsilon(n, initial=1.0, final=0.05, parties=5000):
    return calcul_epsilon(final, initial, n, parties)


@pytest.mark.parametrize("n, attendu", [
    (0, 1.0),          # début : exploration totale
    (1000, 0.81),
    (2500, 0.525),     # mi-parcours
    (5000, 0.05),      # fin exacte de la décroissance
    (10000, 0.05),     # bien après : ε ne descend plus
])
def test_epsilon_valeurs(n, attendu):
    assert epsilon(n) == pytest.approx(attendu)


def test_epsilon_decroit_regulierement():
    valeurs = [epsilon(n) for n in range(0, 5001, 100)]
    ecarts = [a - b for a, b in itertools.pairwise(valeurs)]
    assert all(e == pytest.approx(ecarts[0]) for e in ecarts)


def test_epsilon_ne_descend_jamais_sous_la_valeur_finale():
    assert min(epsilon(n) for n in range(0, 20000, 37)) == pytest.approx(0.05)


def test_epsilon_reste_entre_les_deux_bornes():
    assert all(0.05 <= epsilon(n) <= 1.0 for n in range(0, 20000, 37))


def test_epsilon_autres_reglages():
    # De 0,5 à 0,1 sur 100 parties : 0,3 à la partie 50
    assert calcul_epsilon(0.1, 0.5, 50, 100) == pytest.approx(0.3)


# =============================================================================
# calcul_lambda : λ ← max(0, λ + η (échecs subis − d))
# =============================================================================

def test_lambda_augmente_si_trop_d_echecs():
    # 8 échecs pour un budget de 5 : λ augmente de 0,01 × 3
    assert calcul_lambda(0.2, 0.01, 5, 8) == pytest.approx(0.23)


def test_lambda_diminue_si_peu_d_echecs():
    assert calcul_lambda(0.2, 0.01, 5, 2) == pytest.approx(0.17)


def test_lambda_ne_bouge_pas_au_seuil():
    assert calcul_lambda(0.2, 0.01, 5, 5) == pytest.approx(0.2)


def test_lambda_ne_devient_jamais_negatif():
    # Récompenser le danger n'aurait pas de sens
    assert calcul_lambda(0.0, 0.01, 5, 0) == 0.0
    assert calcul_lambda(0.02, 0.01, 5, 0) == 0.0


def test_lambda_depuis_zero():
    assert calcul_lambda(0.0, 0.01, 5, 12) == pytest.approx(0.07)


def test_eta_nul_lambda_constant():
    assert calcul_lambda(0.4, 0.0, 5, 20) == pytest.approx(0.4)


def test_lambda_sur_plusieurs_parties():
    # Toujours 7 échecs pour un budget de 5 : +0,02 par partie
    lambda_ = 0.0
    for _ in range(10):
        lambda_ = calcul_lambda(lambda_, 0.01, 5, 7)
    assert lambda_ == pytest.approx(0.2)


# =============================================================================
# decision_curriculum
# =============================================================================

def monter(actif=True, z_moyen=0.3, seuil=0.0, nb=100, fenetre=100, niveau=5):
    return decision_curriculum(actif, z_moyen, seuil, nb, fenetre, niveau)


def test_monte_quand_toutes_les_conditions_sont_reunies():
    assert monter()


def test_ne_monte_pas_si_le_curriculum_est_inactif():
    assert not monter(actif=False)


def test_ne_monte_pas_si_le_resultat_est_sous_le_seuil():
    assert not monter(z_moyen=-0.2)


def test_ne_monte_pas_si_le_resultat_est_egal_au_seuil():
    # Il faut DÉPASSER le seuil
    assert not monter(z_moyen=0.0, seuil=0.0)


def test_ne_monte_pas_avant_d_avoir_une_fenetre_complete():
    # Après une montée la fenêtre est vidée : il faut de nouveau 100 parties
    assert not monter(nb=99)
    assert monter(nb=100)


@pytest.mark.parametrize("niveau, attendu", [(0, True), (19, True), (20, False)])
def test_niveau_maximal(niveau, attendu):
    assert monter(niveau=niveau) == attendu


def test_seuil_positif():
    assert monter(z_moyen=0.6, seuil=0.5)
    assert not monter(z_moyen=0.4, seuil=0.5)