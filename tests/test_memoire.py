"""Tests de la mémoire de rejeu : stockage, fin de partie, file circulaire et tirage des lots.

Les positions et décisions sont fabriquées à la main avec des valeurs repérables
(plans remplis avec un numéro, cases et types distincts), pour vérifier que chaque
donnée arrive au bon endroit."""
import chess
import numpy as np
import pytest

from echecs_ma.agents.arbitre import ArbitreDecision, ArbitrePositon
from echecs_ma.entrainement.memoire import NB_PIECES_MAX, Memoire
from echecs_ma.env.encodage import NB_PLANS, NB_SORTIES, Action

# --- Fabrication de données de test -------------------------------------------

def position(numero, nb_pieces=NB_PIECES_MAX):
    """Une position factice : plans remplis avec `numero`, et pour chaque pièce k
    une case 10 + k, un type 1 + k % 6 et un masque avec un seul True, à l'indice k."""
    masques = []
    for k in range(nb_pieces):
        m = np.zeros(NB_SORTIES, dtype=bool)
        m[k] = True
        masques.append(m)
    return ArbitrePositon(
        masques=masques,
        dicos=[{} for _ in range(nb_pieces)],
        pieces_t=[1 + k % 6 for k in range(nb_pieces)],
        cases_p=[10 + k for k in range(nb_pieces)],
        plans=np.full((NB_PLANS, 8, 8), numero, dtype=np.uint8),
    )


def decision(type_piece=chess.PAWN, case=chess.E2, indice=28):
    return ArbitreDecision(coup=chess.Move.from_uci("e2e4"), action=Action(type_piece, case, indice))


def memoire(taille=10, graine=0):
    return Memoire(taille, np.random.default_rng(graine))


def ajouter(m, numero, fin=False, nb_pieces_suiv=NB_PIECES_MAX):
    """Ajoute une transition repérée par `numero` (r = numero, plans de s = numero,
    plans de s' = numero + 100)."""
    suivante = None if fin else position(numero + 100, nb_pieces_suiv)
    m.stocker_en_memoire(float(numero), 0.5, position(numero), decision(), suivante)


def tout(m):
    """Un lot contenant exactement les transitions présentes, dans l'ordre des lignes."""
    return m.tirer_lot(m.taille_actuelle_memoire)


# --- Mémoire vide et compteurs ------------------------------------------------

def test_memoire_neuve_vide():
    m = memoire()
    assert m.taille_actuelle_memoire == 0
    assert m.compteur_transitions == 0
    with pytest.raises(ValueError):
        m.tirer_lot(1)


def test_pas_assez_de_transitions():
    m = memoire()
    ajouter(m, 1)
    ajouter(m, 2)
    with pytest.raises(ValueError):
        m.tirer_lot(3)


def test_compteurs_avant_que_la_memoire_soit_pleine():
    m = memoire(taille=10)
    for i in range(4):
        ajouter(m, i)
    assert m.compteur_transitions == 4
    assert m.taille_actuelle_memoire == 4
    assert m.pointeur_ecriture == 4


# --- Stockage d'une transition ------------------------------------------------

def test_stockage_d_une_transition():
    m = memoire()
    m.stocker_en_memoire(0.7, 1.0, position(3), decision(chess.KNIGHT, chess.G1, 21), position(103))

    assert m.memoire_r[0] == pytest.approx(0.7)
    assert m.memoire_c[0] == 1.0
    assert not m.memoire_indicateur_fin[0]
    assert (m.memoire_plans[0] == 3).all()
    assert (m.memoire_plans_suiv[0] == 103).all()
    assert m.memoire_type_action[0] == chess.KNIGHT
    assert m.memoire_case_dep[0] == chess.G1
    assert m.memoire_indice[0] == 21
    assert m.memoire_nb_pieces_suiv[0] == 16


def test_cases_et_types_de_s_prime_chacun_a_sa_place():
    # Les cases valent 10 + k et les types 1 + k % 6 : on vérifie qu'ils ne sont pas inversés
    m = memoire()
    ajouter(m, 1)
    assert m.memoire_cases_suiv[0].tolist() == [10 + k for k in range(16)]
    assert m.memoire_types_suiv[0].tolist() == [1 + k % 6 for k in range(16)]


def test_masques_de_s_prime():
    m = memoire()
    ajouter(m, 1)
    for k in range(16):
        assert np.flatnonzero(m.memoire_masques_suiv[0, k]).tolist() == [k]


def test_moins_de_16_pieces():
    # Avec 12 pièces, les 4 dernières places restent vides
    m = memoire()
    ajouter(m, 1, nb_pieces_suiv=12)
    assert m.memoire_nb_pieces_suiv[0] == 12
    assert m.memoire_cases_suiv[0, :12].tolist() == [10 + k for k in range(12)]
    assert (m.memoire_cases_suiv[0, 12:] == 0).all()
    assert (m.memoire_types_suiv[0, 12:] == 0).all()
    assert not m.memoire_masques_suiv[0, 12:].any()


def test_fin_de_partie():
    m = memoire()
    ajouter(m, 1, fin=True)
    assert m.memoire_indicateur_fin[0]
    assert m.memoire_nb_pieces_suiv[0] == 0
    assert (m.memoire_plans_suiv[0] == 0).all()
    assert not m.memoire_masques_suiv[0].any()
    assert (m.memoire_cases_suiv[0] == 0).all() and (m.memoire_types_suiv[0] == 0).all()
    # s et a sont quand même stockés
    assert (m.memoire_plans[0] == 1).all()
    assert m.memoire_r[0] == 1.0


# --- File circulaire ----------------------------------------------------------

def test_file_circulaire_ecrase_les_plus_anciennes():
    # Taille 3, 5 ajouts : T4 et T5 écrasent T1 et T2 → lignes [T4, T5, T3]
    m = memoire(taille=3)
    for i in range(1, 6):
        ajouter(m, i)
    assert m.memoire_r.tolist() == [4.0, 5.0, 3.0]
    assert m.compteur_transitions == 5
    assert m.taille_actuelle_memoire == 3
    assert m.pointeur_ecriture == 2


def test_pas_de_pieces_fantomes_apres_ecrasement():
    # Une transition à 16 pièces est écrasée par une à 12 :
    # les 4 dernières places ne doivent rien garder de l'ancienne
    m = memoire(taille=1)
    ajouter(m, 1, nb_pieces_suiv=16)
    ajouter(m, 2, nb_pieces_suiv=12)
    assert m.memoire_nb_pieces_suiv[0] == 12
    assert not m.memoire_masques_suiv[0, 12:].any()
    assert (m.memoire_cases_suiv[0, 12:] == 0).all()
    assert (m.memoire_types_suiv[0, 12:] == 0).all()


def test_fin_de_partie_efface_l_ancienne_s_prime():
    m = memoire(taille=1)
    ajouter(m, 1)
    ajouter(m, 2, fin=True)
    assert m.memoire_indicateur_fin[0]
    assert m.memoire_nb_pieces_suiv[0] == 0
    assert (m.memoire_plans_suiv[0] == 0).all()
    assert not m.memoire_masques_suiv[0].any()


def test_une_partie_normale_apres_une_fin_remet_l_indicateur_a_faux():
    m = memoire(taille=1)
    ajouter(m, 1, fin=True)
    ajouter(m, 2)
    assert not m.memoire_indicateur_fin[0]


# --- Tirage des lots ----------------------------------------------------------

def test_forme_du_lot():
    m = memoire(taille=20)
    for i in range(10):
        ajouter(m, i)
    lot = m.tirer_lot(4)
    assert lot.plans.shape == lot.plans_suiv.shape == (4, NB_PLANS, 8, 8)
    assert lot.masque_suiv.shape == (4, NB_PIECES_MAX, NB_SORTIES)
    assert lot.cases_suiv.shape == lot.types_suiv.shape == (4, NB_PIECES_MAX)
    for champ in (lot.r, lot.c, lot.indicateur_fin, lot.type_action, lot.indice, lot.case_dep, lot.nb_pieces_suiv):
        assert champ.shape == (4,)


def test_les_champs_d_une_meme_transition_restent_alignes():
    # Dans le lot, la ligne k de chaque champ doit venir de la même transition :
    # r = numero, plans = numero, plans de s' = numero + 100
    m = memoire(taille=50)
    for i in range(1, 31):
        ajouter(m, i)
    lot = m.tirer_lot(20)
    for k in range(20):
        numero = int(lot.r[k])
        assert (lot.plans[k] == numero).all()
        assert (lot.plans_suiv[k] == numero + 100).all()


def test_le_tirage_ne_prend_que_des_lignes_remplies():
    # Mémoire de 100 places, 5 remplies : on ne doit jamais tirer une ligne vide (r = 0)
    m = memoire(taille=100)
    for i in range(1, 6):
        ajouter(m, i)
    for _ in range(50):
        assert set(m.tirer_lot(5).r.tolist()) <= {1.0, 2.0, 3.0, 4.0, 5.0}


def test_le_tirage_finit_par_couvrir_toute_la_memoire():
    m = memoire(taille=10)
    for i in range(1, 11):
        ajouter(m, i)
    vus = set()
    for _ in range(100):
        vus |= set(m.tirer_lot(5).r.tolist())
    assert vus == {float(i) for i in range(1, 11)}


def test_meme_graine_memes_lots():
    a, b = memoire(taille=20, graine=3), memoire(taille=20, graine=3)
    for i in range(1, 21):
        ajouter(a, i)
        ajouter(b, i)
    np.testing.assert_array_equal(a.tirer_lot(8).r, b.tirer_lot(8).r)


def test_le_lot_est_une_copie():
    # Modifier un lot ne doit pas modifier la mémoire
    m = memoire()
    ajouter(m, 1)
    lot = m.tirer_lot(1)
    lot.plans[:] = 99
    assert (m.memoire_plans[0] == 1).all()