"""Tests de l'encodage : plans d'entrée du réseau et masque des coups légaux."""
import chess
import numpy as np
import pytest

from echecs_ma.env.encodage import (
    NB_PLANS, NB_SORTIES, encoder_echiquier, encoder_masque_coups_legaux,
)

# Indices des plans, pour rendre les tests lisibles
NOS_PIECES = {t: t - 1 for t in chess.PIECE_TYPES}           # soldat 0, ..., roi 5
PIECES_ADVERSES = {t: t + 5 for t in chess.PIECE_TYPES}      # soldat 6, ..., roi 11
PRISE_EN_PASSANT, ROQUES, REGLE_50, REPETITION, PIECE_CONCERNEE = 12, slice(13, 17), 17, 18, 19


def masque(fen, case, couleur=chess.WHITE):
    return encoder_masque_coups_legaux(chess.Board(fen), couleur, case)


# =============================================================================
# encoder_echiquier
# =============================================================================

def test_forme_et_type():
    plans = encoder_echiquier(chess.Board(), chess.WHITE, chess.E2)
    assert plans.shape == (NB_PLANS, 8, 8)
    assert plans.dtype == np.float32


def test_pieces_position_initiale():
    plans = encoder_echiquier(chess.Board(), chess.WHITE, chess.E2)
    assert plans[NOS_PIECES[chess.PAWN]][1].sum() == 8        # nos soldats sur notre 2e rangée
    assert plans[PIECES_ADVERSES[chess.PAWN]][6].sum() == 8   # les leurs sur la 7e
    assert plans[NOS_PIECES[chess.KING]][0, 4] == 1           # notre Roi en e1
    assert plans[0:6].sum() == 16 and plans[6:12].sum() == 16


def test_miroir_pour_les_noirs():
    # La position initiale est symétrique : vue des Noirs, elle doit donner exactement
    # le même encodage que vue des Blancs (soldat e7 des Noirs ↔ soldat e2 des Blancs).
    echiquier = chess.Board()
    blancs = encoder_echiquier(echiquier, chess.WHITE, chess.E2)
    noirs = encoder_echiquier(echiquier, chess.BLACK, chess.E7)
    np.testing.assert_array_equal(blancs, noirs)


def test_miroir_garde_les_colonnes():
    # Miroir vertical seulement : notre Roi reste en colonne e, pas en colonne d
    plans = encoder_echiquier(chess.Board(), chess.BLACK, chess.E8)
    assert plans[NOS_PIECES[chess.KING]][0, 4] == 1


def test_prise_en_passant():
    # 1. e4 a6 2. e5 d5 : les Blancs peuvent prendre en passant en d6
    echiquier = chess.Board()
    for c in ["e2e4", "a7a6", "e4e5", "d7d5"]:
        echiquier.push_uci(c)
    plans = encoder_echiquier(echiquier, chess.WHITE, chess.E5)
    assert plans[PRISE_EN_PASSANT][5, 3] == 1
    assert plans[PRISE_EN_PASSANT].sum() == 1


def test_pas_de_prise_en_passant_si_elle_n_est_pas_legale():
    # 1. e4 : python-chess note e3 comme case de prise en passant, mais personne ne peut y prendre
    echiquier = chess.Board()
    echiquier.push_uci("e2e4")
    plans = encoder_echiquier(echiquier, chess.BLACK, chess.E7)
    assert plans[PRISE_EN_PASSANT].sum() == 0


def test_droits_de_roque():
    assert (encoder_echiquier(chess.Board(), chess.WHITE, chess.E2)[ROQUES] == 1).all()
    # Les Blancs n'ont plus que le petit roque, les Noirs plus que le grand
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"
    roques = encoder_echiquier(chess.Board(fen), chess.WHITE, chess.E1)[ROQUES]
    assert [plan[0, 0] for plan in roques] == [1, 0, 0, 1]   # nous petit/grand, eux petit/grand


def test_droits_de_roque_vus_des_noirs():
    # Mêmes droits, vus des Noirs : « nous » devient les Noirs
    fen = "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1"
    roques = encoder_echiquier(chess.Board(fen), chess.BLACK, chess.E8)[ROQUES]
    assert [plan[0, 0] for plan in roques] == [0, 1, 1, 0]


@pytest.mark.parametrize("demi_coups, attendu", [(0, 0.0), (50, 0.5), (100, 1.0), (130, 1.0)])
def test_regle_des_50_coups(demi_coups, attendu):
    echiquier = chess.Board(f"4k3/8/8/8/8/8/8/4K2R w - - {demi_coups} 80")
    plans = encoder_echiquier(echiquier, chess.WHITE, chess.E1)
    assert (plans[REGLE_50] == attendu).all()


def test_repetition():
    echiquier = chess.Board()
    assert encoder_echiquier(echiquier, chess.WHITE, chess.G1)[REPETITION].sum() == 0
    # Les cavaliers sortent puis rentrent : retour à la position initiale
    for c in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        echiquier.push_uci(c)
    assert (encoder_echiquier(echiquier, chess.WHITE, chess.G1)[REPETITION] == 1).all()


def test_plan_de_la_piece_concernee():
    plans = encoder_echiquier(chess.Board(), chess.WHITE, chess.G1)
    assert plans[PIECE_CONCERNEE].sum() == 1
    assert plans[PIECE_CONCERNEE][0, 6] == 1


def test_deux_soldats_ont_des_entrees_differentes():
    # C'est le rôle du dernier plan : le réseau partagé doit distinguer les soldats
    echiquier = chess.Board()
    a = encoder_echiquier(echiquier, chess.WHITE, chess.A2)
    e = encoder_echiquier(echiquier, chess.WHITE, chess.E2)
    assert not np.array_equal(a, e)
    np.testing.assert_array_equal(a[:PIECE_CONCERNEE], e[:PIECE_CONCERNEE])


# =============================================================================
# encoder_masque_coups_legaux
# =============================================================================

DEPART = chess.STARTING_FEN


def test_masque_forme_et_type():
    m = masque(DEPART, chess.E2)
    assert m.shape == (NB_SORTIES,)
    assert m.dtype == bool


def test_masque_soldat_au_depart():
    # e2 peut aller en e3 (ligne 2, col 4 → 20) et en e4 (ligne 3, col 4 → 28)
    assert np.flatnonzero(masque(DEPART, chess.E2)).tolist() == [20, 28]


def test_masque_cavalier_au_depart():
    # g1 peut aller en f3 (21) et en h3 (23)
    assert np.flatnonzero(masque(DEPART, chess.G1)).tolist() == [21, 23]


def test_masque_piece_bloquee_est_vide():
    assert not masque(DEPART, chess.A1).any()


def test_masque_promotions():
    # Soldat en a7 : promotion en dame sur a8 (56) + trois sous-promotions, colonne a
    m = masque("8/P6k/8/8/8/8/8/K7 w - - 0 1", chess.A7)
    assert m.sum() == 4
    assert m[56]
    assert m[64:].sum() == 3


def test_masque_promotions_avec_prise():
    # Soldat en b7 : avancer en b8 ou prendre en a8 → 2 cases × 4 pièces
    m = masque("r6k/1P6/8/8/8/8/8/7K w - - 0 1", chess.B7)
    assert m.sum() == 8


def test_masque_miroir():
    # Soldat e2 des Blancs et soldat e7 des Noirs, chacun au départ : même masque
    blancs = masque(DEPART, chess.E2, chess.WHITE)
    noirs = masque("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", chess.E7, chess.BLACK)
    np.testing.assert_array_equal(blancs, noirs)


def test_masque_promotion_des_noirs():
    # Les sous-promotions noires doivent aussi tomber dans 64-87, pas sur des cases ordinaires
    m = masque("k7/8/8/8/8/8/p7/7K b - - 0 1", chess.A2, chess.BLACK)
    assert m.sum() == 4
    assert m[56] and m[64:].sum() == 3


@pytest.mark.parametrize("fen, case", [
    (DEPART, chess.E4),                                       # case vide
    (DEPART, chess.E7),                                       # pièce adverse
])
def test_masque_case_invalide(fen, case):
    with pytest.raises(ValueError):
        masque(fen, case)


def test_masque_pas_notre_trait():
    with pytest.raises(ValueError):
        masque(DEPART, chess.E7, chess.BLACK)                 # c'est aux Blancs de jouer


@pytest.mark.parametrize("fen", [
    DEPART,
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",   # roques possibles
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",               # prise en passant
    "r3k3/1P5P/8/8/8/8/8/7K w - - 0 1",                                              # promotions
    "r3k2r/8/8/8/3pP3/8/8/R3K2R b KQkq e3 0 1",                                     # Noirs : roques + en passant
    "4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1",                                             # Noirs : promotions
])
def test_aucun_coup_perdu(fen):
    # Le test le plus important : sur toutes nos pièces, chaque coup légal
    # correspond à exactement une sortie, sans perte ni doublon.
    echiquier = chess.Board(fen)
    assert echiquier.is_valid()
    couleur = echiquier.turn
    total = 0
    for case, piece in echiquier.piece_map().items():
        if piece.color == couleur:
            m = encoder_masque_coups_legaux(echiquier, couleur, case)
            nb_coups = sum(1 for c in echiquier.legal_moves if c.from_square == case)
            assert m.sum() == nb_coups, f"pièce en {chess.square_name(case)}"
            total += m.sum()
    assert total == echiquier.legal_moves.count()