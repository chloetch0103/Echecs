"""Tests des valeurs des pièces et du bilan matériel."""
import chess
import pytest

from echecs_ma.env.valeurs import VALEURS, bilan_materiel


# --- Table des valeurs -------------------------------------------------------

def test_valeurs_des_pieces():
    assert VALEURS == {chess.PAWN: 1, chess.KNIGHT: 3.25, chess.BISHOP: 3,
                       chess.ROOK: 5, chess.QUEEN: 9}


def test_le_roi_n_a_pas_de_valeur():
    assert chess.KING not in VALEURS


# --- Bilan matériel ----------------------------------------------------------

@pytest.mark.parametrize("couleur", [chess.WHITE, chess.BLACK])
def test_bilan_position_initiale_nul(couleur):
    assert bilan_materiel(chess.Board(), couleur) == 0


def test_bilan_rois_seuls_nul():
    # Le Roi ne compte pas : deux Rois seuls donnent un bilan nul
    assert bilan_materiel(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), chess.WHITE) == 0


def test_bilan_vu_des_deux_camps():
    # Les Noirs n'ont plus de dame : +9 pour les Blancs, −9 pour les Noirs
    echiquier = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert bilan_materiel(echiquier, chess.WHITE) == 9
    assert bilan_materiel(echiquier, chess.BLACK) == -9


def test_bilan_apres_une_prise():
    # 1. e4 d5 2. exd5 : les Blancs ont pris un pion
    echiquier = chess.Board()
    for coup in ["e2e4", "d7d5", "e4d5"]:
        echiquier.push_uci(coup)
    assert bilan_materiel(echiquier, chess.WHITE) == 1


def test_cavalier_et_fou_ne_valent_pas_pareil():
    # Blancs : un cavalier ; Noirs : un fou → 3.25 − 3
    echiquier = chess.Board("4kb2/8/8/8/8/8/8/1N2K3 w - - 0 1")
    assert bilan_materiel(echiquier, chess.WHITE) == pytest.approx(0.25)


@pytest.mark.parametrize("promotion, valeur", [
    ("q", 9), ("r", 5), ("b", 3), ("n", 3.25),
])
def test_promotion_change_la_valeur(promotion, valeur):
    # Le soldat promu compte comme sa nouvelle pièce, et plus comme un soldat
    echiquier = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert bilan_materiel(echiquier, chess.WHITE) == 1
    echiquier.push_uci("a7a8" + promotion)
    assert bilan_materiel(echiquier, chess.WHITE) == pytest.approx(valeur)