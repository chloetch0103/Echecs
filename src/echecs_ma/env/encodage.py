"""
Encodage de l'échiquier pour le réseau de neurones, du point de vue de l'algorithme
(nos pièces toujours en bas). 20 plans 8x8 :
  0-5 : nos pièces
  6-11 : pièces adverses
  12 : case de prise en passant
  13-16 : droits de roque (nous petit/grand, adversaire petit/grand)
  17 : compteur règle des 50 coups (normalisé)
  18 : 1 partout si la position est déjà apparue au moins une fois avant, sinon 0
  19 : 1 partout si c’est sa 3e apparition (ou plus), sinon 0
"""

import numpy as np
import chess

NB_PLANS = 20


def case_vers_indices(square: int, couleur_algo: bool) -> tuple[int, int]:
    """Convertit une case python-chess en (ligne, colonne) dans le tenseur.
    Ligne 0 = notre première rangée. Pour les noirs, on inverse uniquement les rangées
    (miroir vertical) afin de garder l'aile roi du même côté."""
    rang = chess.square_rank(square)
    colonne = chess.square_file(square)
    if couleur_algo == chess.BLACK:
        rang = 7 - rang
    return rang, colonne


def encoder_piece(echiquier: chess.Board, couleur: bool, type_piece: int,
                  couleur_algo: bool) -> np.ndarray:
    plan = np.zeros((8, 8), dtype=np.float32)
    for square in echiquier.pieces(type_piece, couleur):
        i, j = case_vers_indices(square, couleur_algo)
        plan[i, j] = 1.0
    return plan


def encoder_echiquier(echiquier: chess.Board, couleur: bool) -> np.ndarray:
    plans = np.zeros((NB_PLANS, 8, 8), dtype=np.float32)
    couleur_adv = not couleur

    # 1. Pièces (12 plans)
    k = 0
    for coul in (couleur, couleur_adv):
        for type_piece in chess.PIECE_TYPES:
            plans[k] = encoder_piece(echiquier, coul, type_piece, couleur)
            k += 1

    # 2. Prise en passant (1 sur la case cible, seulement si la prise est légale)
    if echiquier.has_legal_en_passant():
        i, j = case_vers_indices(echiquier.ep_square, couleur)
        plans[12, i, j] = 1.0

    # 3. Droits de roque (plans constants)
    plans[13] = float(echiquier.has_kingside_castling_rights(couleur))
    plans[14] = float(echiquier.has_queenside_castling_rights(couleur))
    plans[15] = float(echiquier.has_kingside_castling_rights(couleur_adv))
    plans[16] = float(echiquier.has_queenside_castling_rights(couleur_adv))

    # 4. Règle des 50 coups (plan constant normalisé (100 demi-coups = nulle))
    plans[17] = min(echiquier.halfmove_clock, 100) / 100.0

    # 5. Répétition de positions
    plans[18] = float(echiquier.is_repetition(2))
    plans[19] = float(echiquier.is_repetition(3))

    return plans

