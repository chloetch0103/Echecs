"""
Encodage de l'échiquier pour le réseau de neurones, du point de vue de l'algorithme
(nos pièces toujours en bas). 19 plans 8x8 (le 20e est rajouté par la suite) :
  0-5 : nos pièces
  6-11 : pièces adverses
  12 : case de prise en passant
  13-16 : droits de roque (nous petit/grand, adversaire petit/grand)
  17 : compteur règle des 50 coups
  18 : 1 partout si la position est déjà apparue au moins une fois avant, sinon 0
  19 : ajouté par la suite avec la fonction ajouter_position_aux_plans avec normalisation
  du plan 17

Encodage de la sortie des réseaux :
    - Les réseaux doivent estimer 88 Q_R et Q_C (64 pour chaque case de l'échiquier + promotion en reine
    et 8*3 pour les sous promotions (on ne regarde que les colonnes, car les promotions se font toujours
    sur la même ligne). On doit donc construire un vecteur masque de taille 88 pour ne garder que les
    coups valides.
"""
from dataclasses import dataclass

import chess
import numpy as np

NB_PLANS = 19
NB_PLANS_ENTREE = NB_PLANS + 1
NB_SORTIES = 88
NB_CASES = 64
NB_COLS = 8

TYPE_PIECE_TO_INDICE: dict[chess.PieceType, int] = {
    chess.BISHOP: 0,
    chess.KNIGHT: 1,
    chess.ROOK: 2,
}

@dataclass(frozen=True)
class Action:
    type_piece: chess.PieceType
    case_depart: int
    indice: int

    def __post_init__(self):
        if not 0 <= self.case_depart < 64:
            raise ValueError(f"La position doit être entre 0 et 63: case_depart = {self.case_depart}")
        if not 0 <= self.indice < NB_SORTIES:
            raise ValueError(f"L'indice doit être entre 0 et {NB_SORTIES-1}: indice = {self.indice}")


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
    plan = np.zeros((8, 8), dtype=np.uint8)
    for square in echiquier.pieces(type_piece, couleur):
        i, j = case_vers_indices(square, couleur_algo)
        plan[i, j] = 1.0
    return plan


def encoder_echiquier(echiquier: chess.Board, couleur: bool) -> np.ndarray:
    plans = np.zeros((NB_PLANS, 8, 8), dtype=np.uint8)
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
        plans[12, i, j] = 1

    # 3. Droits de roque (plans constants)
    plans[13] = float(echiquier.has_kingside_castling_rights(couleur))
    plans[14] = float(echiquier.has_queenside_castling_rights(couleur))
    plans[15] = float(echiquier.has_kingside_castling_rights(couleur_adv))
    plans[16] = float(echiquier.has_queenside_castling_rights(couleur_adv))

    # 4. Règle des 50 coups (plan constant normalisé (100 demi-coups = nulle))
    plans[17] = min(echiquier.halfmove_clock, 100)

    # 5. Répétition de positions
    plans[18] = np.int8(echiquier.is_repetition(2))

    return plans


def ajouter_position_aux_plans(plans: np.ndarray, pos_depart: int, couleur: bool) -> np.ndarray:
    plans_complet = np.zeros((NB_PLANS_ENTREE, 8, 8), dtype=np.float32)
    i, j = case_vers_indices(pos_depart, couleur)
    plan = np.zeros((8, 8), dtype=np.float32)
    plan[i, j] = 1
    plans_complet[:NB_PLANS] = plans
    plans_complet[NB_PLANS] = plan

    # Normalisation de la répétition des 50 coups (17e plan)
    plans_complet[17] = plans_complet[17]/100
    return plans_complet


def coup_et_couleur_to_indice(coup: chess.Move, couleur: bool) -> int:
    """
    Détermine l'indice dans masque du coup considéré
    :param coup:
    :param couleur:
    :return:
    """
    promotion = coup.promotion
    i, j = case_vers_indices(coup.to_square, couleur)

    # coup sans promotion ou promotion en tant que reine
    if not promotion or promotion == chess.QUEEN:
        return 8*i + j

    # sous promotion
    return NB_CASES + NB_COLS*TYPE_PIECE_TO_INDICE[promotion] + j


def encoder_masque_coups_legaux(echiquier: chess.Board, couleur: bool, pos_piece: int) -> tuple[np.ndarray, dict]:
    """
    Met des True partout où l'agent peut aller. Construit également un dictionnaire permettant de passer des indices du
    masque aux coups python-chess
    :param echiquier:
    :param couleur: En tant que qui on joue
    :param pos_piece: position actuelle de la pièce
    :return: masque, dictionnaire indice ==> coups
    """
    masque = np.full(NB_SORTIES, False, dtype=bool)
    indice_to_coups = {}

    # 1. On vérifie si c'est bien à l'algorithme de jouer
    if echiquier.turn != couleur:
        raise ValueError("Ce n'est pas à l'algorithme de jouer")

    # 2. On vérifie que le bon pion se trouve bien à la bonne position
    p = echiquier.piece_at(pos_piece)
    if p is None or p.color != couleur:
        raise ValueError(f"Aucune des pièces de l'algorithme ne se trouve à l'emplacement {pos_piece}")

    # 3. On remplit le masque et le dictionnaire indice_to_coup
    for coup in echiquier.legal_moves:
        if coup.from_square == pos_piece:
            indice = coup_et_couleur_to_indice(coup, couleur)
            masque[indice] = True
            indice_to_coups[indice] = coup

    return masque, indice_to_coups


def move_to_action(echiquier: chess.Board, coup: chess.Move, couleur: bool) -> Action:
    """
    Détermine le type de pièce, sa position en carré dans l'échiquier avant coup et son indice
    dans le masque à partir de coup
    :param echiquier:
    :param couleur:
    :param coup:
    :return:
    """
    # 1. Type de pièce
    piece_t = echiquier.piece_at(coup.from_square)
    if piece_t is None or piece_t.color != couleur:
        raise ValueError(f"Aucune des pièces de l'algorithme ne se trouve à l'emplacement {coup.from_square}")

    # 2. Position en square dans l'échiquier avant coup
    case_depart = coup.from_square

    # 3. Indice du coup dans le masque
    indice = coup_et_couleur_to_indice(coup, couleur)

    return Action(type_piece=piece_t.piece_type, indice=indice, case_depart=case_depart)



