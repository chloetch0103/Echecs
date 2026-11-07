from dataclasses import dataclass

import chess
import numpy as np
import torch
from torch import nn

from echecs_ma.env.encodage import (
    Action,
    ajouter_position_aux_plans,
    encoder_echiquier,
    encoder_masque_coups_legaux,
    move_to_action,
)


@dataclass(frozen=True)
class ArbitreDecision:
    coup: chess.Move
    action: Action


@dataclass(frozen=True)
class ArbitrePositon:
    masques: list[np.ndarray]
    dicos: dict[int, chess.Move]
    pieces_t: list[chess.PieceType]
    cases_p: list[chess.Square]
    plans: np.ndarray


class Arbitre:
    def __init__(self, generateur_aleatoire:np.random.Generator):
        self.generateur_aleatoire = generateur_aleatoire


    def calcul_coup_aleatoire(self, echiquier: chess.Board, couleur: bool):
        # 1. On vérifie que c'est bien à l'algorithme de jouer
        if echiquier.turn != couleur:
            raise ValueError("Ce n'est pas à l'algorithme de jouer")

        # 2. On choisit aléatoirement un coup parmi les coups légaux
        coups_legaux = list(echiquier.legal_moves)
        indice_aleatoire = self.generateur_aleatoire.integers(0, len(coups_legaux))
        coup = coups_legaux[indice_aleatoire]
        return coup


    def calcul_coup(self, echiquier: chess.Board, couleur: bool,
                    reseaux: dict[chess.PieceType, nn.Module], lambda_: float, epsilon: float):
        plans = encoder_echiquier(echiquier, couleur)
        masques = []
        dicos = []
        pieces_t = []
        cases_p = []
        val_fonct_obj_max = -np.inf
        flag = True
        coup = None

        # 1. Choix aléatoire de position
        if self.generateur_aleatoire.random() < epsilon:
            coup = self.calcul_coup_aleatoire(echiquier, couleur)
            flag = False

        # 2. Choix basé sur les réseaux de neurones
        for type_piece in chess.PIECE_TYPES:
            reseau = reseaux.get(type_piece, None)
            if reseau is None:
                raise ValueError(f"Le réseau de la pièce {type_piece} n'existe pas.")
            for pos_en_square in list(echiquier.pieces(type_piece, couleur)):
                plans_complets = ajouter_position_aux_plans(plans, pos_en_square, couleur)
                masque, indice_to_coup = encoder_masque_coups_legaux(echiquier, couleur, pos_en_square)
                masques.append(masque)
                dicos.append(indice_to_coup)
                pieces_t.append(type_piece)
                cases_p.append(pos_en_square)
                if flag:
                    sortie = est_coup_maxi(masque, lambda_, reseau, plans_complets, indice_to_coup,
                                                                val_fonct_obj_max)
                    if sortie:
                        coup, val_fonct_obj_max = sortie

        return (ArbitreDecision(coup=coup, action=move_to_action(echiquier, coup, couleur)),
                ArbitrePositon(masques=masques, dicos=dicos, pieces_t=pieces_t, cases_p=cases_p, plans=plans))


def est_coup_maxi(masque: np.ndarray, lambda_: float, reseau: nn.Module, plans_complets: np.ndarray,
                                 indice_to_coup: dict, val_fonct_obj_max: float):
    """
    Détermine si le réseau considéré donne un coup avec une meilleure évaluation que val_fonct_obj_max.
    :param val_fonct_obj_max:
    :param indice_to_coup:
    :param plans_complets:
    :param reseau:
    :param masque:
    :param lambda_:
    :return:
    """
    if not masque.any():  # Si la pièce considérée n'a aucun coup valide
        return None

    # Prédiction de Q_R et Q_C par les réseaux de neurones
    with torch.no_grad():
        sortie_qr, sortie_qc = reseau(torch.from_numpy(plans_complets).unsqueeze(0))
        sortie_qr = sortie_qr[0]
        sortie_qc = sortie_qc[0]
    coup_temp, val_fonct_max_temp = calcul_coup_optim_local(lambda_, masque, indice_to_coup,
                                                            sortie_qr, sortie_qc)
    if val_fonct_max_temp > val_fonct_obj_max:
        return coup_temp, val_fonct_max_temp


def calcul_coup_optim_local(lambda_: float, masque: np.ndarray, indice_to_coup: dict,
                            sortie_qr, sortie_qc):
    """
    Calcul du coup optimal pour un réseau (donc un type de pièce)
    :param sortie_qr:
    :param sortie_qc:
    :param lambda_:
    :param masque:
    :param indice_to_coup:
    :return:
    """
    # 2. Recherche du coup qui maximise la fonction objective
    val_fonct_obj = sortie_qr - lambda_ * sortie_qc
    val_fonct_obj[~torch.from_numpy(masque)] = -np.inf  # On met les scores invalides à -inf
    indice = torch.argmax(val_fonct_obj).item()

    return indice_to_coup[indice], torch.max(val_fonct_obj).item()

