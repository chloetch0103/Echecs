import chess
import numpy as np
import torch
from torch import nn, optim

from echecs_ma.entrainement.memoire import Lot
from echecs_ma.env.encodage import ajouter_position_aux_plans


def prediction_reseau_cible(reseaux_cibles: dict[chess.PieceType, nn.Module], lambda_: float, masque_suiv: np.ndarray,
                            plans: np.ndarray, types_suiv: np.ndarray, cases_suiv: np.ndarray, couleur: bool,
                            nbr_pieces_suiv: int):
    """

    :param reseaux_cibles:
    :param lambda_:
    :param masque_suiv:
    :param plans:
    :param types_suiv:
    :param cases_suiv:
    :param couleur:
    :param nbr_pieces_suiv:
    :return:
    """
    val_fonct_obj_max = -np.inf
    sortie_qr_opti = None
    sortie_qc_opti = None

    # On trouve la prédiction qui maximise l'objectif sur tous les réseaux
    for i in range(nbr_pieces_suiv):
        type_piece = types_suiv[i]
        masque = masque_suiv[i]
        plans_complets = ajouter_position_aux_plans(plans, cases_suiv[i], couleur)
        reseau = reseaux_cibles.get(type_piece, None)
        if reseau is None:
            raise ValueError(f"Le réseau de la pièce {type_piece} n'existe pas.")

        # 1. Prédiction sans correction des poids
        with torch.no_grad():
            sortie_qr, sortie_qc = reseau(torch.from_numpy(plans_complets).unsqueeze(0))
            sortie_qr = sortie_qr[0]
            sortie_qc = sortie_qc[0]

        # 2. Calcul des valeurs de la fonction objectif
        val_fonct_obj = sortie_qr - lambda_ * sortie_qc
        val_fonct_obj[~torch.from_numpy(masque)] = -np.inf  # On met les scores invalides à -inf
        val_fonct_max_temp = torch.max(val_fonct_obj).item()

        # 3. Mise à jour du max
        if val_fonct_max_temp > val_fonct_obj_max:
            val_fonct_obj_max = val_fonct_max_temp
            indice = torch.argmax(val_fonct_obj).item()
            sortie_qc_opti = sortie_qc[indice]
            sortie_qr_opti = sortie_qr[indice]

    return sortie_qr_opti.item(), sortie_qc_opti.item()


def apprentissage(lot: Lot,  reseaux: dict[chess.PieceType, nn.Module], gamma: float,
                  reseaux_cibles: dict[chess.PieceType, nn.Module], lambda_: float, optimizer: optim.Optimizer):
    """
    Entraîne les réseaux de neurones à partir d'un lot
    :param optimizer:
    :param gamma:
    :param lambda_:
    :param reseaux_cibles:
    :param reseaux:
    :param lot: Lot de transitions tirées aléatoirement en mémoire
    :return:
    """
    criterion = nn.MSELoss()
    predictions_cible_r = []
    predictions_cible_c = []
    predictions_main_r = []
    predictions_main_c = []

    # 1. Prédiction sur tous les lots
    for i in range(lot.r.shape[0]):
        reseau = reseaux[lot.type_action[i]]
        plans_complets = ajouter_position_aux_plans(lot.plans[i], lot.case_dep[i], lot.couleur[i])

        # a. Prédiction réseau principal
        indice = int(lot.indice[i])
        sortie_qr, sortie_qc = reseau(torch.from_numpy(plans_complets).unsqueeze(0))
        sortie_qr = sortie_qr[0][indice]
        sortie_qc = sortie_qc[0][indice]

        # b. Prédiction réseseau cible
        y_r_c = lot.r[i]
        y_c_c = lot.c[i]
        with torch.no_grad():
            if not lot.indicateur_fin[i]:
                qr_r_c, qc_c_c = prediction_reseau_cible(reseaux_cibles, lambda_, lot.masque_suiv[i],
                                                                   lot.plans_suiv[i], lot.types_suiv[i],
                                                                   lot.cases_suiv[i], lot.couleur[i],
                                                         lot.nb_pieces_suiv[i])
                y_r_c = y_r_c + gamma*qr_r_c
                y_c_c = y_c_c + qc_c_c

        predictions_cible_r.append(y_r_c)
        predictions_cible_c.append(y_c_c)
        predictions_main_r.append(sortie_qr)
        predictions_main_c.append(sortie_qc)


    # 2. Cacul d'erreurs (entre les têtes qui ont joué)
    predictions_cible_r = torch.tensor(predictions_cible_r)
    predictions_cible_c = torch.tensor(predictions_cible_c)
    predictions_main_r = torch.stack(predictions_main_r)
    predictions_main_c = torch.stack(predictions_main_c)
    erreur_c = criterion(predictions_cible_c, predictions_main_c)
    erreur_r = criterion(predictions_cible_r, predictions_main_r)

    # 3. Rétropropagation
    optimizer.zero_grad()
    erreur = erreur_r + erreur_c
    erreur.backward()

    # 4. MÀJ des poids via optimiseur
    optimizer.step()

    return erreur_r.item(), erreur_c.item()

