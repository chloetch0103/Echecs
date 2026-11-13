"""Étape d'apprentissage DQN, par lots.

Tous les passages dans les réseaux sont regroupés par type de pièce : au plus
6 appels pour les prédictions et 6 pour les cibles, quel que soit la taille du lot.
(6 étant le nombre de réseaux)

Notations des formes :
    B : taille du lot (nombre de transitions)
    P : nombre maximal de pièces par position (NB_PIECES_MAX = 16)
    M : nombre total de vraies pièces de s' dans le lot (transitions non finales,
        places au-delà de nb_pieces_suiv ignorées)
"""
import chess
import numpy as np
import torch
from torch import nn, optim

from echecs_ma.entrainement.memoire import Lot
from echecs_ma.env.encodage import ajouter_positions_aux_plans_lot


def _reseau(reseaux: dict[chess.PieceType, nn.Module], type_piece: int) -> nn.Module:
    reseau = reseaux.get(type_piece)
    if reseau is None:
        raise ValueError(f"Le réseau de la pièce {type_piece} n'existe pas.")
    return reseau


def meilleurs_coups(q_r: np.ndarray, q_c: np.ndarray, masques: np.ndarray, lambda_: float):
    """
    Meilleur coup autorisé de chaque ligne selon Q_R − lambda Q_C.
    :param q_c: (N, 88) sorties de l'une des deux têtes
    :param q_r: (N, 88) sorties de l'une des deux têtes
    :param lambda_:
    :param masques: (N, 88) coups autorisés
    :return: indices (N,), valeurs de l'objectif (N,) ; −inf pour une ligne sans aucun coup autorisé
    """
    objectif = np.where(masques, q_r - lambda_ * q_c, -np.inf)
    indices = objectif.argmax(axis=1)
    valeurs = objectif[np.arange(len(indices)), indices]
    return indices, valeurs


def cibles_suivantes(reseaux_cibles: dict[chess.PieceType, nn.Module], lambda_: float,
                     plans_suiv: np.ndarray, types_suiv: np.ndarray, cases_suiv: np.ndarray,
                     masques_suiv: np.ndarray, couleurs: np.ndarray, nb_pieces_suiv: np.ndarray,
                     fins: np.ndarray):
    """
    Q_R(s', a') et Q_C(s', a') pour chaque transition, a' étant le coup que choisirait l'arbitre
    dans s' avec les réseaux cibles.
    :param lambda_:
    :param reseaux_cibles:
    :param plans_suiv: (B, 19, 8, 8)
    :param types_suiv: (B, P)
    :param cases_suiv: (B, P)
    :param masques_suiv: (B, P, 88)
    :param couleurs: (B,)
    :param nb_pieces_suiv: (B,)
    :param fins: (B,)
    :return: deux tableaux (B,) et 0 pour les transitions finales
    """
    b = len(nb_pieces_suiv)
    q_r_suiv = np.zeros(b, dtype=np.float32)
    q_c_suiv = np.zeros(b, dtype=np.float32)

    # 1. "Déplier" les pièces : une ligne par vraie pièce de s', avec le numéro de sa transition
    places = np.arange(types_suiv.shape[1])
    # vraies, de forme (B, P) : vraies[t, k] vaut True si la place k de la transition t contient
    # une vraie pièce de s'. Il vaut False si k >= nb_pieces_suiv[t] (place vide, ou fantôme),
    # ou si la transition t est terminale
    vraies = (places[None, :] < np.asarray(nb_pieces_suiv)[:, None]) & ~np.asarray(fins, dtype=bool)[:, None]
    transition, place = np.nonzero(vraies)  # (M,), (M,)
    m = len(transition)
    if m == 0:
        return q_r_suiv, q_c_suiv  # toutes les transitions sont terminales

    types = np.asarray(types_suiv)[transition, place]             # (M,)
    masques = np.asarray(masques_suiv)[transition, place]         # (M, 88)
    entrees = ajouter_positions_aux_plans_lot(np.asarray(plans_suiv)[transition],
                                              np.asarray(cases_suiv)[transition, place],
                                              np.asarray(couleurs)[transition])   # (M, 20, 8, 8)

    # 2. Un passage par réseau cible, puis le meilleur coup de chaque pièce
    q_r_piece = np.zeros(m, dtype=np.float32)
    q_c_piece = np.zeros(m, dtype=np.float32)
    valeur_piece = np.full(m, -np.inf)
    for type_piece in chess.PIECE_TYPES:
        lignes = np.nonzero(types == type_piece)[0]
        if len(lignes) == 0:
            continue
        with torch.no_grad():
            sortie_r, sortie_c = _reseau(reseaux_cibles, type_piece)(torch.from_numpy(entrees[lignes]))
        sortie_r, sortie_c = np.asarray(sortie_r), np.asarray(sortie_c)                   # (n, 88)
        indices, valeurs = meilleurs_coups(sortie_r, sortie_c, masques[lignes], lambda_)
        k = np.arange(len(lignes))
        q_r_piece[lignes] = sortie_r[k, indices]
        q_c_piece[lignes] = sortie_c[k, indices]
        valeur_piece[lignes] = valeurs

    # 3. Pour chaque transition, la pièce dont le meilleur coup est le plus haut
    #    Simple boucle sur les M pièces (uniquement des comparaisons de nombres, aucun appel au réseau)
    meilleure_valeur = np.full(b, -np.inf)
    for ligne in range(m):
        t = transition[ligne]
        if valeur_piece[ligne] > meilleure_valeur[t]:
            meilleure_valeur[t] = valeur_piece[ligne]
            q_r_suiv[t] = q_r_piece[ligne]
            q_c_suiv[t] = q_c_piece[ligne]

    return q_r_suiv, q_c_suiv


def prediction_reseau_cible(reseaux_cibles: dict[chess.PieceType, nn.Module], lambda_: float, masque_suiv: np.ndarray,
                            plans: np.ndarray, types_suiv: np.ndarray, cases_suiv: np.ndarray, couleur: bool,
                            nbr_pieces_suiv: int):
    """Version pour une seule transition (non finale) : Q_R(s', a') et Q_C(s', a') en nombres Python."""
    q_r, q_c = cibles_suivantes(reseaux_cibles, lambda_, plans[None], np.asarray(types_suiv)[None],
                                np.asarray(cases_suiv)[None], np.asarray(masque_suiv)[None],
                                np.array([couleur]), np.array([nbr_pieces_suiv]), np.array([False]))
    return float(q_r[0]), float(q_c[0])


def apprentissage(lot: Lot, reseaux: dict[chess.PieceType, nn.Module], gamma: float,
                  reseaux_cibles: dict[chess.PieceType, nn.Module], lambda_: float, optimizer: optim.Optimizer):
    """
    Une correction des réseaux à partir d'un lot de transitions.
    :return: erreur sur Q_R, erreur sur Q_C (nombres Python)
    """
    # 1. Prédictions Q(s, a) (un passage par type de pièce qui a joué)
    entrees = ajouter_positions_aux_plans_lot(lot.plans, lot.case_dep, lot.couleur)   # (B, 20, 8, 8)
    morceaux_r, morceaux_c, ordre = [], [], []
    for type_piece in chess.PIECE_TYPES:
        lignes = np.nonzero(lot.type_action == type_piece)[0]       # transitions jouées par ce type
        if len(lignes) == 0:
            continue
        sortie_r, sortie_c = _reseau(reseaux, type_piece)(torch.from_numpy(entrees[lignes]))   # (n, 88)
        k = torch.arange(len(lignes))
        coups = torch.as_tensor(lot.indice[lignes].astype(np.int64))
        morceaux_r.append(sortie_r[k, coups])                       # (n,) : la sortie du coup joué, ligne par ligne
        morceaux_c.append(sortie_c[k, coups])
        ordre.append(lignes)
    ordre = np.concatenate(ordre)                                   # numéro de transition de chaque prédiction
    prediction_r = torch.cat(morceaux_r)                            # (B,), dans l'ordre de `ordre`
    prediction_c = torch.cat(morceaux_c)

    # 2. Cibles, sans gradient : r + gamma Q_R(s', a') et c + Q_C(s', a') (pas de gamma pour le danger)
    q_r_suiv, q_c_suiv = cibles_suivantes(reseaux_cibles, lambda_, lot.plans_suiv, lot.types_suiv, lot.cases_suiv,
                                          lot.masque_suiv, lot.couleur, lot.nb_pieces_suiv, lot.indicateur_fin)
    fins = lot.indicateur_fin.astype(bool)
    cible_r = np.where(fins, lot.r, lot.r + gamma * q_r_suiv)
    cible_c = np.where(fins, lot.c, lot.c + q_c_suiv)
    cible_r = torch.as_tensor(cible_r[ordre], dtype=torch.float32)   # remises dans l'ordre des prédictions
    cible_c = torch.as_tensor(cible_c[ordre], dtype=torch.float32)

    # 3. Erreur et une seule correction pour tout le lot
    critere = nn.MSELoss()
    erreur_r = critere(prediction_r, cible_r)
    erreur_c = critere(prediction_c, cible_c)
    optimizer.zero_grad()
    (erreur_r + erreur_c).backward()
    optimizer.step()

    return erreur_r.item(), erreur_c.item()