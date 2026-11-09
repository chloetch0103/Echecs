from dataclasses import dataclass

import chess
from torch import nn

from echecs_ma.agents.arbitre import Arbitre
from echecs_ma.entrainement.memoire import Memoire
from echecs_ma.env.environnement import Environnement


@dataclass
class StatPartie:
    z: int
    nbr_echecs_subis: int
    nbr_coups_joues: int


def jouer_partie_complete(env: Environnement, memoire: Memoire, arbitre: Arbitre, couleur: bool,
                          reseaux: dict[chess.PieceType, nn.Module], lambda_: float, epsilon: float):
    fin_partie = False
    decision_pred, position_pred = None, None
    r, c = None, None
    env.reset()
    nbr_echecs_subis = 0
    nbr_coups_joues = 0

    # Boucle de jeu
    while not fin_partie:
        decision, position = arbitre.calcul_coup(env.echiquier, couleur, reseaux, lambda_, epsilon)
        if decision_pred is not None:
            memoire.stocker_en_memoire(r, c, position_pred, decision_pred, position, couleur)
        _, r, c, fin_partie = env.step(decision.coup)
        if fin_partie:
            memoire.stocker_en_memoire(r, c, position, decision, None, couleur)
        nbr_echecs_subis += c
        nbr_coups_joues += 1
        position_pred = position
        decision_pred = decision

    return StatPartie(z=env.get_z(), nbr_coups_joues=nbr_coups_joues, nbr_echecs_subis=nbr_echecs_subis)