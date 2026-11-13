import enum
from dataclasses import dataclass

import chess
from torch import nn

from echecs_ma.agents.arbitre import Arbitre
from echecs_ma.entrainement.memoire import Memoire
from echecs_ma.env.adversaire import EvaluateurMateriel
from echecs_ma.env.environnement import Environnement
from echecs_ma.env.valeurs import bilan_materiel


class RaisonFinPartie(enum.Enum):
    """Enum with reasons for a game to be over."""

    CHECKMATE = enum.auto()
    STALEMATE = enum.auto()
    INSUFFICIENT_MATERIAL = enum.auto()
    SEVENTYFIVE_MOVES = enum.auto()
    FIVEFOLD_REPETITION = enum.auto()
    FIFTY_MOVES = enum.auto()
    THREEFOLD_REPETITION = enum.auto()
    VARIANT_WIN = enum.auto()
    VARIANT_LOSS = enum.auto()
    VARIANT_DRAW = enum.auto()
    PARTIE_TROP_LONGUE = enum.auto()

    @classmethod
    def depuis_python_chess(cls, termination: chess.Termination):
        """Convertit instantanément un chess.Termination en RaisonFinPartie"""
        return cls[termination.name]


@dataclass
class StatPartie:
    z: int
    nbr_echecs_subis: int
    nbr_coups_joues: int
    bilan_mat_fin_partie: float
    raison_fin_partie: RaisonFinPartie


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
    bilan_mat_fin_partie = bilan_materiel(env.echiquier, couleur)

    resultat = env.echiquier.outcome()
    if resultat is not None:
        raison_fin_partie = RaisonFinPartie.depuis_python_chess(resultat.termination)
        z = env.get_z()
    else:
        eval_mat = EvaluateurMateriel()
        raison_fin_partie = RaisonFinPartie.PARTIE_TROP_LONGUE
        z = eval_mat.potentiel(env.echiquier, couleur)

    return StatPartie(z=z, nbr_coups_joues=nbr_coups_joues, nbr_echecs_subis=nbr_echecs_subis,
                      bilan_mat_fin_partie=bilan_mat_fin_partie,
                      raison_fin_partie=raison_fin_partie)