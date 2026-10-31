import chess
import chess.engine
from abc import ABC, abstractmethod


class Adversaire(ABC):
    @abstractmethod
    def jouer(self, echequier: chess.Board):
        """
        Implémenté par les classes enfants
        :param echequier:
        :return:
        """
        ...


class AdversaireStockFish(Adversaire):
    def __init__(self, moteur: chess.engine.SimpleEngine, temps_coup, niveau_init=None):
        self.moteur = moteur
        self.temps_coup = temps_coup
        if niveau_init is not None:
            self.moteur.configure({"Skill Level": niveau_init})

    def set_niveau_joueur(self, niveau: int):
        self.moteur.configure({"Skill Level": niveau})

    def jouer(self, echequier: chess.Board):
        return self.moteur.play(echequier, chess.engine.Limit(self.temps_coup)).move

    def fermer(self):
        self.moteur.close()


class Evaluateur(ABC):
    @abstractmethod
    def potentiel(self, echiquier: chess.Board, couleur):
        """
        Implémenté par les classes enfants
        :param echiquier:
        :param couleur:
        :return:
        """
        ...


class EvaluateurStockfish(Evaluateur):
    def __init__(self, moteur: chess.engine.SimpleEngine, temps_analyse):
        self.moteur = moteur
        self.temps_analyse = temps_analyse

    def potentiel(self, echiquier: chess.Board, couleur):
        info = self.moteur.analyse(echiquier, chess.engine.Limit(self.temps_analyse))
        score = info["score"]
        esperance_vnd = score.pov(couleur).wdl().expectation()
        return 2 * esperance_vnd - 1

    def fermer(self):
        self.moteur.close()