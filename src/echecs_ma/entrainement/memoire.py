from dataclasses import dataclass

import numpy as np

from echecs_ma.agents.arbitre import ArbitreDecision, ArbitrePositon
from echecs_ma.env.encodage import NB_PLANS, NB_SORTIES

NB_PIECES_MAX= 16


@dataclass
class Lot:
    plans: np.ndarray
    r: np.ndarray
    c: np.ndarray
    indicateur_fin: np.ndarray
    masque_suiv: np.ndarray
    type_action: np.ndarray
    nb_pieces_suiv: np.ndarray
    indice: np.ndarray
    case_dep: np.ndarray
    cases_suiv: np.ndarray
    types_suiv: np.ndarray
    plans_suiv: np.ndarray


class Memoire:
    def __init__(self, taille_memoire: int, gen: np.random.Generator):
        self.taille_memoire = taille_memoire
        self.compteur_transitions = 0  # Nombre de transitions ajoutées en mémoires
        self.pointeur_ecriture = 0  # Ligne où écrire (modulo taille mémoire pour pas faire de buffer overflow)
        self.taille_actuelle_memoire = 0  # minimum entre le compteur de transition et la taille de la mémoire
        self.generateur_aleatoire = gen
        self.memoire_plans = np.zeros((taille_memoire, NB_PLANS, 8, 8), dtype=np.uint8)
        self.memoire_r = np.zeros(taille_memoire, dtype=np.float32)
        self.memoire_c = np.zeros(taille_memoire, dtype=np.float32)
        self.memoire_indicateur_fin = np.zeros(taille_memoire, dtype=np.bool)
        self.memoire_masques_suiv = np.zeros((taille_memoire, NB_PIECES_MAX, NB_SORTIES), dtype=np.bool)
        self.memoire_nb_pieces_suiv = np.zeros(taille_memoire, dtype=np.uint8)
        self.memoire_type_action = np.zeros(taille_memoire, dtype=np.uint8)
        self.memoire_indice = np.zeros(taille_memoire, dtype=np.uint8)
        self.memoire_case_dep = np.zeros(taille_memoire, dtype=np.uint8)
        self.memoire_cases_suiv = np.zeros((taille_memoire, NB_PIECES_MAX), dtype=np.uint8)
        self.memoire_types_suiv = np.zeros((taille_memoire, NB_PIECES_MAX), dtype=np.uint8)
        self.memoire_plans_suiv = np.zeros((taille_memoire, NB_PLANS, 8, 8), dtype=np.uint8)

    def stocker_en_memoire(self,  r: float, c: float, pos_arbitre_av: ArbitrePositon, dec_arbitre_av: ArbitreDecision,
                           pos_arbitre_ap: ArbitrePositon):
        """
        Stocke une transition en memoire (r, c, s, s', a).
        s' à None ==> fin de la partie donc fin = True
        :param r:
        :param c:
        :param pos_arbitre_av:
        :param dec_arbitre_av:
        :param pos_arbitre_ap:
        :return:
        """
        # 1. Stockage récompense et coup
        self.memoire_r[self.pointeur_ecriture] = r
        self.memoire_c[self.pointeur_ecriture] = c

        # 2. Stockage configuration échiquier avant coup (s)
        self.memoire_plans[self.pointeur_ecriture] = pos_arbitre_av.plans

        # 3. Stockage action avant coup (a)
        self.memoire_type_action[self.pointeur_ecriture] = dec_arbitre_av.action.type_piece
        self.memoire_indice[self.pointeur_ecriture] = dec_arbitre_av.action.indice
        self.memoire_case_dep[self.pointeur_ecriture] = dec_arbitre_av.action.case_depart

        # 4. On initialise la mémoire s'
        self.memoire_masques_suiv[self.pointeur_ecriture] = 0
        self.memoire_cases_suiv[self.pointeur_ecriture] = 0
        self.memoire_types_suiv[self.pointeur_ecriture] = 0
        self.memoire_plans_suiv[self.pointeur_ecriture] = 0
        self.memoire_nb_pieces_suiv[self.pointeur_ecriture] = 0

        # 5. On regarde si la partie est terminée
        if pos_arbitre_ap is None:
            self.memoire_indicateur_fin[self.pointeur_ecriture] = True
        else:
            self.memoire_indicateur_fin[self.pointeur_ecriture] = False

            # 6. Stockage configuration échiquier après coup adversaire (s')
            self.memoire_masques_suiv[self.pointeur_ecriture][:len(pos_arbitre_ap.masques)] = pos_arbitre_ap.masques
            self.memoire_cases_suiv[self.pointeur_ecriture][:len(pos_arbitre_ap.cases_p)] = pos_arbitre_ap.cases_p
            self.memoire_types_suiv[self.pointeur_ecriture][:len(pos_arbitre_ap.pieces_t)] = pos_arbitre_ap.pieces_t
            self.memoire_plans_suiv[self.pointeur_ecriture] = pos_arbitre_ap.plans

            # 7. On stocke le nombre de pièces suivantes
            self.memoire_nb_pieces_suiv[self.pointeur_ecriture] = len(pos_arbitre_ap.pieces_t)

        # 8. Mise à jour des paramètres de la mémoire
        self.compteur_transitions += 1
        self.pointeur_ecriture = (self.pointeur_ecriture + 1)%self.taille_memoire
        self.taille_actuelle_memoire = min(self.compteur_transitions, self.taille_memoire)

    def tirer_lot(self, nbr: int):
        """
        Tire aléatoirement nbr transitions en mémoire
        :param nbr:
        :return: Les nombres tirés
        """
        # 1. On vérifie d'abord qu'il y a assez d'éléments en mémoire pour le tirage
        if nbr > self.taille_actuelle_memoire:
            raise ValueError(f"pas assez de transitions en mémoire pour tirer un lot de {nbr}")

        # 2. Tirage !
        indices_transitions = self.generateur_aleatoire.integers(0, self.taille_actuelle_memoire, nbr)
        tirage = Lot(r=self.memoire_r[indices_transitions], c=self.memoire_c[indices_transitions],
                indicateur_fin=self.memoire_indicateur_fin[indices_transitions], masque_suiv=self.memoire_masques_suiv[indices_transitions],
                nb_pieces_suiv=self.memoire_nb_pieces_suiv[indices_transitions], types_suiv=self.memoire_types_suiv[indices_transitions],
                cases_suiv=self.memoire_cases_suiv[indices_transitions], case_dep=self.memoire_case_dep[indices_transitions],
                plans=self.memoire_plans[indices_transitions], indice=self.memoire_indice[indices_transitions],
                type_action=self.memoire_type_action[indices_transitions], plans_suiv=self.memoire_plans_suiv[indices_transitions])
        return tirage