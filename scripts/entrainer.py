import argparse
import csv
import os
import time
from collections import Counter, deque
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from itertools import chain
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
import yaml
from torch import optim

from echecs_ma.agents.arbitre import Arbitre
from echecs_ma.agents.reseaux import Reseau
from echecs_ma.entrainement.apprentissage import apprentissage
from echecs_ma.entrainement.boucle import RaisonFinPartie, jouer_partie_complete
from echecs_ma.entrainement.fonctions_utiles import calcul_epsilon, calcul_lambda, decision_curriculum
from echecs_ma.entrainement.memoire import Memoire
from echecs_ma.env.adversaire import AdversaireAleatoire, AdversaireStockFish, EvaluateurMateriel, EvaluateurStockfish
from echecs_ma.env.environnement import Environnement

# Colonnes du journal (une ligne toutes les `journal.frequence` parties).
# Les colonnes fin_* sont construites à partir de l'énumération
# donne toujours l'ordre de déclaration, donc en-tête et valeurs restent alignés.
COLONNES_JOURNAL = ["parties", "z_moyen", "echecs_subis", "lambda", "niveau", "epsilon", "erreur_r", "erreur_c",
                    "pourcentage_victoire", "pourcentage_nulle", "pourcentage_defaite", "longueur_moy_partie",
                    "duree", "bilan_mat_moy"]
COLONNES_FIN = [f"fin_{raison.name.lower()}" for raison in RaisonFinPartie]

# Colonnes du fichier par partie (une ligne par partie, aucune agrégation)
COLONNES_PARTIES = ["partie", "couleur", "z", "raison_fin", "nb_coups", "bilan_mat", "echecs_subis", "epsilon",
                    "niveau"]


def lecture_config(path):
    with open(path, "r", encoding="utf-8") as fichier:
        config = yaml.safe_load(fichier)
    return config


def chemin_stockfish(config):
    chemin = config["stockfish"]["chemin"] or os.environ.get("STOCKFISH_PATH")
    if chemin is None:
        raise RuntimeError("Chemin de Stockfish introuvable : renseigne stockfish.chemin "
                           "dans la config ou la variable d'environnement STOCKFISH_PATH.")
    return chemin


class Entrainement:
    def __init__(self, config):
        self.chemin_complet = Path(__file__).resolve().parent.parent
        self.graine = config["experience"]["graine"]
        self.generateur = np.random.default_rng(self.graine)
        self.gamma = config["recompense"]["gamma"]
        self.potentiel = config["recompense"]["potentiel"]
        self.couches_cachees = config["agents"]["reseau"]["couches_cachees"]
        self.taux_apprentissage = config["agents"]["reseau"]["taux_apprentissage"]
        self.epsilon_initial = config["agents"]["exploration"]["epsilon_initial"]
        self.epsilon = self.epsilon_initial
        self.epsilon_final = config["agents"]["exploration"]["epsilon_final"]
        self.parties_decroissance = config["agents"]["exploration"]["parties_decroissance"]
        self.nb_parties = config["entrainement"]["nb_parties"]
        self.taille_memoire = config["entrainement"]["taille_memoire"]
        self.debut_apprentissage = config["entrainement"]["debut_apprentissage"]
        self.taille_lot = config["entrainement"]["taille_lot"]
        self.frequence_cible = config["entrainement"]["frequence_cible"]
        self.frequence_sauvegarde = config["entrainement"]["frequence_sauvegarde"]
        self.dossier_journal = config["journal"]["dossier"]
        self.csv_journal = config["journal"]["csv_file_name"]
        self.csv_parties = config["journal"].get("csv_parties", "parties.csv")
        self.frequence = config["journal"]["frequence"]
        self.stockfish_actif = config["stockfish"]["actif"]
        if self.stockfish_actif or self.potentiel == "stockfish":
            self.chemin_stockfish = chemin_stockfish(config)
        else:
            self.chemin_stockfish = None
        self.niveau_initial_stockfish = config["stockfish"]["niveau_initial"]
        self.niveau_stockfish = self.niveau_initial_stockfish
        self.temps_coup = config["stockfish"]["temps_coup"]
        self.temps_analyse = config["stockfish"]["temps_analyse"]
        self.etat_curriculum = config["stockfish"]["curriculum"]["actif"]
        self.pas_niveau_curriculum = config["stockfish"]["curriculum"]["pas_niveau"]
        self.seuil_montee_curriculum = config["stockfish"]["curriculum"]["seuil_montee"]
        self.fenetre = config["stockfish"]["curriculum"]["fenetre"]
        self.lambda_initial = config["contrainte"]["lagrangien"]["lambda_initial"]
        self.lagrangien_actif = config["contrainte"]["lagrangien"]["actif"]
        self.seuil_danger = config["contrainte"]["lagrangien"]["seuil_danger"]
        self.eta = config["contrainte"]["lagrangien"]["eta"]
        self.bouclier_actif = config["contrainte"]["bouclier"]["actif"]
        self.bouclier_profondeur = config["contrainte"]["bouclier"]["profondeur"]
        self.lambda_ = self.lambda_initial
        self.couleur = bool(self.generateur.choice([True, False]))
        self.adversaire = None
        self.evaluateur = None
        self.arbitre = Arbitre(self.generateur)
        self.environnement = None
        self.evaluateur_mat = EvaluateurMateriel()
        self.memoire = Memoire(self.taille_memoire, self.generateur)
        self.reseaux_principaux = {}
        self.reseaux_cibles = {}
        self.optimiseur = None
        self.creer_les_reseaux_et_optimiseur()
        os.makedirs(self.chemin_complet / self.dossier_journal, exist_ok=True)
        maintenant = datetime.now()
        horodatage = maintenant.strftime("%Y-%m-%d_%Hh%M")
        self.chemin_journal = self.chemin_complet / self.dossier_journal / f"essai_{horodatage}"
        self.chemin_checkpoints = self.chemin_journal / "checkpoints"
        os.makedirs(self.chemin_journal, exist_ok=True)
        os.makedirs(self.chemin_checkpoints, exist_ok=True)
        with open(self.chemin_journal / "config.yaml", "w", encoding="utf-8") as fichier:
            yaml.dump(
                config,
                fichier,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )

    def creer_les_reseaux_et_optimiseur(self):
        torch.manual_seed(self.graine)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.graine)  # graine pour le GPU (au cas où)
        for type_p in chess.PIECE_TYPES:
            self.reseaux_principaux[type_p] = Reseau(self.couches_cachees)
            self.reseaux_cibles[type_p] = deepcopy(self.reseaux_principaux[type_p])

        parametres_combines = chain.from_iterable(r.parameters() for r in self.reseaux_principaux.values())
        self.optimiseur = optim.Adam(parametres_combines, lr=self.taux_apprentissage)

    def lancer_entrainement(self):
        # Initialisation de variables de la boucle
        nb_corr = 0  # nombre de fois le réseau principal a été corrigé
        # erreurs du bloc en cours (vidées à chaque écriture du journal)
        erreurs_r = []
        erreurs_c = []
        val_retour = deque(maxlen=self.fenetre)
        val_retour_journal = deque(maxlen=self.frequence)
        nb_echecs_subits = 0
        nb_checkpoints = 0
        nb_coups = 0
        bilan_mat = 0
        fins_du_bloc = Counter()  # RaisonFinPartie -> nombre de parties du bloc finies ainsi

        # Contexte moteurs stockfish
        contexte_moteur_ad = (
            chess.engine.SimpleEngine.popen_uci(self.chemin_stockfish)
            if self.stockfish_actif
            else nullcontext(enter_result=None)
        )
        contexte_moteur_eval = (
            chess.engine.SimpleEngine.popen_uci(self.chemin_stockfish)
            if self.potentiel == "stockfish"
            else nullcontext(enter_result=None)
        )

        with (open(self.chemin_journal / self.csv_journal, mode="w", newline="", encoding="utf-8") as j,
              open(self.chemin_journal / self.csv_parties, mode="w", newline="", encoding="utf-8") as p):
            writer = csv.writer(j)
            writer_parties = csv.writer(p)

            # En-têtes (chaque entraînement a son propre dossier, les fichiers sont neufs)
            writer.writerow(COLONNES_JOURNAL + COLONNES_FIN)
            writer_parties.writerow(COLONNES_PARTIES)
            with contexte_moteur_ad as moteur_stockfish_ad, contexte_moteur_eval as moteur_stockfish_eval:
                # Création adversaire, évaluateur et environnement
                if self.potentiel == "stockfish":
                    self.evaluateur = EvaluateurStockfish(moteur_stockfish_eval, self.temps_analyse)
                elif self.potentiel == "materiel":
                    self.evaluateur = self.evaluateur_mat
                else:
                    raise ValueError(f"Evaluation par {self.potentiel} pas encore implémentée")
                if moteur_stockfish_ad is None:
                    self.adversaire = AdversaireAleatoire(self.generateur)
                    self.etat_curriculum = False
                else:
                    self.adversaire = AdversaireStockFish(moteur_stockfish_ad, self.temps_coup, self.niveau_stockfish)
                self.environnement = Environnement(self.couleur, self.adversaire, self.evaluateur, self.gamma)

                # Boucle d'apprentissage
                debut = time.perf_counter()
                for n in range(self.nb_parties):
                    self.epsilon = calcul_epsilon(self.epsilon_final, self.epsilon_initial, n,
                                                  self.parties_decroissance)
                    self.couleur = bool(self.generateur.choice([True, False]))
                    self.environnement.couleur = self.couleur
                    stat_partie = jouer_partie_complete(self.environnement, self.memoire, self.arbitre, self.couleur,
                                                        self.reseaux_principaux, self.lambda_, self.epsilon)
                    val_retour.append(stat_partie.z)
                    val_retour_journal.append(stat_partie.z)
                    nb_echecs_subits += stat_partie.nbr_echecs_subis
                    nb_coups += stat_partie.nbr_coups_joues
                    bilan_mat += stat_partie.bilan_mat_fin_partie
                    fins_du_bloc[stat_partie.raison_fin_partie] += 1
                    writer_parties.writerow([n + 1, "blancs" if self.couleur else "noirs", stat_partie.z,
                                             stat_partie.raison_fin_partie.name, stat_partie.nbr_coups_joues,
                                             stat_partie.bilan_mat_fin_partie, stat_partie.nbr_echecs_subis,
                                             self.epsilon, self.niveau_stockfish])

                    # on regarde si la mémoire est assez remplie pour faire un tirage
                    if self.memoire.taille_actuelle_memoire >= self.debut_apprentissage:
                        for i in range(stat_partie.nbr_coups_joues):
                            lot = self.memoire.tirer_lot(self.taille_lot)
                            err_r, err_c = apprentissage(lot, self.reseaux_principaux, self.gamma, self.reseaux_cibles,
                                                         self.lambda_, self.optimiseur)
                            erreurs_r.append(err_r)
                            erreurs_c.append(err_c)
                            nb_corr += 1
                            if nb_corr%self.frequence_cible == 0:
                                self.reseaux_cibles = deepcopy(self.reseaux_principaux)

                    if self.lagrangien_actif:
                        self.lambda_ = calcul_lambda(self.lambda_, self.eta, self.seuil_danger, stat_partie.nbr_echecs_subis)

                    # On regarde si c'est le moment d'écrire dans le journal
                    retour_moyen_fenetre = sum(val_retour)/len(val_retour)
                    retour_moyen_frequence = sum(val_retour_journal)/len(val_retour_journal)
                    if (n+1)%self.frequence == 0:
                        fin = time.perf_counter()
                        nb_victoires = val_retour_journal.count(1)
                        nb_defaites = val_retour_journal.count(-1)
                        # tout le reste est une nulle, y compris une partie coupée (z non entier)
                        nb_nulles = len(val_retour_journal) - nb_victoires - nb_defaites
                        err_r_moy = sum(erreurs_r)/len(erreurs_r) if len(erreurs_r) > 0 else 0
                        err_c_moy = sum(erreurs_c)/len(erreurs_c) if len(erreurs_c) > 0 else 0
                        proportions_fins = [fins_du_bloc[raison] / self.frequence for raison in RaisonFinPartie]
                        writer.writerow([n+1, retour_moyen_frequence, nb_echecs_subits/self.frequence, self.lambda_,
                                         self.niveau_stockfish, self.epsilon, err_r_moy,  err_c_moy, nb_victoires/self.frequence,
                                        nb_nulles/self.frequence, nb_defaites/self.frequence, nb_coups/self.frequence,
                                         fin-debut, bilan_mat/self.frequence] + proportions_fins)
                        j.flush()  # le journal reste lisible par analyser.py pendant l'entraînement
                        p.flush()
                        debut = time.perf_counter()
                        nb_echecs_subits = 0
                        nb_coups = 0
                        bilan_mat = 0
                        fins_du_bloc.clear()
                        erreurs_r.clear()
                        erreurs_c.clear()

                    # On regarde si l'adversaire doit monter en niveau
                    if decision_curriculum(self.etat_curriculum, retour_moyen_fenetre, self.seuil_montee_curriculum, len(val_retour),
                                           self.fenetre, self.niveau_stockfish):
                        self.niveau_stockfish += self.pas_niveau_curriculum
                        self.niveau_stockfish = min(20, self.niveau_stockfish)
                        self.adversaire.set_niveau_joueur(self.niveau_stockfish)
                        val_retour.clear()

                    # On regarde si on doit sauvegarder le réseau
                    if (n+1)%self.frequence_sauvegarde == 0:
                        nb_checkpoints += 1
                        checkpoint = {
                            'epoch': n,
                            'optimizer_state_dict': self.optimiseur.state_dict(),
                            'lambda': self.lambda_,
                            'niveau_stockfish': self.niveau_stockfish,
                            'epsilon': self.epsilon
                        }
                        for type_p in chess.PIECE_TYPES:
                            checkpoint['model_state_dict'+str(type_p)] = self.reseaux_principaux[type_p].state_dict()
                        torch.save(checkpoint, self.chemin_checkpoints / f"checkpoint_{nb_checkpoints}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="fichier de configuration")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin vers le fichier de configuration YAML (ex: config.yaml)"
    )
    args = parser.parse_args()
    config = lecture_config(args.config)
    entrainement = Entrainement(config)
    entrainement.lancer_entrainement()