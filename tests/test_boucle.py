"""Tests de la boucle de jeu : une partie complète remplit correctement la mémoire.

Sans Stockfish : adversaire aléatoire ou scripté, évaluateur nul (Φ = 0, donc la
récompense finale vaut exactement z), et réseaux factices."""
import chess
import numpy as np
import pytest
import torch

from echecs_ma.agents.arbitre import Arbitre
from echecs_ma.entrainement.boucle import RaisonFinPartie, StatPartie, jouer_partie_complete
from echecs_ma.entrainement.memoire import Memoire
from echecs_ma.env.adversaire import Adversaire, Evaluateur
from echecs_ma.env.encodage import NB_SORTIES, case_vers_indices
from echecs_ma.env.environnement import Environnement

# --- Doublures ----------------------------------------------------------------

class AdversaireAleatoire(Adversaire):
    def __init__(self, graine=0):
        self.gen = np.random.default_rng(graine)

    def jouer(self, echiquier):
        coups = list(echiquier.legal_moves)
        return coups[self.gen.integers(len(coups))]


class AdversaireScripte(Adversaire):
    def __init__(self, coups):
        self.coups = [chess.Move.from_uci(c) for c in coups]

    def jouer(self, echiquier):
        return self.coups.pop(0)


class EvaluateurNul(Evaluateur):
    def potentiel(self, echiquier, couleur):
        return 0.0


class ReseauFixe:
    """Faux réseau : Q_R vaut q_r[k] à l'indice k (0 ailleurs), Q_C vaut 0."""
    def __init__(self, q_r=None):
        self.q_r = q_r or {}

    def __call__(self, x):
        sortie_r, sortie_c = torch.zeros(x.shape[0], NB_SORTIES), torch.zeros(x.shape[0], NB_SORTIES)
        for k, v in self.q_r.items():
            sortie_r[:, k] = v
        return sortie_r, sortie_c


def reseaux(pion=None):
    tous = {t: ReseauFixe() for t in chess.PIECE_TYPES}
    if pion is not None:
        tous[chess.PAWN] = pion
    return tous


def jouer(couleur=chess.WHITE, adversaire=None, memoire=None, graine=0, epsilon=1.0, r=None):
    """Joue une partie ; par défaut agents au hasard (ε = 1) contre un adversaire au hasard."""
    memoire = memoire or Memoire(2000, np.random.default_rng(graine))
    env = Environnement(couleur, adversaire or AdversaireAleatoire(graine), EvaluateurNul(), gamma=0.99)
    arbitre = Arbitre(np.random.default_rng(graine))
    stats = jouer_partie_complete(env, memoire, arbitre, couleur, r or reseaux(), 0.0, epsilon)
    return stats, memoire, env


def lignes(memoire):
    """Les lignes remplies, dans l'ordre d'ajout (valable tant que la mémoire n'a pas débordé)."""
    return range(memoire.taille_actuelle_memoire)


# --- get_z --------------------------------------------------------------------

def env_sur(fen, couleur=chess.WHITE):
    env = Environnement(couleur, AdversaireAleatoire(), EvaluateurNul(), gamma=0.99)
    env.reset(fen=fen)
    return env


@pytest.mark.parametrize("couleur, attendu", [(chess.WHITE, -1), (chess.BLACK, 1)])
def test_get_z_mat(couleur, attendu):
    # Mat du couloir : les Blancs sont matés
    env = env_sur("6k1/8/8/8/8/8/5PPP/r5K1 w - - 0 1")
    env.couleur = couleur
    assert env.get_z() == attendu


def test_get_z_pat():
    env = env_sur("7k/8/6Q1/8/8/8/8/K7 b - - 0 1", chess.BLACK)
    assert env.get_z() == 0


def test_get_z_regle_des_50_coups():
    # 100 demi-coups sans prise ni coup de soldat : l'environnement arrête la partie
    # (claim_draw=True), donc get_z doit aussi y voir une nulle.
    env = env_sur("4k3/8/8/8/8/8/8/4K2R w - - 100 80")
    assert env.get_z() == 0


# --- Une partie au hasard -----------------------------------------------------

@pytest.mark.parametrize("graine", [0, 1, 2])
@pytest.mark.parametrize("couleur", [chess.WHITE, chess.BLACK])
def test_une_transition_par_coup_joue(couleur, graine):
    stats, memoire, _ = jouer(couleur, graine=graine)
    assert isinstance(stats, StatPartie)
    assert memoire.compteur_transitions == stats.nbr_coups_joues


@pytest.mark.parametrize("graine", [0, 1, 2])
@pytest.mark.parametrize("couleur", [chess.WHITE, chess.BLACK])
def test_une_seule_fin_et_c_est_la_derniere(couleur, graine):
    _stats, memoire, _ = jouer(couleur, graine=graine)
    fins = memoire.memoire_indicateur_fin[:memoire.taille_actuelle_memoire]
    assert fins.sum() == 1
    assert fins[memoire.taille_actuelle_memoire - 1]


@pytest.mark.parametrize("graine", [0, 1, 2])
@pytest.mark.parametrize("couleur", [chess.WHITE, chess.BLACK])
def test_recompense_finale_egale_au_resultat(couleur, graine):
    # Φ = 0 partout : la dernière transition a r = z, les autres r = 0
    stats, memoire, _ = jouer(couleur, graine=graine)
    n = memoire.taille_actuelle_memoire
    assert stats.z in (-1, 0, 1)
    assert memoire.memoire_r[n - 1] == stats.z
    assert (memoire.memoire_r[:n - 1] == 0).all()


@pytest.mark.parametrize("graine", [0, 1, 2])
def test_echecs_subis_coherents_avec_la_memoire(graine):
    stats, memoire, _ = jouer(graine=graine)
    assert memoire.memoire_c[:memoire.taille_actuelle_memoire].sum() == stats.nbr_echecs_subis


@pytest.mark.parametrize("graine", [0, 1, 2])
@pytest.mark.parametrize("couleur", [chess.WHITE, chess.BLACK])
def test_les_transitions_s_enchainent(couleur, graine):
    # Le s' d'une transition est le s de la suivante : c'est ce qui prouve
    # qu'aucune transition n'est décalée d'un coup
    _, memoire, _ = jouer(couleur, graine=graine)
    n = memoire.taille_actuelle_memoire
    for i in range(n - 1):
        np.testing.assert_array_equal(memoire.memoire_plans_suiv[i], memoire.memoire_plans[i + 1])
        assert memoire.memoire_nb_pieces_suiv[i] > 0


@pytest.mark.parametrize("graine", [0, 1, 2])
@pytest.mark.parametrize("couleur", [chess.WHITE, chess.BLACK])
def test_l_action_correspond_a_une_de_nos_pieces_dans_s(couleur, graine):
    # Dans les plans de s, la case de départ doit contenir une de nos pièces du bon type
    _, memoire, _ = jouer(couleur, graine=graine)
    for i in lignes(memoire):
        type_piece = int(memoire.memoire_type_action[i])
        ligne, col = case_vers_indices(int(memoire.memoire_case_dep[i]), couleur)
        assert memoire.memoire_plans[i, type_piece - 1, ligne, col] == 1


@pytest.mark.parametrize("graine", [0, 1, 2])
def test_l_action_est_autorisee_par_le_masque_de_s(graine):
    # Le coup joué depuis s doit figurer dans le masque de sa pièce, tel que stocké
    # comme s' dans la transition précédente
    _, memoire, _ = jouer(graine=graine)
    for i in range(1, memoire.taille_actuelle_memoire):
        cases = memoire.memoire_cases_suiv[i - 1, :memoire.memoire_nb_pieces_suiv[i - 1]].tolist()
        k = cases.index(int(memoire.memoire_case_dep[i]))
        assert memoire.memoire_masques_suiv[i - 1, k, memoire.memoire_indice[i]]


# --- Une partie scriptée ------------------------------------------------------

def test_mat_du_lion():
    # Les agents (Blancs) jouent f3 puis g4, l'adversaire e5 puis Dh4# :
    # 2 transitions, la dernière avec fin, r = −1 et c = 1 (le mat est un échec)
    pion = ReseauFixe(q_r={21: 2.0, 30: 1.0})               # f3 (21) d'abord, puis g4 (30)
    stats, memoire, env = jouer(adversaire=AdversaireScripte(["e7e5", "d8h4"]), epsilon=0.0,
                                r=reseaux(pion=pion))
    assert [m.uci() for m in env.echiquier.move_stack] == ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert stats == StatPartie(z=-1, nbr_echecs_subis=1, nbr_coups_joues=2, bilan_mat_fin_partie=0.0, raison_fin_partie=RaisonFinPartie.CHECKMATE)
    assert memoire.compteur_transitions == 2
    assert memoire.memoire_indicateur_fin.tolist()[:2] == [False, True]
    assert memoire.memoire_r[1] == -1 and memoire.memoire_c[1] == 1
    assert memoire.memoire_case_dep[:2].tolist() == [chess.F2, chess.G2]
    assert memoire.memoire_indice[:2].tolist() == [21, 30]


# --- Plusieurs parties --------------------------------------------------------

def test_plusieurs_parties_dans_la_meme_memoire():
    # La mémoire et l'environnement sont réutilisés : les transitions s'accumulent,
    # et chaque partie ajoute exactement une fin
    memoire = Memoire(5000, np.random.default_rng(0))
    env = Environnement(chess.WHITE, AdversaireAleatoire(0), EvaluateurNul(), gamma=0.99)
    arbitre = Arbitre(np.random.default_rng(0))
    total = 0
    for _ in range(3):
        stats = jouer_partie_complete(env, memoire, arbitre, chess.WHITE, reseaux(), 0.0, 1.0)
        total += stats.nbr_coups_joues
    assert memoire.compteur_transitions == total
    assert memoire.memoire_indicateur_fin[:total].sum() == 3