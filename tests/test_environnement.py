"""Tests de l'environnement, sans Stockfish.

On remplace l'adversaire et l'évaluateur par des versions prévisibles :
- AdversaireScripte joue une liste de coups fixée à l'avance ;
- EvaluateurNul renvoie toujours phi = 0, donc r = z en fin de partie ;
- EvaluateurScripte renvoie une suite de valeurs de phi choisies.
"""
import chess
import pytest

from echecs_ma.env.adversaire import Adversaire, Evaluateur
from echecs_ma.env.environnement import Environnement


# --- Doublures pour les tests ------------------------------------------------

class AdversaireScripte(Adversaire):
    """Joue les coups donnés, dans l'ordre (notation UCI : "e7e5")."""
    def __init__(self, coups):
        self.coups = [chess.Move.from_uci(c) for c in coups]

    def jouer(self, echiquier):
        coup = self.coups.pop(0)
        assert coup in echiquier.legal_moves, f"coup scripté illégal : {coup}"
        return coup


class EvaluateurNul(Evaluateur):
    def potentiel(self, echiquier, couleur):
        return 0.0


class EvaluateurScripte(Evaluateur):
    """Renvoie les valeurs de phi données, dans l'ordre."""
    def __init__(self, valeurs):
        self.valeurs = list(valeurs)

    def potentiel(self, echiquier, couleur):
        return self.valeurs.pop(0)


def creer_env(couleur=chess.WHITE, coups_adverses=(), evaluateur=None, gamma=0.99, fen=None):
    env = Environnement(couleur, AdversaireScripte(coups_adverses),
                        evaluateur or EvaluateurNul(), gamma)
    env.reset(fen=fen)
    return env


def coup(uci):
    return chess.Move.from_uci(uci)


# --- Coups légaux par agent --------------------------------------------------

def test_coups_legaux_position_initiale():
    env = creer_env()
    env.compute_coups_legaux_par_agent()
    agents = env.coups_legaux_par_agent

    assert len(agents) == 6                                  # une clé par pièce blanche
    assert len(agents[chess.PAWN][chess.E2]) == 2           # e3 et e4
    assert len(agents[chess.KNIGHT][chess.G1]) == 2         # f3 et h3
    assert agents[chess.ROOK][chess.A1] == []               # bloquée
    assert agents[chess.KING][chess.E1] == []
    assert sum(len(c) for val in agents.values() for _, c in val.items()) == 20         # 20 coups légaux au départ


def test_promotion_donne_quatre_coups_au_soldat():
    env = creer_env(fen="8/P6k/8/8/8/8/8/K7 w - - 0 1")
    env.compute_coups_legaux_par_agent()
    coups_soldat = env.coups_legaux_par_agent[chess.PAWN][chess.A7]

    promotions = {c.promotion for c in coups_soldat}
    assert promotions == {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}


def test_soldat_promu_change_de_type():
    env = creer_env(coups_adverses=["h7h6"], fen="8/P6k/8/8/8/8/8/K7 w - - 0 1")
    env.step(coup("a7a8q"))
    env.compute_coups_legaux_par_agent()

    assert chess.QUEEN in env.coups_legaux_par_agent
    assert chess.A8 in env.coups_legaux_par_agent[chess.QUEEN]
    assert chess.PAWN in env.coups_legaux_par_agent
    assert chess.A7 not in env.coups_legaux_par_agent[chess.PAWN]


# --- Fins de partie ----------------------------------------------------------

def test_mat_par_l_agent_donne_plus_un():
    env = creer_env(fen="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")   # mat du couloir
    _, r, c, fin = env.step(coup("a1a8"))
    assert fin
    assert r == 1
    assert c == 0          # l'adversaire n'a pas joué


def test_mat_par_l_adversaire_donne_moins_un():
    # Mat du berger inversé ("mat du lion") : 1. f3 e5 2. g4 Dh4#
    env = creer_env(coups_adverses=["e7e5", "d8h4"])
    env.step(coup("f2f3"))
    _, r, c, fin = env.step(coup("g2g4"))
    assert fin
    assert r == -1
    assert c == 1          # un mat est aussi un échec


def test_pat_donne_zero():
    env = creer_env(fen="7k/8/8/8/8/8/6Q1/K7 w - - 0 1")
    _, r, c, fin = env.step(coup("g2g6"))                      # le Roi noir n'a plus de coup
    assert fin
    assert r == 0


def test_step_apres_la_fin_leve_une_erreur():
    env = creer_env(fen="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    env.step(coup("a1a8"))
    with pytest.raises(RuntimeError):
        env.step(coup("g1f1"))


# --- Récompense et coût en cours de partie -----------------------------------

def test_recompense_de_faconnage():
    # phi(s0) = 0.2 au reset, puis phi(s1) = 0.5 après le premier échange de coups
    env = creer_env(coups_adverses=["e7e5"], evaluateur=EvaluateurScripte([0.2, 0.5]))
    _, r, c, fin = env.step(coup("e2e4"))
    assert not fin
    assert r == pytest.approx(0.99 * 0.5 - 0.2)
    assert c == 0


def test_fin_de_partie_utilise_phi_suivant_nul():
    # phi(s0) = 0.3, puis l'agent mate : r = z − phi(s) = 1 − 0.3, sans gamma*phi(s')
    env = creer_env(evaluateur=EvaluateurScripte([0.3]),
                    fen="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    _, r, _, _ = env.step(coup("a1a8"))
    assert r == pytest.approx(0.7)


def test_echec_de_l_adversaire_donne_c_egal_un():
    # 1. e4 e5 2. f3 Dh4+ : échec, mais pas mat (Re2 ou g3 possibles)
    env = creer_env(coups_adverses=["e7e5", "d8h4"])
    env.step(coup("e2e4"))
    _, _, c, fin = env.step(coup("f2f3"))
    assert c == 1
    assert not fin


# --- Couleur des agents ------------------------------------------------------

def test_agents_noirs_l_adversaire_commence():
    env = creer_env(couleur=chess.BLACK, coups_adverses=["e2e4"])
    assert env.echiquier.turn == chess.BLACK
    assert env.echiquier.move_stack == [coup("e2e4")]