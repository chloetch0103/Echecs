"""Tests de l'environnement, sans Stockfish.

On remplace l'adversaire et l'évaluateur par des versions prévisibles :
- AdversaireScripte joue une liste de coups fixée à l'avance ;
- EvaluateurNul renvoie toujours phi = 0, donc r = z en fin de partie ;
- EvaluateurScripte renvoie une suite de valeurs de phi choisies.
"""
import chess
import pytest
import math

from echecs_ma.env.adversaire import Adversaire, Evaluateur, EvaluateurMateriel
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
    _, r, _, fin = env.step(coup("g2g6"))                      # le Roi noir n'a plus de coup
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


# --- Tests des évaluateurs de position (potentiel phi), sans Stockfish ---


SANS_DAME_NOIRE = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SANS_PION_NOIR = "rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ENORME_AVANTAGE = "QQQQ3k/8/8/8/8/8/8/QQQ1K3 w - - 0 1"      # 7 dames blanches contre un Roi seul


def phi(fen, couleur=chess.WHITE):
    return EvaluateurMateriel().potentiel(chess.Board(fen), couleur)


# --- Classes abstraites ------------------------------------------------------

def test_on_ne_peut_pas_creer_un_evaluateur_abstrait():
    with pytest.raises(TypeError):
        Evaluateur()


def test_on_ne_peut_pas_creer_un_adversaire_abstrait():
    with pytest.raises(TypeError):
        Adversaire()


def test_evaluateur_materiel_est_un_evaluateur():
    assert isinstance(EvaluateurMateriel(), Evaluateur)


# --- Valeurs du potentiel ----------------------------------------------------

@pytest.mark.parametrize("couleur", [chess.WHITE, chess.BLACK])
def test_position_initiale_potentiel_nul(couleur):
    assert phi(chess.STARTING_FEN, couleur) == 0


def test_une_dame_d_avance():
    # tanh(9 / 10) ≈ 0,716
    assert phi(SANS_DAME_NOIRE) == pytest.approx(math.tanh(0.9))
    assert phi(SANS_DAME_NOIRE) == pytest.approx(0.716, abs=1e-3)


@pytest.mark.parametrize("fen", [SANS_DAME_NOIRE, SANS_PION_NOIR, ENORME_AVANTAGE])
def test_symetrie_entre_les_deux_camps(fen):
    # Ce qui est un avantage pour l'un est un désavantage pour l'autre
    assert phi(fen, chess.BLACK) == pytest.approx(-phi(fen, chess.WHITE))


def test_plus_de_materiel_donne_un_plus_grand_potentiel():
    assert 0 < phi(SANS_PION_NOIR) < phi(SANS_DAME_NOIRE) < phi(ENORME_AVANTAGE)


@pytest.mark.parametrize("couleur", [chess.WHITE, chess.BLACK])
def test_potentiel_reste_strictement_entre_moins_un_et_un(couleur):
    # Même avec 63 points d'avance, Φ ne doit jamais atteindre ±1 (la valeur de z)
    assert -1 < phi(ENORME_AVANTAGE, couleur) < 1


def test_renvoie_un_nombre():
    assert isinstance(float(phi(SANS_DAME_NOIRE)), float)


def test_dans_l_environnement_le_facconnage_suit_le_materiel():
    # 1. e4 d5 2. exd5 : les Blancs gagnent un pion, phi passe de 0 à tanh(0,1)
    env = Environnement(chess.WHITE, AdversaireScripte(["d7d5", "a7a6"]), EvaluateurMateriel(), gamma=0.99)
    env.reset()
    assert env.phi == 0
    env.step(chess.Move.from_uci("e2e4"))
    _, r, _, _ = env.step(chess.Move.from_uci("e4d5"))
    assert env.phi == pytest.approx(math.tanh(0.1))
    assert r == pytest.approx(0.99 * math.tanh(0.1) - 0)