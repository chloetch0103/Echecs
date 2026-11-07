"""Tests de l'arbitre : choix du coup (hasard ou maximisation de Q_R − λ·Q_C),
respect des masques, et données renvoyées pour la mémoire.

Les réseaux sont remplacés par des doublures dont on choisit les sorties :
on sait ainsi exactement quel coup l'arbitre doit choisir."""
import chess
import numpy as np
import pytest
import torch

from echecs_ma.agents.arbitre import Arbitre
from echecs_ma.env.encodage import NB_SORTIES, encoder_echiquier, move_to_action

DEPART = chess.STARTING_FEN
APRES_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
SOLDAT_EN_A7 = "8/P6k/8/8/8/8/8/K7 w - - 0 1"


# --- Doublures ----------------------------------------------------------------

class ReseauFixe:
    """Faux réseau : renvoie toujours les mêmes Q_R et Q_C, quelle que soit l'entrée.
    `q_r` et `q_c` sont des dictionnaires {indice de sortie: valeur}, 0 ailleurs."""
    def __init__(self, q_r=None, q_c=None):
        self.q_r = q_r or {}
        self.q_c = q_c or {}

    def __call__(self, x):
        n = x.shape[0]
        sortie_r, sortie_c = torch.zeros(n, NB_SORTIES), torch.zeros(n, NB_SORTIES)
        for k, v in self.q_r.items():
            sortie_r[:, k] = v
        for k, v in self.q_c.items():
            sortie_c[:, k] = v
        return sortie_r, sortie_c


def reseaux(**par_type):
    """Un réseau par type de pièce ; ceux qu'on ne précise pas renvoient des zéros.
    Exemple : reseaux(pion=ReseauFixe({28: 1.0}))"""
    noms = {"pion": chess.PAWN, "cavalier": chess.KNIGHT, "fou": chess.BISHOP,
            "tour": chess.ROOK, "dame": chess.QUEEN, "roi": chess.KING}
    tous = {t: ReseauFixe() for t in chess.PIECE_TYPES}
    for nom, reseau in par_type.items():
        tous[noms[nom]] = reseau
    return tous


def arbitre(graine=0):
    return Arbitre(np.random.default_rng(graine))


def choisir(fen, epsilon=0.0, lambda_=0.0, couleur=None, graine=0, **par_type):
    echiquier = chess.Board(fen)
    couleur = echiquier.turn if couleur is None else couleur
    return arbitre(graine).calcul_coup(echiquier, couleur, reseaux(**par_type), lambda_, epsilon)


# --- Le coup choisi est toujours légal ----------------------------------------

@pytest.mark.parametrize("epsilon", [0.0, 1.0])
@pytest.mark.parametrize("fen", [DEPART, APRES_E4, SOLDAT_EN_A7])
def test_coup_toujours_legal(fen, epsilon):
    decision, _ = choisir(fen, epsilon=epsilon)
    assert decision.coup in chess.Board(fen).legal_moves


def test_masque_respecte_meme_si_le_reseau_favorise_un_coup_illegal():
    # Tous les réseaux donnent une valeur énorme à l'indice 87 (sous-promotion en
    # tour sur la colonne h), illégal au départ : il ne doit jamais être choisi.
    enorme = {87: 1e9}
    decision, _ = choisir(DEPART, pion=ReseauFixe(enorme), cavalier=ReseauFixe(enorme),
                          fou=ReseauFixe(enorme), tour=ReseauFixe(enorme),
                          dame=ReseauFixe(enorme), roi=ReseauFixe(enorme))
    assert decision.coup in chess.Board(DEPART).legal_moves


# --- Maximisation de Q_R − λ·Q_C ----------------------------------------------

def test_choisit_le_coup_le_mieux_note():
    # L'indice 28 (case e4) n'est atteignable que par le soldat e2
    decision, _ = choisir(DEPART, pion=ReseauFixe(q_r={28: 1.0}))
    assert decision.coup == chess.Move.from_uci("e2e4")


def test_compare_les_differents_types_de_pieces():
    # Le cavalier (f3 = indice 21) est mieux noté que tous les soldats
    decision, _ = choisir(DEPART, pion=ReseauFixe(q_r={28: 1.0}), cavalier=ReseauFixe(q_r={21: 2.0}))
    assert decision.coup == chess.Move.from_uci("g1f3")


@pytest.mark.parametrize("lambda_, attendu", [(0.0, "e2e4"), (1.0, "e2e3")])
def test_lambda_penalise_le_danger(lambda_, attendu):
    # e4 rapporte plus (1,0 contre 0,8) mais est plus dangereux (1,0 contre 0).
    # λ = 0 : on ignore le danger → e4.  λ = 1 : 1,0 − 1,0 = 0 < 0,8 → e3.
    soldat = ReseauFixe(q_r={28: 1.0, 20: 0.8}, q_c={28: 1.0})
    decision, _ = choisir(DEPART, lambda_=lambda_, pion=soldat)
    assert decision.coup == chess.Move.from_uci(attendu)


def test_miroir_pour_les_noirs():
    # L'indice 28 correspond, vu des Noirs, au soldat e7 qui va en e5
    decision, _ = choisir(APRES_E4, pion=ReseauFixe(q_r={28: 1.0}))
    assert decision.coup == chess.Move.from_uci("e7e5")


def test_sous_promotion_choisie_par_le_reseau():
    # Indice 72 = promotion en cavalier sur la colonne a
    decision, _ = choisir(SOLDAT_EN_A7, pion=ReseauFixe(q_r={72: 5.0}))
    assert decision.coup == chess.Move.from_uci("a7a8n")


def test_promotion_en_dame_choisie_par_le_reseau():
    decision, _ = choisir(SOLDAT_EN_A7, pion=ReseauFixe(q_r={56: 5.0}))
    assert decision.coup == chess.Move.from_uci("a7a8q")


def test_valeurs_negatives():
    # Même si tous les coups sont mal notés, l'arbitre prend le moins mauvais
    toutes_negatives = {k: -10.0 for k in range(NB_SORTIES)}
    toutes_negatives[28] = -1.0
    decision, _ = choisir(DEPART, pion=ReseauFixe(q_r=toutes_negatives),
                          cavalier=ReseauFixe(q_r={k: -10.0 for k in range(NB_SORTIES)}))
    assert decision.coup == chess.Move.from_uci("e2e4")


# --- Exploration (ε) ----------------------------------------------------------

def test_epsilon_un_explore_tous_les_coups():
    # Avec ε = 1, sur assez de tirages, chacun des 20 coups de départ doit sortir
    a = arbitre(graine=1)
    echiquier = chess.Board()
    vus = {a.calcul_coup(echiquier, chess.WHITE, reseaux(), 0.0, 1.0)[0].coup for _ in range(400)}
    assert vus == set(echiquier.legal_moves)


def test_epsilon_zero_n_explore_jamais():
    # Avec ε = 0, toujours le même coup (le mieux noté)
    a = arbitre(graine=1)
    echiquier = chess.Board()
    r = reseaux(pion=ReseauFixe(q_r={28: 1.0}))
    coups = {a.calcul_coup(echiquier, chess.WHITE, r, 0.0, 0.0)[0].coup for _ in range(50)}
    assert coups == {chess.Move.from_uci("e2e4")}


def test_meme_graine_memes_tirages():
    echiquier = chess.Board()
    a, b = arbitre(graine=7), arbitre(graine=7)
    tirages_a = [a.calcul_coup(echiquier, chess.WHITE, reseaux(), 0.0, 1.0)[0].coup for _ in range(30)]
    tirages_b = [b.calcul_coup(echiquier, chess.WHITE, reseaux(), 0.0, 1.0)[0].coup for _ in range(30)]
    assert tirages_a == tirages_b


# --- L'action renvoyée --------------------------------------------------------

@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_action_correspond_au_coup(epsilon):
    decision, _ = choisir(DEPART, epsilon=epsilon, pion=ReseauFixe(q_r={28: 1.0}))
    assert decision.action == move_to_action(chess.Board(), decision.coup, chess.WHITE)


def test_action_d_une_promotion_garde_le_type_soldat():
    decision, _ = choisir(SOLDAT_EN_A7, pion=ReseauFixe(q_r={72: 5.0}))
    assert decision.action.type_piece == chess.PAWN
    assert decision.action.case_depart == chess.A7
    assert decision.action.indice == 72


# --- La position renvoyée (pour la mémoire) -----------------------------------

@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_position_complete_dans_les_deux_branches(epsilon):
    # Hasard ou maximisation : la position doit toujours décrire toutes nos pièces
    echiquier = chess.Board()
    _, position = choisir(DEPART, epsilon=epsilon)

    assert len(position.masques) == len(position.dicos) == len(position.pieces_t) == len(position.cases_p) == 16
    assert set(position.cases_p) == {c for c, p in echiquier.piece_map().items() if p.color == chess.WHITE}
    np.testing.assert_array_equal(position.plans, encoder_echiquier(echiquier, chess.WHITE))


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_position_types_et_cases_coherents(epsilon):
    echiquier = chess.Board()
    _, position = choisir(DEPART, epsilon=epsilon)
    for type_piece, case in zip(position.pieces_t, position.cases_p):
        assert echiquier.piece_type_at(case) == type_piece


@pytest.mark.parametrize("fen", [DEPART, APRES_E4, SOLDAT_EN_A7])
def test_position_couvre_tous_les_coups(fen):
    # Réunis, les dictionnaires de la position redonnent exactement les coups légaux
    echiquier = chess.Board(fen)
    _, position = choisir(fen)
    decodes = [coup for d in position.dicos for coup in d.values()]
    assert len(decodes) == len(set(decodes))
    assert set(decodes) == set(echiquier.legal_moves)
    assert sum(m.sum() for m in position.masques) == echiquier.legal_moves.count()


# --- Erreurs ------------------------------------------------------------------

@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_reseau_manquant(epsilon):
    r = reseaux()
    del r[chess.BISHOP]
    with pytest.raises(ValueError):
        arbitre().calcul_coup(chess.Board(), chess.WHITE, r, 0.0, epsilon)


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_pas_notre_trait(epsilon):
    with pytest.raises(ValueError):
        arbitre().calcul_coup(chess.Board(), chess.BLACK, reseaux(), 0.0, epsilon)


# --- Avec les vrais réseaux ---------------------------------------------------

def test_avec_les_vrais_reseaux():
    # Réseaux non entraînés : le coup n'a aucun sens, mais tout doit s'emboîter
    from echecs_ma.agents.reseaux import Reseau
    torch.manual_seed(0)
    vrais = {t: Reseau([8]) for t in chess.PIECE_TYPES}
    echiquier = chess.Board()
    decision, position = arbitre().calcul_coup(echiquier, chess.WHITE, vrais, 0.5, 0.0)
    assert decision.coup in echiquier.legal_moves
    assert len(position.cases_p) == 16