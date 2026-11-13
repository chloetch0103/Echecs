"""Tests de l'étape d'apprentissage.

- prediction_reseau_cible : testée de façon exacte, avec des réseaux cibles factices
  dont on choisit les sorties.
- apprentissage : testée par propriétés (seul le bon réseau bouge, les réseaux cibles
  ne bougent pas, l'erreur baisse, les valeurs convergent vers la bonne cible)."""
import chess
import numpy as np
import pytest
import torch

from echecs_ma.agents.arbitre import Arbitre, ArbitreDecision
from echecs_ma.entrainement.apprentissage import apprentissage, cibles_suivantes, prediction_reseau_cible
from echecs_ma.entrainement.memoire import NB_PIECES_MAX, Memoire
from echecs_ma.env.encodage import NB_PLANS, NB_PLANS_ENTREE, NB_SORTIES, Action, encoder_echiquier

APRES_E4_E5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"


# --- Doublures ----------------------------------------------------------------

class ReseauFixe:
    """Faux réseau : Q_R vaut q_r[k] et Q_C vaut q_c[k] à l'indice k, `defaut` ailleurs."""
    def __init__(self, q_r=None, q_c=None, defaut=0.0):
        self.q_r, self.q_c, self.defaut = q_r or {}, q_c or {}, defaut

    def __call__(self, x):
        n = x.shape[0]
        sortie_r = torch.full((n, NB_SORTIES), self.defaut)
        sortie_c = torch.full((n, NB_SORTIES), self.defaut)
        for k, v in self.q_r.items():
            sortie_r[:, k] = v
        for k, v in self.q_c.items():
            sortie_c[:, k] = v
        return sortie_r, sortie_c


class ReseauQuiLitLaCase:
    """Faux réseau : Q_R vaut 1 à l'indice de la case désignée par le plan de la pièce
    (8 × ligne + colonne, dans le repère de nos agents), 0 ailleurs."""
    def __call__(self, x):
        plan_piece = np.asarray(x)[:, NB_PLANS].reshape(x.shape[0], 64)
        sortie_r = torch.zeros(x.shape[0], NB_SORTIES)
        for b in range(x.shape[0]):
            sortie_r[b, int(np.argmax(plan_piece[b]))] = 1.0
        return sortie_r, torch.zeros(x.shape[0], NB_SORTIES)


def reseaux_fixes():
    """Un faux réseau par type de pièce, qui renvoie des zéros ; on remplace ceux qu'on veut avec |."""
    return {t: ReseauFixe() for t in chess.PIECE_TYPES}


def masque(*indices):
    m = np.zeros(NB_SORTIES, dtype=bool)
    m[list(indices)] = True
    return m


def cible(reseaux_cibles, pieces, lambda_=0.0, couleur=chess.WHITE, fantomes=0):
    """Appelle prediction_reseau_cible sur une liste de pièces [(type, case, masque), ...].
    `fantomes` ajoute des cases vides (type 0) au-delà des vraies pièces, comme dans la mémoire."""
    n = len(pieces)
    types = np.zeros(NB_PIECES_MAX, dtype=np.uint8)
    cases = np.zeros(NB_PIECES_MAX, dtype=np.uint8)
    masques = np.zeros((NB_PIECES_MAX, NB_SORTIES), dtype=bool)
    for k, (t, c, m) in enumerate(pieces):
        types[k], cases[k], masques[k] = t, c, m
    if fantomes:
        masques[n:n + fantomes] = True        # des données fantômes qu'il faut ignorer
    plans = encoder_echiquier(chess.Board(), chess.WHITE)
    return prediction_reseau_cible(reseaux_cibles, lambda_, masques, plans, types, cases, couleur, n)


# =============================================================================
# prediction_reseau_cible
# =============================================================================

def test_renvoie_deux_nombres():
    q_r, q_c = cible(reseaux_fixes(), [(chess.PAWN, chess.E2, masque(20, 28))])
    assert isinstance(q_r, float) and isinstance(q_c, float)


def test_valeurs_au_meilleur_coup():
    # Le soldat a deux coups : 20 (Q_R = 0,3) et 28 (Q_R = 0,9, Q_C = 0,4) → a' = 28
    soldat = ReseauFixe(q_r={20: 0.3, 28: 0.9}, q_c={28: 0.4})
    q_r, q_c = cible(reseaux_fixes() | {chess.PAWN: soldat}, [(chess.PAWN, chess.E2, masque(20, 28))])
    assert q_r == pytest.approx(0.9)
    assert q_c == pytest.approx(0.4)


def test_chaque_piece_utilise_le_reseau_de_son_type():
    # Le cavalier propose mieux que le soldat : a' est le coup du cavalier
    r = reseaux_fixes() | {chess.PAWN: ReseauFixe(q_r={28: 1.0}), chess.KNIGHT: ReseauFixe(q_r={21: 2.0}, q_c={21: 0.7})}
    q_r, q_c = cible(r, [(chess.PAWN, chess.E2, masque(28)), (chess.KNIGHT, chess.G1, masque(21))])
    assert (q_r, q_c) == pytest.approx((2.0, 0.7))


@pytest.mark.parametrize("lambda_, attendu", [(0.0, (1.0, 1.0)), (1.0, (0.8, 0.0))])
def test_lambda_change_le_coup_retenu(lambda_, attendu):
    # Coup 28 : Q_R = 1,0 mais Q_C = 1,0.  Coup 20 : Q_R = 0,8, Q_C = 0.
    # λ = 0 → 28 ;  λ = 1 → 1,0 − 1,0 = 0 < 0,8 → 20
    soldat = ReseauFixe(q_r={28: 1.0, 20: 0.8}, q_c={28: 1.0})
    q_r, q_c = cible(reseaux_fixes() | {chess.PAWN: soldat}, [(chess.PAWN, chess.E2, masque(20, 28))], lambda_)
    assert (q_r, q_c) == pytest.approx(attendu)


def test_les_coups_masques_sont_ignores():
    # L'indice 87 a une valeur énorme mais n'est pas dans le masque
    soldat = ReseauFixe(q_r={87: 1e9, 28: 0.5})
    q_r, _ = cible(reseaux_fixes() | {chess.PAWN: soldat}, [(chess.PAWN, chess.E2, masque(28))])
    assert q_r == pytest.approx(0.5)


def test_valeurs_negatives():
    soldat = ReseauFixe(q_r={20: -3.0, 28: -1.0}, defaut=-10.0)
    q_r, _ = cible(reseaux_fixes() | {chess.PAWN: soldat}, [(chess.PAWN, chess.E2, masque(20, 28))])
    assert q_r == pytest.approx(-1.0)


def test_les_cases_fantomes_sont_ignorees():
    # Au-delà de nb_pieces_suiv, les types valent 0 : aucun réseau ne doit être demandé
    q_r, _ = cible(reseaux_fixes() | {chess.PAWN: ReseauFixe(q_r={28: 0.5})},
                   [(chess.PAWN, chess.E2, masque(28))], fantomes=3)
    assert q_r == pytest.approx(0.5)


@pytest.mark.parametrize("case, couleur, valeur", [
    (chess.E2, chess.WHITE, 1.0),     # e2 vue des Blancs : ligne 1, colonne 4 → indice 12
    (chess.E7, chess.BLACK, 1.0),     # e7 vue des Noirs : même position grâce au miroir
    (chess.E7, chess.WHITE, 0.0),     # e7 vue des Blancs : ligne 6 → indice 52, pas 12
])
def test_le_plan_de_la_piece_utilise_sa_case_et_la_couleur(case, couleur, valeur):
    r = {t: ReseauQuiLitLaCase() for t in chess.PIECE_TYPES}
    q_r, _ = cible(r, [(chess.PAWN, case, masque(12))], couleur=couleur)
    assert q_r == pytest.approx(valeur)


# =============================================================================
# apprentissage (ces tests demandent le vrai PyTorch)
# =============================================================================

def position_de(echiquier, couleur):
    """La position encodée telle que l'arbitre la renvoie (branche hasard : aucun réseau appelé)."""
    _, position = Arbitre(np.random.default_rng(0)).calcul_coup(
        echiquier, couleur, reseaux_fixes(), 0.0, 1.0)
    return position


def lot_fixe(fin, r=0.0, c=0.0, taille=16):
    """Un lot de `taille` copies de la même transition : le soldat e2 joue e4 (indice 28)."""
    memoire = Memoire(taille, np.random.default_rng(0))
    s = position_de(chess.Board(), chess.WHITE)
    s_suiv = None if fin else position_de(chess.Board(APRES_E4_E5), chess.WHITE)
    decision = ArbitreDecision(coup=chess.Move.from_uci("e2e4"), action=Action(chess.PAWN, chess.E2, 28))
    for _ in range(taille):
        memoire.stocker_en_memoire(r, c, s, decision, s_suiv, chess.WHITE)
    return memoire.tirer_lot(taille)


def vrais_reseaux(graine=0):
    from echecs_ma.agents.reseaux import Reseau
    torch.manual_seed(graine)
    return {t: Reseau([16]) for t in chess.PIECE_TYPES}


def copie_des_poids(reseau):
    return [p.detach().clone() for p in reseau.parameters()]


def poids_identiques(reseau, poids):
    return all(torch.equal(p, q) for p, q in zip(reseau.parameters(), poids))


def prediction(reseaux, lot):
    """Q_R et Q_C du soldat e2 pour le coup 28, avec les vrais réseaux."""
    from echecs_ma.env.encodage import ajouter_position_aux_plans
    x = ajouter_position_aux_plans(lot.plans[0], int(lot.case_dep[0]), bool(lot.couleur[0]))
    with torch.no_grad():
        q_r, q_c = reseaux[chess.PAWN](torch.from_numpy(x).unsqueeze(0))
    return q_r[0, 28].item(), q_c[0, 28].item()


def test_renvoie_deux_erreurs_positives():
    reseaux = vrais_reseaux()
    optimiseur = torch.optim.Adam([p for r in reseaux.values() for p in r.parameters()], lr=1e-3)
    erreur_r, erreur_c = apprentissage(lot_fixe(fin=True, r=1.0), reseaux, 0.99, vrais_reseaux(1), 0.0, optimiseur)
    assert isinstance(erreur_r, float) and isinstance(erreur_c, float)
    assert erreur_r >= 0 and erreur_c >= 0


def test_les_predictions_gardent_le_lien_avec_les_reseaux():
    # Si le lien était coupé (torch.tensor au lieu de torch.stack), les poids ne bougeraient pas
    reseaux = vrais_reseaux()
    avant = copie_des_poids(reseaux[chess.PAWN])
    optimiseur = torch.optim.Adam([p for r in reseaux.values() for p in r.parameters()], lr=1e-2)
    apprentissage(lot_fixe(fin=True, r=1.0), reseaux, 0.99, vrais_reseaux(1), 0.0, optimiseur)
    assert not poids_identiques(reseaux[chess.PAWN], avant)


def test_seul_le_reseau_qui_a_joue_change():
    # Toutes les transitions du lot sont des coups de soldat
    reseaux = vrais_reseaux()
    avant = {t: copie_des_poids(r) for t, r in reseaux.items()}
    optimiseur = torch.optim.Adam([p for r in reseaux.values() for p in r.parameters()], lr=1e-2)
    apprentissage(lot_fixe(fin=False, r=0.1), reseaux, 0.99, vrais_reseaux(1), 0.0, optimiseur)
    for t in chess.PIECE_TYPES:
        if t != chess.PAWN:
            assert poids_identiques(reseaux[t], avant[t]), chess.piece_name(t)


def test_les_reseaux_cibles_ne_changent_pas():
    reseaux, cibles = vrais_reseaux(), vrais_reseaux(1)
    avant = {t: copie_des_poids(r) for t, r in cibles.items()}
    optimiseur = torch.optim.Adam([p for r in reseaux.values() for p in r.parameters()], lr=1e-2)
    for _ in range(5):
        apprentissage(lot_fixe(fin=False, r=0.1), reseaux, 0.99, cibles, 0.0, optimiseur)
    for t in chess.PIECE_TYPES:
        assert poids_identiques(cibles[t], avant[t])


def test_l_erreur_baisse_sur_un_lot_fixe():
    reseaux = vrais_reseaux()
    optimiseur = torch.optim.Adam([p for r in reseaux.values() for p in r.parameters()], lr=1e-2)
    lot = lot_fixe(fin=True, r=1.0, c=1.0)
    premiere = sum(apprentissage(lot, reseaux, 0.99, vrais_reseaux(1), 0.0, optimiseur))
    for _ in range(100):
        derniere = sum(apprentissage(lot, reseaux, 0.99, vrais_reseaux(1), 0.0, optimiseur))
    assert derniere < premiere / 10


def test_fin_de_partie_la_cible_est_r_et_c():
    # Que des fins de partie avec r = 1 et c = 0,5 : Q_R → 1 et Q_C → 0,5
    reseaux = vrais_reseaux()
    optimiseur = torch.optim.Adam([p for r in reseaux.values() for p in r.parameters()], lr=1e-2)
    lot = lot_fixe(fin=True, r=1.0, c=0.5)
    for _ in range(300):
        apprentissage(lot, reseaux, 0.99, vrais_reseaux(1), 0.0, optimiseur)
    q_r, q_c = prediction(reseaux, lot)
    assert q_r == pytest.approx(1.0, abs=0.05)
    assert q_c == pytest.approx(0.5, abs=0.05)


# def test_gamma_s_applique_au_retour_mais_pas_au_danger():
#     # Réseaux cibles factices : Q_R = Q_C = 1 partout. Transitions non finales avec r = c = 0.
#     # Cibles attendues : y_R = 0 + γ × 1 = 0,5 et y_C = 0 + 1 = 1 (pas de γ pour le danger)
#     reseaux = vrais_reseaux()
#     cibles = reseaux_fixes() | {t: ReseauFixe(defaut=1.0) for t in chess.PIECE_TYPES}
#     optimiseur = torch.optim.Adam([p for r in reseaux.values() for p in r.parameters()], lr=1e-2)
#     lot = lot_fixe(fin=False)
#     for _ in range(300):
#         apprentissage(lot, reseaux, 0.5, cibles, 0.0, optimiseur)
#     q_r, q_c = prediction(reseaux, lot)
#     assert q_r == pytest.approx(0.5, abs=0.05)
#     assert q_c == pytest.approx(1.0, abs=0.05)


# =============================================================================
# Version par lots : mêmes résultats que le calcul transition par transition
# =============================================================================

class ReseauLineaire:
    """Faux réseau qui dépend vraiment de l'entrée (une application linéaire fixe), sans PyTorch."""
    def __init__(self, graine):
        gen = np.random.default_rng(graine)
        self.w_r = gen.normal(size=(NB_PLANS_ENTREE * 64, NB_SORTIES)).astype(np.float32)
        self.w_c = gen.normal(size=(NB_PLANS_ENTREE * 64, NB_SORTIES)).astype(np.float32)

    def __call__(self, x):
        plat = np.asarray(x).reshape(len(x), -1)
        return torch.from_numpy(plat @ self.w_r), torch.from_numpy(plat @ self.w_c)


def cible_de_reference(reseaux_cibles, lambda_, lot, t):
    """L'ancien algorithme, pour une transition : pièce par pièce, un appel au réseau par pièce."""
    from echecs_ma.env.encodage import ajouter_position_aux_plans
    meilleure, resultat = -np.inf, (0.0, 0.0)
    for k in range(int(lot.nb_pieces_suiv[t])):
        x = ajouter_position_aux_plans(lot.plans_suiv[t], int(lot.cases_suiv[t, k]), bool(lot.couleur[t]))
        q_r, q_c = reseaux_cibles[int(lot.types_suiv[t, k])](x[None])
        q_r, q_c = np.asarray(q_r)[0], np.asarray(q_c)[0]
        objectif = np.where(lot.masque_suiv[t, k], q_r - lambda_ * q_c, -np.inf)
        i = int(np.argmax(objectif))
        if objectif[i] > meilleure:
            meilleure, resultat = objectif[i], (float(q_r[i]), float(q_c[i]))
    return resultat


def lot_de_vraies_parties(graine=0, nb_parties=3, taille=64):
    """Un lot tiré d'une mémoire remplie par de vraies parties (agents et adversaire au hasard)."""
    from echecs_ma.entrainement.boucle import jouer_partie_complete
    from echecs_ma.env.adversaire import AdversaireAleatoire, EvaluateurMateriel
    from echecs_ma.env.environnement import Environnement
    gen = np.random.default_rng(graine)
    memoire = Memoire(5000, gen)
    arbitre = Arbitre(gen)
    for n in range(nb_parties):
        couleur = chess.WHITE if n % 2 == 0 else chess.BLACK
        env = Environnement(couleur, AdversaireAleatoire(gen), EvaluateurMateriel(), 0.99)
        jouer_partie_complete(env, memoire, arbitre, couleur, reseaux_fixes(), 0.0, 1.0)
    return memoire.tirer_lot(taille)


@pytest.mark.parametrize("lambda_", [0.0, 0.7])
@pytest.mark.parametrize("graine", [0, 1])
def test_cibles_par_lot_identiques_a_l_ancien_calcul(graine, lambda_):
    lot = lot_de_vraies_parties(graine)
    cibles = {t: ReseauLineaire(t) for t in chess.PIECE_TYPES}
    q_r, q_c = cibles_suivantes(cibles, lambda_, lot.plans_suiv, lot.types_suiv, lot.cases_suiv,
                                lot.masque_suiv, lot.couleur, lot.nb_pieces_suiv, lot.indicateur_fin)
    for t in range(len(lot.r)):
        if lot.indicateur_fin[t]:
            assert q_r[t] == 0 and q_c[t] == 0
        else:
            attendu_r, attendu_c = cible_de_reference(cibles, lambda_, lot, t)
            assert q_r[t] == pytest.approx(attendu_r, rel=1e-4, abs=1e-3)
            assert q_c[t] == pytest.approx(attendu_c, rel=1e-4, abs=1e-3)


def test_le_lot_contient_bien_des_parties_des_deux_couleurs():
    # Garde-fou du test précédent : il doit couvrir les Blancs et les Noirs
    lot = lot_de_vraies_parties(0)
    assert lot.couleur.any() and not lot.couleur.all()


def test_lot_entierement_fini():
    lot = lot_fixe(fin=True)
    q_r, q_c = cibles_suivantes(reseaux_fixes(), 0.0, lot.plans_suiv, lot.types_suiv, lot.cases_suiv,
                                lot.masque_suiv, lot.couleur, lot.nb_pieces_suiv, lot.indicateur_fin)
    assert (q_r == 0).all() and (q_c == 0).all()


def test_apprentissage_lot_melange_garde_chaque_cible_a_sa_place():
    # Moitié de coups de soldat (r = +1), moitié de coups de cavalier (r = −1), tous finaux.
    # Si prédictions et cibles étaient mal remises en ordre, les deux valeurs se mélangeraient.
    memoire = Memoire(32, np.random.default_rng(0))
    s = position_de(chess.Board(), chess.WHITE)
    soldat = ArbitreDecision(coup=chess.Move.from_uci("e2e4"), action=Action(chess.PAWN, chess.E2, 28))
    cavalier = ArbitreDecision(coup=chess.Move.from_uci("g1f3"), action=Action(chess.KNIGHT, chess.G1, 21))
    for _ in range(16):
        memoire.stocker_en_memoire(1.0, 0.0, s, soldat, None, chess.WHITE)
        memoire.stocker_en_memoire(-1.0, 0.0, s, cavalier, None, chess.WHITE)
    lot = memoire.tirer_lot(32)
    assert set(lot.type_action.tolist()) == {chess.PAWN, chess.KNIGHT}

    reseaux = vrais_reseaux()
    optimiseur = torch.optim.Adam([p for r in reseaux.values() for p in r.parameters()], lr=1e-2)
    for _ in range(300):
        apprentissage(lot, reseaux, 0.99, vrais_reseaux(1), 0.0, optimiseur)

    from echecs_ma.env.encodage import ajouter_position_aux_plans
    with torch.no_grad():
        x_soldat = torch.from_numpy(ajouter_position_aux_plans(s.plans, chess.E2, chess.WHITE)[None])
        x_cavalier = torch.from_numpy(ajouter_position_aux_plans(s.plans, chess.G1, chess.WHITE)[None])
        q_soldat = reseaux[chess.PAWN](x_soldat)[0][0, 28].item()
        q_cavalier = reseaux[chess.KNIGHT](x_cavalier)[0][0, 21].item()
    assert q_soldat == pytest.approx(1.0, abs=0.05)
    assert q_cavalier == pytest.approx(-1.0, abs=0.05)