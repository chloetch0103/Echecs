"""Tests des réseaux des agents : forme des sorties, deux têtes indépendantes,
compatibilité avec l'encodage. Aucun apprentissage ici."""
import chess
import pytest
import torch

from echecs_ma.agents.reseaux import Reseau
from echecs_ma.env.encodage import (
    NB_PLANS_ENTREE,
    NB_SORTIES,
    ajouter_position_aux_plans,
    encoder_echiquier,
)

TAILLE_ENTREE = NB_PLANS_ENTREE * 8 * 8      # 1 280


@pytest.fixture(autouse=True)
def graine_fixe():
    # Les poids sont tirés au hasard à la création : on fixe la graine
    # pour que chaque test donne toujours le même résultat.
    torch.manual_seed(0)


def entree_aleatoire(taille_lot):
    return torch.rand(taille_lot, NB_PLANS_ENTREE, 8, 8)


# --- Forme des sorties -------------------------------------------------------

@pytest.mark.parametrize("taille_lot", [1, 5, 64])
def test_forme_des_sorties(taille_lot):
    reseau = Reseau([16])
    q_r, q_c = reseau(entree_aleatoire(taille_lot))
    assert q_r.shape == (taille_lot, NB_SORTIES)
    assert q_c.shape == (taille_lot, NB_SORTIES)
    assert q_r.dtype == q_c.dtype == torch.float32


@pytest.mark.parametrize("couches_cachees", [[], [16], [32, 16], [256, 256]])
def test_differentes_couches_cachees(couches_cachees):
    # Sans couche cachée, les têtes sont directement reliées à l'entrée aplatie
    q_r, q_c = Reseau(couches_cachees)(entree_aleatoire(3))
    assert q_r.shape == q_c.shape == (3, NB_SORTIES)


# --- Structure : un tronc partagé, deux têtes --------------------------------

def test_nombre_de_parametres_sans_couche_cachee():
    # Deux têtes de 1 280 → 88, chacune avec ses poids et ses biais
    reseau = Reseau([])
    attendu = 2 * (TAILLE_ENTREE * NB_SORTIES + NB_SORTIES)
    assert sum(p.numel() for p in reseau.parameters()) == attendu


def test_nombre_de_parametres_avec_tronc():
    # Le tronc 1 280 → 16 n'est compté qu'une fois : il est partagé par les deux têtes
    reseau = Reseau([16])
    tronc = TAILLE_ENTREE * 16 + 16
    tetes = 2 * (16 * NB_SORTIES + NB_SORTIES)
    assert sum(p.numel() for p in reseau.parameters()) == tronc + tetes


def test_les_deux_tetes_sont_independantes():
    with torch.no_grad():
        q_r, q_c = Reseau([16])(entree_aleatoire(4))
    assert not torch.equal(q_r, q_c)


def test_corriger_q_r_ne_touche_pas_la_tete_q_c():
    # Une erreur calculée seulement sur Q_R doit remonter dans le tronc et la tête Q_R,
    # mais pas dans la tête Q_C : c'est ce qui permet d'apprendre les deux séparément.
    reseau = Reseau([16])
    q_r, _ = reseau(entree_aleatoire(4))
    q_r.sum().backward()
    assert reseau.tete_Q_R.weight.grad is not None
    assert reseau.tete_Q_C.weight.grad is None
    assert all(p.grad is not None for p in reseau.reseau_partage.parameters())


def test_les_deux_tetes_entrainent_le_tronc():
    reseau = Reseau([16])
    _, q_c = reseau(entree_aleatoire(4))
    q_c.sum().backward()
    assert all(p.grad is not None for p in reseau.reseau_partage.parameters())


# --- Compatibilité avec l'encodage -------------------------------------------

def vraie_entree(echiquier, case, couleur=chess.WHITE):
    plans = encoder_echiquier(echiquier, couleur)
    x = ajouter_position_aux_plans(plans, case, couleur)
    return torch.from_numpy(x).unsqueeze(0)          # ajoute la dimension du lot


def test_accepte_la_sortie_de_l_encodage():
    reseau = Reseau([16])
    with torch.no_grad():
        q_r, q_c = reseau(vraie_entree(chess.Board(), chess.E2))
    assert q_r.shape == q_c.shape == (1, NB_SORTIES)


def test_deux_soldats_donnent_des_sorties_differentes():
    # Même position, même réseau : seul le plan de la pièce change,
    # et le réseau doit en tenir compte.
    reseau = Reseau([16])
    echiquier = chess.Board()
    with torch.no_grad():
        q_a2, _ = reseau(vraie_entree(echiquier, chess.A2))
        q_e2, _ = reseau(vraie_entree(echiquier, chess.E2))
    assert not torch.equal(q_a2, q_e2)


def test_lot_de_plusieurs_pieces():
    # Comme le fera l'arbitre : toutes les pièces d'un même type en un seul passage
    reseau = Reseau([16])
    echiquier = chess.Board()
    soldats = list(echiquier.pieces(chess.PAWN, chess.WHITE))
    lot = torch.cat([vraie_entree(echiquier, case) for case in soldats])
    with torch.no_grad():
        q_r, _ = reseau(lot)
    assert q_r.shape == (8, NB_SORTIES)


def test_meme_entree_meme_sortie():
    reseau = Reseau([16])
    x = entree_aleatoire(2)
    with torch.no_grad():
        a, _ = reseau(x)
        b, _ = reseau(x)
    assert torch.equal(a, b)