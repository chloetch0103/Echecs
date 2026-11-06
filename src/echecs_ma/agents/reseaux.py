"""
Création des réseaux de neurones :
    Un réseau par type de pièces
        Entrée : les plans définis dans le fichier encodage 20x8x8 = 1280
        Sortie : deux têtes de 88 valeurs (une pour Q_R et l'autre pour Q_C)
        Couches cachées : définies dans base.yaml dans les configs

"""
from torch import nn

from echecs_ma.env.encodage import NB_PLANS_ENTREE, NB_SORTIES


class Reseau(nn.Module):
    def __init__(self, couches_cachees):
        super().__init__()

        # 1. Création du réseau partagé
        couches = [nn.Flatten()]
        couche_precedente = NB_PLANS_ENTREE*8*8
        for couche in couches_cachees:
            couches.append(nn.Linear(couche_precedente, couche))
            couches.append(nn.ReLU())
            couche_precedente = couche
        self.reseau_partage = nn.Sequential(*couches)

        # 2. Création de la tête pour Q_R
        self.tete_Q_R = nn.Linear(couche_precedente, NB_SORTIES)

        # 3. Création de la tête pour Q_C
        self.tete_Q_C = nn.Linear(couche_precedente, NB_SORTIES)

    def forward(self, x):
        caracteristiques_partagees = self.reseau_partage(x)
        return self.tete_Q_R(caracteristiques_partagees), self.tete_Q_C(caracteristiques_partagees)


