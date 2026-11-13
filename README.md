# Échecs multi-agents
[![Tests](https://github.com/chloetch0103/Echecs/actions/workflows/test.yaml/badge.svg)](https://github.com/chloetch0103/Echecs/actions/workflows/test.yaml)

Système d'apprentissage par renforcement multi-agents pour les échecs : chaque pièce est un agent autonome, et toutes coopèrent autour d'une récompense commune. Les agents s'entraînent contre Stockfish, dont la force augmente progressivement.

> **Statut : premiers entraînements.** La chaîne complète est en place et testée : environnement de jeu, encodage, réseaux, arbitre, mémoire de rejeu, apprentissage DQN et orchestration de l'entraînement, avec journal, analyse statistique et comparaison d'entraînements. Les premiers essais se font contre un adversaire aléatoire, avant le passage à Stockfish.

## Approche

- **Agents** : une pièce = un agent, avec des paramètres partagés par type de pièce (6 réseaux, Roi compris).
- **Coordination** : chaque agent propose ses coups, un arbitre central choisit le meilleur.
- **Apprentissage** : DQN avec mémoire de rejeu et réseaux cibles ; chaque réseau estime deux valeurs par coup, le retour attendu (Q_R) et le danger attendu (Q_C).
- **Récompense** : résultat de la partie (+1 / 0 / −1), avec un façonnage par potentiel fondé sur l'évaluation de Stockfish (ou sur le bilan matériel).
- **Sécurité du Roi** :
  - relaxation lagrangienne qui limite les échecs subis (budget de danger d, prix $\lambda$ appris) ;
  - contrainte dure ("bouclier") qui écarte tout coup permettant un mat au coup suivant *(prévue)*.
- **Encodage** : la position est vue du côté de l'agent (miroir vertical pour les Noirs), en 21 plans 8×8 : pièces, prise en passant, droits de roque, compteur de demi-coups et trois plans de répétition (position déjà vue 2, 3 et 4 fois). Un 22ᵉ plan désigne la pièce qui joue.
- **Fins de partie** : en plus du mat, du pat et du matériel insuffisant, seules les nulles automatiques arrêtent la partie (quintuple répétition, règle des 75 coups). Aucune nulle n'est réclamée à la place des joueurs.

Le détail (formulation, justification du façonnage, contraintes, points ouverts) est dans [`docs/specifications.md`](docs/specifications.md).

## Installation

Prérequis : Python 3.14 et, pour jouer contre Stockfish ou l'utiliser comme évaluateur, [Stockfish](https://stockfishchess.org/download/).

```bash
git clone https://github.com/chloetch0103/Echecs.git
cd Echecs
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS : source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Indiquer ensuite le chemin de l'exécutable Stockfish, soit dans la configuration (`stockfish.chemin`), soit dans la variable d'environnement `STOCKFISH_PATH` :

```bash
setx STOCKFISH_PATH "C:\chemin\vers\stockfish.exe"      # Windows
export STOCKFISH_PATH=/chemin/vers/stockfish            # Linux / macOS
```

## Utilisation

### Entraîner

```bash
python scripts/entrainer.py --config configs/base.yaml
```

La configuration choisit l'adversaire et l'évaluateur. Pour un essai rapide sans Stockfish (adversaire aléatoire, potentiel matériel, petits réseaux) :

```bash
python scripts/entrainer.py --config configs/essai.yaml
```

Chaque entraînement crée son propre dossier dans `runs/` :

```
runs/essai_2026-11-09_19h30/
  config.yaml       # copie de la configuration utilisée
  journal.csv       # une ligne toutes les `journal.frequence` parties (moyennes du bloc)
  parties.csv       # une ligne par partie, sans agrégation
  checkpoints/      # réseaux et état de l'entraînement, sauvegardés régulièrement
  analyse/          # produit par les scripts d'analyse et de comparaison
```

Le journal suit le résultat moyen z, la répartition victoires / nulles / défaites, le bilan matériel en fin de partie, les raisons de fin de partie, les échecs subis, $\lambda$, $\epsilon$, les erreurs d'apprentissage, le niveau de l'adversaire, la longueur des parties et la durée. Le fichier `parties.csv` garde, pour chaque partie, la couleur jouée, z, la raison de fin, le nombre de coups, le bilan matériel, les échecs subis, $\epsilon$ et le niveau de l'adversaire.

### Analyser un entraînement

```bash
python scripts/analyser.py                                  # le dernier entraînement de runs/
python scripts/analyser.py --dossier runs/essai_2026-11-09_19h30
```

Le script produit dans `<dossier>/analyse/` :

- un **tableau de bord** (`tableau_de_bord.png`) : courbes brutes et lissées de chaque indicateur, et répartition des raisons de fin de partie ;
- un **résumé** (`resume.md`) : déroulement de l'entraînement, statistiques de la **phase d'exploitation** (parties jouées une fois $\epsilon$ au plancher, avec un intervalle de confiance à 95 % par bootstrap pour chaque indicateur), évolution entre la première et la seconde moitié de cette phase, et alertes (divergence des erreurs, $\lambda$ qui grimpe pendant l'exploration, budget de danger dépassé, parties qui tournent en rond…).

Options : `--lissage N` fixe la fenêtre de la moyenne glissante, `--estimer N` estime la durée d'un entraînement de N parties à partir de la vitesse observée.

### Comparer deux entraînements

```bash
python scripts/comparer.py runs/essai_avec_apprentissage runs/essai_temoin
```

Le premier dossier est l'essai, le second le témoin. Pour chaque indicateur de la phase d'exploitation (bilan matériel, victoires, nulles, défaites, z, échecs subis, longueur des parties), le script donne la différence essai − témoin, son intervalle de confiance à 95 % par bootstrap, et indique si elle est significative (intervalle qui exclut 0). Le tableau est écrit dans `<essai>/analyse/comparaison_<témoin>.md`. L'option `--graine` fixe la graine du bootstrap.

Pour que la comparaison isole un seul effet, les deux entraînements doivent avoir la même configuration (graine, nombre de parties, décroissance de $\epsilon$, début de l'apprentissage), à l'exception du réglage étudié.

## Tests

```bash
python -m pytest tests/
```

Les tests n'ont pas besoin de Stockfish : l'adversaire, l'évaluation et parfois les réseaux sont remplacés par des versions scriptées, ce qui rend chaque résultat prévisible (mat, pat, promotion, répétitions, façonnage, échec subi, choix de l'arbitre, calcul des cibles, intervalles de confiance).

## Performances

Le profilage (`cProfile`) d'un essai de 20 parties contre un adversaire aléatoire a montré que 90 % du temps partait dans l'étape d'apprentissage : chaque correction faisait environ 500 appels aux réseaux, un par transition pour les prédictions et un par pièce de la position suivante pour les cibles.

Regrouper ces appels par type de pièce (un passage par réseau pour tout le lot) et construire les entrées sans boucle Python (indexation avancée de NumPy) a donné, sur le même essai :

| | Avant         | Après    | Gain |
|---|---------------|----------|---|
| Durée totale | 445 s         | 57 s     | × 7,8 |
| Étape d'apprentissage | 403 s         | 25,6 s   | × 16 |
| Calcul des cibles | 302 s         | 8,5 s    | × 35 |
| Appels aux réseaux par correction | $\approx$ 500 | $\le$ 12 | |

Un test vérifie que la version par lots donne exactement les mêmes cibles que l'ancien calcul, transition par transition.

## Résultats

Tous les essais sont faits contre un adversaire aléatoire, sur 300 parties dont 100 d'exploration ($\epsilon$ décroissant de 1 à 0,05). Les statistiques portent sur les 200 parties suivantes (phase d'exploitation). Chaque comparaison oppose deux entraînements de même configuration qui ne diffèrent que par le réglage étudié. Les intervalles sont à 95 %, par bootstrap.

**Apprentissage contre agent non entraîné** (témoin : mêmes réseaux, jamais corrigés) :

- Sur 3 graines, l'entraînement améliore le résultat moyen z de $+0,09$ à $+0,11$ (intervalle excluant 0 dans les trois cas), surtout en réduisant les défaites : l'agent a appris à moins perdre, pas encore à gagner (3,5 à 4 % de victoires).
- L'effet sur le bilan matériel en fin de partie est significatif sur 2 graines sur 3 : $+2,2$ et $+2,3$ points. Sur la troisième, le témoin non entraîné a joué nettement mieux que les deux autres ($-1,25$ point contre $-3,7$ et $-4,0$), ce qui illustre la variabilité d'une graine à l'autre.

**Relaxation lagrangienne contre agent sans contrainte** (budget d = 13 échecs par partie, une graine) :

- Le lagrangien réduit les échecs subis de $1,7$ par partie, soit environ 11 % (IC : $[-3,1 ; -0,2]$).
- Les autres indicateurs vont dans le sens d'un coût sur le jeu (z : $-0,06$ $[-0,135 ; +0,015]$ ; bilan matériel : $-1,25$ point $[-2,7 ; +0,3]$), sans que ce coût soit significatif.

## Contenu du dépôt

```
configs/          # fichiers de configuration (base.yaml, essai.yaml)
docs/             # spécification
scripts/          # entrainer.py, analyser.py, comparer.py
src/echecs_ma/
  agents/         # réseaux de neurones, arbitre
  entrainement/   # boucle de jeu, mémoire de rejeu, apprentissage, réglages (ε, λ, curriculum)
  env/            # environnement de jeu, adversaires et évaluateurs, valeurs des pièces, encodage
  utils/          # analyse des journaux, statistiques et comparaison d'entraînements
tests/            # tests unitaires
runs/             # entraînements (non versionné)
pyproject.toml    # métadonnées et dépendances du projet
```

## Avancement

- [x] Environnement : coups légaux par agent, récompense façonnée, coût de danger, fins de partie.
- [x] Valeurs des pièces et potentiel matériel de repli.
- [x] Encodage des données pour les réseaux de neurones, dont les répétitions de position.
- [x] Agents et arbitre.
- [x] Mémoire de rejeu.
- [x] Boucle de jeu et étape d'apprentissage DQN.
- [x] Orchestration de l'entraînement : exploration $\epsilon$, relaxation lagrangienne, curriculum, journal, sauvegardes.
- [x] Analyse des entraînements : tableau de bord, intervalles de confiance, comparaison de deux entraînements.
- [ ] Calibrage de la relaxation lagrangienne ($\eta$, $d$, mise à jour de $\lambda$ pendant l'exploration).
- [ ] Entraînements plus longs et sur plusieurs graines.
- [ ] Entraînement contre Stockfish.
- [ ] Bouclier anti-mat.

## Limites et pistes

- **Durée des entraînements** : 300 parties, dont 200 en exploitation, laissent des intervalles de confiance larges et mesurent surtout un régime transitoire. Des entraînements plus longs sont nécessaires pour voir si l'agent apprend à gagner, et pas seulement à moins perdre.
- **Variabilité d'une graine à l'autre** : les intervalles de confiance mesurent le hasard d'une partie à l'autre, pas celui d'un entraînement à l'autre. Un agent non entraîné, en particulier, joue plus ou moins bien selon ses poids initiaux. Une conclusion générale demande plusieurs graines.
- **$\lambda$ et la qualité de Q_C** : l'arbitre maximise Q_R − $\lambda$·Q_C. Quand $\lambda$ grandit, les choix dépendent de plus en plus de Q_C ; si son estimation est encore bruitée, l'agent joue moins bien sans beaucoup réduire le danger. Avec un $\eta$ petit (0,001), $\lambda$ n'a pas convergé en 300 parties.
- **$\lambda$ pendant l'exploration** : tant que $\epsilon$ est élevé, les coups sont surtout aléatoires et les échecs subis restent au-dessus du budget, quelle que soit la valeur de $\lambda$. Mis à jour à chaque partie, $\lambda$ grimpe alors sans effet sur le comportement et risque d'écraser le retour au moment où l'exploitation commence. Pistes : ne mettre $\lambda$ à jour qu'une fois $\epsilon$ assez bas, réduire η, recalibrer d.
- **Réclamation automatique des nulles** : quand l'environnement réclamait la triple répétition, la majorité des parties finissaient par répétition, y compris quand l'agent menait au matériel. Seules les nulles automatiques sont désormais appliquées.

## Références

- A. Y. Ng, D. Harada, S. Russell (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML.
- E. Altman (1999). *Constrained Markov Decision Processes.* Chapman & Hall/CRC.
- V. Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- [python-chess](https://python-chess.readthedocs.io/) · [Stockfish](https://stockfishchess.org/)