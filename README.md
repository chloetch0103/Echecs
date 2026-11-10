# Échecs multi-agents
[![Tests](https://github.com/chloetch0103/Echecs/actions/workflows/test.yaml/badge.svg)](https://github.com/chloetch0103/Echecs/actions/workflows/test.yaml)

Système d'apprentissage par renforcement multi-agents pour les échecs : chaque pièce est un agent autonome, et toutes coopèrent autour d'une récompense commune. Les agents s'entraînent contre Stockfish, dont la force augmente progressivement.

> **Statut : premiers entraînements.** La chaîne complète est en place et testée : environnement de jeu, encodage, réseaux, arbitre, mémoire de rejeu, apprentissage DQN et orchestration de l'entraînement, avec journal et outil d'analyse. Les premiers essais se font contre un adversaire aléatoire, avant le passage à Stockfish.

## Approche

- **Agents** : une pièce = un agent, avec des paramètres partagés par type de pièce (6 réseaux, Roi compris).
- **Coordination** : chaque agent propose ses coups, un arbitre central choisit le meilleur.
- **Apprentissage** : DQN avec mémoire de rejeu et réseaux cibles ; chaque réseau estime deux valeurs par coup, le retour attendu (Q_R) et le danger attendu (Q_C).
- **Récompense** : résultat de la partie (+1 / 0 / −1), avec un façonnage par potentiel fondé sur l'évaluation de Stockfish (ou sur le bilan matériel).
- **Sécurité du Roi** :
  - relaxation lagrangienne qui limite les échecs subis (budget de danger d, prix $\lambda$ appris) ;
  - contrainte dure ("bouclier") qui écarte tout coup permettant un mat au coup suivant *(prévue)*.

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
  journal.csv       # une ligne toutes les `journal.frequence` parties
  checkpoints/      # réseaux et état de l'entraînement, sauvegardés régulièrement
```

Le journal suit le résultat moyen z, la répartition victoires / nulles / défaites, les échecs subis, $\lambda$, $\epsilon$, les erreurs d'apprentissage, le niveau de l'adversaire, la longueur des parties et la durée.

### Analyser un entraînement

```bash
python scripts/analyser.py                                  # le dernier entraînement de runs/
python scripts/analyser.py --dossier runs/essai_2026-11-09_19h30
```

Le script affiche un résumé chiffré et des alertes (divergence des erreurs, $\lambda$ qui grimpe pendant l'exploration, budget de danger dépassé…), et produit dans `<dossier>/analyse/` un tableau de bord (`tableau_de_bord.png`) et le résumé (`resume.md`). L'option `--estimer N` estime la durée d'un entraînement de N parties à partir de la vitesse observée.

## Tests

```bash
python -m pytest tests/
```

Les tests n'ont pas besoin de Stockfish : l'adversaire, l'évaluation et parfois les réseaux sont remplacés par des versions scriptées, ce qui rend chaque résultat prévisible (mat, pat, promotion, façonnage, échec subi, choix de l'arbitre, calcul des cibles).

## Contenu du dépôt

```
configs/          # fichiers de configuration (base.yaml, essai.yaml)
docs/             # spécification
scripts/          # entrainer.py, analyser.py
src/echecs_ma/
  agents/         # réseaux de neurones, arbitre
  entrainement/   # boucle de jeu, mémoire de rejeu, apprentissage, réglages (ε, λ, curriculum)
  env/            # environnement de jeu, adversaires et évaluateurs, valeurs des pièces, encodage
  utils/          # analyse des journaux
tests/            # tests unitaires
runs/             # entraînements (non versionné)
pyproject.toml    # métadonnées et dépendances du projet
```

## Avancement

- [x] Environnement : coups légaux par agent, récompense façonnée, coût de danger, fins de partie.
- [x] Valeurs des pièces et potentiel matériel de repli.
- [x] Encodage des données pour les réseaux de neurones.
- [x] Agents et arbitre.
- [x] Mémoire de rejeu.
- [x] Boucle de jeu et étape d'apprentissage DQN.
- [x] Orchestration de l'entraînement : exploration ε, relaxation lagrangienne, curriculum, journal, sauvegardes.
- [x] Analyse des entraînements.
- [ ] Calibrage de la relaxation lagrangienne (η, d, mise à jour de λ pendant l'exploration).
- [ ] Entraînement contre Stockfish.
- [ ] Bouclier anti-mat.

## Limites et pistes

- **$\lambda$ pendant l'exploration** : tant que $\epsilon$ est élevé, les coups sont surtout aléatoires et les échecs subis restent au-dessus du budget, quelle que soit la valeur de $\lambda$. Mis à jour à chaque partie, $\lambda$ grimpe alors sans effet sur le comportement et risque d'écraser le retour au moment où l'exploitation commence. Pistes : ne mettre $\lambda$ à jour qu'une fois $\epsilon$ assez bas, réduire η, recalibrer d.

## Références

- A. Y. Ng, D. Harada, S. Russell (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML.
- E. Altman (1999). *Constrained Markov Decision Processes.* Chapman & Hall/CRC.
- V. Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- [python-chess](https://python-chess.readthedocs.io/) · [Stockfish](https://stockfishchess.org/)