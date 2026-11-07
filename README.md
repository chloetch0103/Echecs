# Échecs multi-agents
[![Tests](https://github.com/chloetch0103/Echecs/actions/workflows/test.yaml/badge.svg)](https://github.com/chloetch0103/Echecs/actions/workflows/test.yaml)

Système d'apprentissage par renforcement multi-agents pour les échecs : chaque pièce est un agent autonome, et toutes coopèrent autour d'une récompense commune. Les agents s'entraîneront contre Stockfish, dont la force augmentera progressivement.

> **Statut : implémentation en cours.** L'environnement de jeu (récompense, coût de danger, fins de partie), les éléments de calcul du bilan matériel sont en place et testés. Le dispositif d'encodage avant l'entrée dans les réseaux a également déjà été mis en place. Les agents et la boucle d'entraînement sont les prochaines étapes.

## Approche

- **Agents** : une pièce = un agent, avec des paramètres partagés par type de pièce (6 réseaux, Roi compris).
- **Coordination** : chaque agent propose ses coups, un arbitre central choisit le meilleur.
- **Récompense** : résultat de la partie (+1 / 0 / −1), avec un façonnage par potentiel fondé sur l'évaluation de Stockfish.
- **Sécurité du Roi** :
  - relaxation lagrangienne qui limite les échecs subis (budget de danger d, prix $\lambda$ appris) ;
  - contrainte dure ("bouclier") qui écarte tout coup permettant un mat au coup suivant *(prévue)*.

Le détail (formulation, justification du façonnage, contraintes, points ouverts) est dans [`docs/specifications.md`](docs/specifications.md).

## Installation

Prérequis : Python 3.14 et [Stockfish](https://stockfishchess.org/download/).

```bash
git clone https://github.com/chloetch0103/Echecs.git
cd Echecs
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS : source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Indiquer ensuite le chemin de l'exécutable Stockfish, soit dans `configs/base.yaml` (`stockfish.chemin`), soit dans la variable d'environnement `STOCKFISH_PATH` :

```bash
setx STOCKFISH_PATH "C:\chemin\vers\stockfish.exe"      # Windows
export STOCKFISH_PATH=/chemin/vers/stockfish            # Linux / macOS
```

## Tests

```bash
python -m pytest tests/
```

Les tests de l'environnement n'ont pas besoin de Stockfish : l'adversaire et l'évaluation sont remplacés par des versions scriptées, ce qui rend chaque résultat prévisible (mat, pat, promotion, façonnage, échec subi).

## Contenu du dépôt

```
configs/          # fichiers de configuration
docs/             # spécification
src/echecs_ma/
  agents/         # réseaux de neurones, arbitre
  env/            # environnement de jeu, adversaire et évaluateur Stockfish, valeurs des pièces et fonction de calcul du bilan matériel, encodage
tests/            # tests unitaires
pyproject.toml    # métadonnées et dépendances du projet
```

## Avancement

- [x] Environnement : coups légaux par agent, récompense façonnée, coût de danger, fins de partie.
- [x] Valeurs des pièces et potentiel matériel de repli.
- [x] Encodage des données pour les réseaux de neurones.
- [x] Agents, arbitre 
- [ ] relaxation lagrangienne.
- [ ] Bouclier anti-mat.
- [ ] Boucle d'entraînement et curriculum.

## Références

- A. Y. Ng, D. Harada, S. Russell (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML.
- E. Altman (1999). *Constrained Markov Decision Processes.* Chapman & Hall/CRC.
- [python-chess](https://python-chess.readthedocs.io/) · [Stockfish](https://stockfishchess.org/)