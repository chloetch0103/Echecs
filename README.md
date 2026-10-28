# Échecs multi-agents

Système d'apprentissage par renforcement multi-agents pour les échecs : chaque pièce est un agent autonome, et toutes coopèrent autour d'une récompense commune. Les agents s'entraîneront contre Stockfish, dont la force augmentera progressivement.

> **Statut : en phase de conception.** Le dépôt contient pour l'instant la spécification et les fichiers de configuration. L'implémentation est la prochaine étape.

## Approche

- **Agents** : une pièce = un agent, avec des paramètres partagés par type de pièce (6 réseaux, Roi compris).
- **Coordination** : chaque agent propose ses coups, un arbitre central choisit le meilleur.
- **Récompense** : résultat de la partie (+1 / 0 / −1), avec un façonnage par potentiel fondé sur l'évaluation de Stockfish.
- **Sécurité du Roi** :
  - contrainte dure ("bouclier") qui écarte tout coup permettant un mat au coup suivant ;
  - relaxation lagrangienne qui limite les échecs subis (budget de danger d, prix $\lambda$ appris).

Le détail (formulation, justification du façonnage, contraintes, points ouverts) est dans [`docs/specification.md`](docs/specification.md).

## Contenu du dépôt

```
configs/   # fichiers de configuration
docs/      # spécification
pyproject.toml  # Métadonnées et dépendances du projet (pas encore complet)
```

## Prochaines étapes

1. Environnement (python-chess, interface Stockfish, potentiel $\Phi$).
2. Agents, arbitre et relaxation lagrangienne.
3. Bouclier anti-mat.
4. Boucle d'entraînement et curriculum.

## Références

- A. Y. Ng, D. Harada, S. Russell (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML.
- E. Altman (1999). *Constrained Markov Decision Processes.* Chapman & Hall/CRC.
- [python-chess](https://python-chess.readthedocs.io/) · [Stockfish](https://stockfishchess.org/)
