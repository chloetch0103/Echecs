# Échecs multi-agents : spécification

Oct 18, 2026 - @Chloé

## 1. Environnement

Chaque pièce est un agent. À chaque tour, un seul agent joue.

- **État** $s_t$ : la position complète (plateau, camp au trait, droits de roque, prise en passant).
- **Agents** : $I_t$ est l'ensemble des pièces encore en vie au temps t. Il rétrécit au fil des captures, et une promotion change le type d'un agent : il prend alors le réseau et la valeur de sa nouvelle pièce.
- **Actions** : l'agent i dispose de ses coups légaux $A_i(s_t)$, qui peuvent être vides (pièce bloquée).

L'adversaire est **Stockfish**, bridé au départ puis renforcé progressivement (curriculum). Il sert aussi d'évaluateur de position pour la récompense (section 3).

## 2. Valeurs des pièces

Ces valeurs n'entrent pas directement dans la récompense (section 3). Elles restent utiles comme potentiel de repli et pour suivre le matériel pendant l'entraînement. Le Roi n'a pas de valeur.

| Pièce | Valeur |
| --- | --- |
| Reine | 9 |
| Tour | 5 |
| Cavalier | 3,25 |
| Fou | 3 |
| Soldat | 1 |

## 3. Récompense : résultat de la partie + façonnage par potentiel

L'objectif est de gagner la partie ; l'évaluation de la position ne sert qu'à guider l'apprentissage. Un joueur avec moins de pièces peut être en avantage s'il est proche du mat, et la récompense doit en tenir compte.

$$
r_t = z \cdot \mathbb{1}[\text{fin}] + \gamma\,\Phi(s_{t+1}) - \Phi(s_t)
$$

- $z$ : le résultat de la partie, +1 (victoire), 0 (nulle) ou −1 (défaite).
- $\mathbb{1}[\text{fin}]$ : vaut 1 si la partie se termine à ce pas, 0 sinon. Le résultat n'est donc compté qu'une fois, au dernier coup.
- $\Phi(s_t)$ et $\Phi(s_{t+1})$ : le potentiel de la position avant le coup de l'algorithme, puis après son coup et la réponse adverse.
- **$\gamma$** :  facteur d'actualisation. Varie entre 0 et 1 et détermine l'importance qu'accorde l'agent aux récompenses futures plutôt qu'aux récompenses immédiates.
- **Règle de fin** : $\Phi(s_{t+1}) = 0$ quand $s_{t+1}$ est une position terminale. Au dernier pas, $r_t = z − \Phi(s_t)$, ce qui évite de compter la victoire deux fois.

**Pourquoi ce bonus ne fausse pas l'objectif** (façonnage par potentiel, Ng et al., 1999) : sommés sur la partie, les bonus se compensent presque tous. Il ne reste que $-\Phi(s_0)$ qui ne dépend pas des choix des agents. La meilleure politique est donc la même qu'avec $z$ seul.

### Objectif

$$
\max_\pi \; J(\pi) = \mathbb{E}_{\tau \sim \pi}\Big[\sum_t \gamma^t r_t\Big] \quad \text{sous les contraintes de la section 4}
$$

- $\pi$ est la politique : la règle qui dit quel coup jouer dans chaque position. $\tau ∼ \pi$ désigne une partie entière (états, coups, récompenses) jouée en suivant $\pi$. $J(\pi)$ est le résultat moyen obtenu avec la stratégie $\pi$.
- $\gamma$ (entre 0 et 1, on va le fixer à 0,99) est le facteur d'actualisation : une récompense obtenue dans t coups compte $\gamma^t$ fois sa valeur. Il pousse à conclure plus tôt et garantit une somme finie. Proche de 1 : agent patient, capable de sacrifices à long terme ; proche de 0 : agent glouton.

### Le potentiel : l'évaluation de Stockfish

Stockfish renvoie soit une évaluation en centipions (100 cp équivalent environ à un soldat d'avance, en tenant compte aussi de la sécurité du Roi, de l'activité des pièces, des menaces…), soit un **mat en N** : N coups du camp qui mate, mat forcé trouvé dans la profondeur de recherche. On convertit ce score avec le modèle victoire/nulle/défaite de Stockfish :

$$
\Phi(s) = 2\,\mathbb{E}_{\text{VND}}(s) - 1 \in [-1, 1]
$$

$E_{VND}(s)$ est le score attendu entre 0 et 1, obtenu avec `score.pov(couleur).wdl().expectation()` dans python-chess. Un mat pour l'algorithme donne $\Phi = +1$, un mat contre $\Phi = -1$ : $\Phi$ est sur la même échelle que $z$.

- **Point de vue** : toujours `score.pov(couleur des agents)`.
- **Profondeur ou temps d'analyse** : compromis entre précision du potentiel et coût de calcul.
- **Repli** : le bilan matériel (section 2), plus simple mais aveugle aux menaces de mat.

## 4. Sécurité du Roi : contrainte dure + relaxation lagrangienne

La sécurité du Roi combine deux mécanismes complémentaires : une contrainte dure contre le **mat** (non négociable) et une contrainte souple contre les **échecs** (risque acceptable avec modération).

### 4.1 Contrainte dure : le bouclier anti-mat

**Légalité (déjà assurée) :** Les agents ne choisissent que parmi leurs coups légaux $A_i(s)$, générés par python-chess. Un coup qui laisse le Roi en échec est donc impossible : c'est un masque d'actions.

**Bouclier anti-mat.** Avant l'arbitre, un filtre vérifie chaque coup candidat. On note s · a la position obtenue en jouant a depuis s :

$$
\text{sûr}(s, a) \iff \forall b \in \mathcal{A}_{\text{adv}}(s \cdot a),\; \neg\,\text{mat}(s \cdot a \cdot b)
$$

Autrement dit : après mon coup a, aucune réponse b de l'adversaire ne me met mat. L'ensemble des coups sûrs est :

$$
\mathcal{S}(s) = \{ (i, a) \;:\; i \in \mathcal{I}_t,\; a \in \mathcal{A}_i(s),\; \text{sûr}(s, a) \}
$$

- **Satisfiabilité** : si S(s) est vide, le mat est forcé et la partie est perdue ; n'importe quel coup légal est alors joué.
- **Pourquoi le mat et pas l'échec** : la contrainte "ne pas subir d'échec au coup suivant" serait très souvent insatisfiable, car l'adversaire peut presque toujours donner un échec.
- **Extension au mat en 2** : après chaque réponse adverse, il doit rester un coup sûr.

$$
\text{sûr}_2(s, a) \iff \text{sûr}(s, a) \;\wedge\; \forall b \in \mathcal{A}_{\text{adv}}(s \cdot a),\; \exists a' \in \mathcal{A}(s \cdot a \cdot b),\; \text{sûr}(s \cdot a \cdot b, a')
$$

- **Implémentation** : énumération des réponses avec python-chess, ou encodage de la formule avec Z3 pour pratiquer un vrai solveur SAT/SMT.


### 4.2 Contrainte souple : relaxation lagrangienne sur les échecs

Les échecs subis sont limités par un budget de danger, avec un prix $\lambda$ appris pendant l'entraînement.

- **Coût** : $c_t = 1$ si le coup suivant de l'adversaire met le Roi en échec, 0 sinon. Extension possible : compter les cases autour du Roi attaquées par l'adversaire.
- **Seuil** : $d = 5$, le danger total toléré en moyenne par partie (un peu plus de 5 échecs, à cause de $\gamma$).

Le problème complet :

$$
\max_\pi \; J_R(\pi) = \mathbb{E}_{\tau \sim \pi}\Big[\sum_t \gamma^t r_t\Big] \quad \text{s.c.} \quad J_C(\pi) = \mathbb{E}_{\tau \sim \pi}\Big[\sum_t \gamma^t c_t\Big] \le d \quad \text{et coups dans } \mathcal{S}(s)
$$

Relaxation lagrangienne :

$$
\mathcal{L}(\pi, \lambda) = J_R(\pi) - \lambda\,\big(J_C(\pi) - d\big), \qquad \lambda \ge 0
$$

L'entraînement alterne deux mises à jour :

1. Améliorer la politique avec la récompense pénalisée $r_t − \lambda c_t$
2. Ajuster le multiplicateur, $\hat{J}_C$ étant le danger mesuré sur les dernières parties et $\eta$ un petit pas (par exemple 0,01) :

$$
\lambda \leftarrow \max\big(0,\; \lambda + \eta\,(\hat J_C - d)\big)
$$

- **$\lambda$ est le prix du danger** : combien de récompense vaut une unité de danger. Budget dépassé $(\hat{J}_C > d) \longrightarrow \lambda$ monte, les agents deviennent prudents ; marge restante $(\hat{J}_C < d) \longrightarrow \lambda$ baisse, ils osent davantage. Le max(0, ·) empêche de récompenser le danger.
- **À surveiller** : $\lambda$ peut osciller ou exploser si η est mal réglé. On va donc l'enregistrer dans les journaux dès le départ

### 4.3 Ordre d'implémentation

1. **Relaxation lagrangienne d'abord** : S(s) est alors simplement l'ensemble des coups légaux.
2. **Bouclier ensuite** (`contrainte/bouclier.py`, à côté de `contrainte/lagrangien.py`) : il restreint S(s) aux coups sûrs, sans rien changer au reste.

## 5. Coordination : proposition + arbitre

Chaque agent propose ses coups, un arbitre retient le meilleur.

1. Chaque agent évalue ses coups avec deux estimateurs : $Q^R_i(s, a)$ pour le retour attendu (section 3) et $Q^C_i(s, a)$ pour le danger attendu (section 4.2).
2. Le bouclier écarte les coups non sûrs (section 4.1).
3. L'arbitre choisit le meilleur coup sûr, en tenant compte du prix du danger :

$$
(i^*, a^*) = \arg\max_{(i, a) \in \mathcal{S}(s)} \Big[ Q^R_i(s,a) - \lambda\, Q^C_i(s,a) \Big]
$$

**Partage de paramètres par type** : tous les soldats partagent un réseau, tous les cavaliers un autre, etc. Six réseaux au total, Roi compris (il est aussi un agent).

**Précision** : avec une récompense commune et un arbitre qui prend le maximum, le système équivaut à une seule fonction de valeur sur tous les coups, structurée par pièce. Le côté multi-agents apporte l'architecture et l'interprétabilité. Pour des dynamiques vraiment multi-agents, on pourra ajouter de la communication entre agents ou un arbitre qui apprend plus tard.

## 6. Points ouverts

- [x] Choisir la récompense : résultat de la partie + façonnage par potentiel (évaluation de Stockfish).
- [x] Choisir l'adversaire : Stockfish.
- [x] Choisir la sécurité du Roi : bouclier anti-mat (dur) + relaxation lagrangienne sur les échecs (souple, d = 5).
- [x] Décider du sort d'un agent soldat promu : changement de réseau et de valeur.
- [ ] Fixer la profondeur ou le temps d'analyse de Stockfish pour $\Phi$.
- [ ] Fixer le niveau de départ et les paliers du curriculum de Stockfish.
- [ ] Choisir le pas $\eta$ et la valeur initiale de $\lambda$.
- [ ] Étendre le bouclier au mat en 2 ?
