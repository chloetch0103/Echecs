import chess
from .adversaire import Adversaire, Evaluateur


class Environnement:
    """
    Donne les informations sur la configuration de l'échiquier, les coups légaux
    par agent, en tant que qui l'algorithme joue, la récompense accordée à
    celle-ci (r_t) et indique si le roi est en échec (c_t).
    """
    def __init__(self, couleur: bool, adversaire: Adversaire, evaluateur: Evaluateur, gamma):
        self.echiquier = None
        self.phi = None
        self.couleur = couleur  # True pour blanc et False pour Noir
        self.coups_legaux_par_agent = None
        self.adversaire = adversaire
        self.evaluateur = evaluateur
        self.gamma = gamma

    def compute_coups_legaux_par_agent(self):
        # 1. Premier niveau du dictionnaire {type pièce : {case de départ : ...}}
        self.coups_legaux_par_agent = dict()
        for type_piece in chess.PIECE_TYPES:
            self.coups_legaux_par_agent[type_piece] = dict()
            for square in self.echiquier.pieces(type_piece, self.couleur):
                position = square
                self.coups_legaux_par_agent[type_piece][position] = list()

        # 2. Second niveau du dictionnaire {type pièce : {case de départ : liste des positions d'arrivée possible}}
        for coup in self.echiquier.legal_moves:
            type_piece = self.echiquier.piece_at(coup.from_square).piece_type
            self.coups_legaux_par_agent[type_piece][coup.from_square].append(coup)

    def reset(self, fen=None):
        self.echiquier = chess.Board(fen) if fen else chess.Board()
        if self.echiquier.turn != self.couleur:  # l'adversaire a le trait
            self.echiquier.push(self.adversaire.jouer(self.echiquier))
        self.phi = self.evaluateur.potentiel(self.echiquier, self.couleur)

    def _z(self, issue):
        if issue.winner is None:
            return 0
        return 1 if issue.winner == self.couleur else -1

    def step(self, coup):
        """
        Permet de faire avancer la partie
        :param coup: coup de l'algorithme
        :return: une copie de l'échiquier après coup, r_t, c_t et un booléen si indique si la partie est terminée
        """
        fin = False
        phi_avant = self.phi
        c_t = 0

        # 1. La partie s'est achevée au tour précédent
        issue = self.echiquier.outcome(claim_draw=True)
        if issue is not None:
            raise RuntimeError("step() appelé sur une partie terminée")

        # 2. L'agorithme joue + on teste si la partie s'achève
        self.echiquier.push(coup)
        issue = self.echiquier.outcome(claim_draw=True)
        if issue is not None:
            fin = True
            z = self._z(issue)
            r_t = z - phi_avant
        else:

            # 3. L'adversaire joue + on teste si la partie s'achève
            coup_stockfish = self.adversaire.jouer(self.echiquier)
            self.echiquier.push(coup_stockfish)
            c_t = 1 if self.echiquier.is_check() else 0
            issue = self.echiquier.outcome(claim_draw=True)
            if issue is not None:
                fin = True
                z = self._z(issue)
                r_t = z - phi_avant
            else:
                self.phi = self.evaluateur.potentiel(self.echiquier, self.couleur)
                r_t = self.gamma * self.phi - phi_avant
        return self.echiquier.copy(), r_t, c_t, fin

