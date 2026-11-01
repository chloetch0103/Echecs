import chess

VALEURS: dict[chess.PieceType, float] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.25,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def bilan_materiel(echiquier: chess.Board, couleur: chess.Color) -> float:
    """Matériel de l'algorithme moins celui de l'adversaire. Le Roi ne compte pas."""
    return sum(
        valeur * (len(echiquier.pieces(piece, couleur)) - len(echiquier.pieces(piece, not couleur)))
        for piece, valeur in VALEURS.items()
    )


