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


if __name__ == "__main__":
    print(bilan_materiel(chess.Board(), chess.WHITE))                                  # 0
    print(bilan_materiel(chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                         chess.WHITE))                                                  # 9 : les Noirs n'ont plus de dame