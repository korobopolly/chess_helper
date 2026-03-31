"""8x8 기물 배열을 FEN 문자열로 변환합니다."""


class FenGenerator:
    """8x8 기물 배열을 FEN 문자열로 변환합니다."""

    @staticmethod
    def board_to_fen(pieces_8x8, orientation="white", turn="w"):
        """
        pieces_8x8: 8x8 리스트, 각 원소는 FEN 문자 또는 None
        orientation: "white" = 백이 아래 (rank 8이 row 0)
        turn: "w" 또는 "b"
        """
        rows = pieces_8x8
        if orientation == "black":
            rows = [list(reversed(row)) for row in reversed(rows)]

        fen_rows = []
        for row in rows:
            fen_row = ""
            empty_count = 0
            for cell in row:
                if cell is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += cell
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)

        fen = "/".join(fen_rows)
        castling = FenGenerator._infer_castling(pieces_8x8, orientation)

        return f"{fen} {turn} {castling} - 0 1"

    @staticmethod
    def _infer_castling(pieces_8x8, orientation):
        """킹과 룩의 위치로 캐슬링 가능 여부를 추론합니다."""
        castling = ""

        if orientation == "white":
            if (pieces_8x8[7][4] == "K" and pieces_8x8[7][7] == "R"):
                castling += "K"
            if (pieces_8x8[7][4] == "K" and pieces_8x8[7][0] == "R"):
                castling += "Q"
            if (pieces_8x8[0][4] == "k" and pieces_8x8[0][7] == "r"):
                castling += "k"
            if (pieces_8x8[0][4] == "k" and pieces_8x8[0][0] == "r"):
                castling += "q"
        else:
            if (pieces_8x8[0][3] == "K" and pieces_8x8[0][0] == "R"):
                castling += "K"
            if (pieces_8x8[0][3] == "K" and pieces_8x8[0][7] == "R"):
                castling += "Q"
            if (pieces_8x8[7][3] == "k" and pieces_8x8[7][0] == "r"):
                castling += "k"
            if (pieces_8x8[7][3] == "k" and pieces_8x8[7][7] == "r"):
                castling += "q"

        return castling if castling else "-"
