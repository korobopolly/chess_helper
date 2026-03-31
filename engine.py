"""Stockfish 엔진 연동."""

import chess
import chess.engine


class ChessAdvisor:
    """Stockfish를 사용하여 최선의 수를 분석합니다."""

    def __init__(self, stockfish_path, depth=18):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.engine = None

    def start(self):
        """엔진을 시작합니다."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            print(f"[+] Stockfish 엔진 시작: {self.stockfish_path}")
        except Exception as e:
            print(f"[!] Stockfish 시작 실패: {e}")
            self.engine = None

    def stop(self):
        """엔진을 종료합니다."""
        if self.engine:
            self.engine.quit()
            self.engine = None

    def get_best_move(self, fen, num_lines=3):
        """
        FEN에서 최선의 수를 찾습니다.
        Returns: (best_move, score, top_lines)
        """
        if not self.engine:
            print("[!] 엔진이 시작되지 않았습니다.")
            return None, None, []

        try:
            board = chess.Board(fen)
        except ValueError as e:
            print(f"[!] 잘못된 FEN: {fen}")
            print(f"    오류: {e}")
            return None, None, []

        if not board.is_valid():
            print(f"[!] 유효하지 않은 보드 상태: {fen}")
            return None, None, []

        try:
            result = self.engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth),
                multipv=min(num_lines, 3)
            )

            if isinstance(result, list):
                lines = []
                for info in result:
                    move = info.get("pv", [None])[0]
                    score = info.get("score")
                    if move and score:
                        lines.append((move, score))

                if lines:
                    best_move, best_score = lines[0]
                    return best_move, best_score, lines
            else:
                move = result.get("pv", [None])[0]
                score = result.get("score")
                return move, score, [(move, score)]

        except Exception as e:
            print(f"[!] 분석 오류: {e}")
            self._restart_engine()

        return None, None, []

    def _restart_engine(self):
        """죽은 엔진을 재시작합니다."""
        print("[i] Stockfish 엔진 재시작 중...")
        try:
            if self.engine:
                try:
                    self.engine.quit()
                except Exception:
                    pass
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            print("[+] 엔진 재시작 완료.")
        except Exception as e:
            print(f"[!] 엔진 재시작 실패: {e}")
            self.engine = None
