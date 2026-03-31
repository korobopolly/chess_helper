"""
Chess Vision - 화면 인식 체스 프로그램
화면에서 체스 보드를 인식하고 Stockfish로 최선의 수를 찾아줍니다.

사용법:
  python chess_vision.py --calibrate          # 초기 보정 (시작 위치 보드 필요)
  python chess_vision.py                      # 분석 모드 (F8/F9로 트리거)
  python chess_vision.py --manual-select      # 수동 보드 영역 선택
"""

import argparse
import time

import chess
import keyboard

from core import load_config, save_config, capture_screen, find_stockfish
from core import ChessAdvisor, FenGenerator
from vision import BoardDetector, PieceRecognizer


def format_score(score):
    """PovScore를 읽기 쉬운 문자열로 변환합니다."""
    white = score.white()
    if white.is_mate():
        mate_in = white.mate()
        return f"M{mate_in}" if mate_in > 0 else f"M{mate_in}"
    cp = white.score()
    return f"{cp/100:+.2f}"


def print_board(pieces):
    """콘솔에 보드를 출력합니다."""
    print("    a  b  c  d  e  f  g  h")
    print("  +------------------------+")
    for row in range(8):
        rank = 8 - row
        line = f"{rank} |"
        for col in range(8):
            piece = pieces[row][col]
            if piece is None:
                if (row + col) % 2 == 0:
                    line += " . "
                else:
                    line += " · "
            else:
                line += f" {piece} "
        line += f"| {rank}"
        print(line)
    print("  +------------------------+")
    print("    a  b  c  d  e  f  g  h")


def main():
    parser = argparse.ArgumentParser(description="Chess Vision - 화면 인식 체스 프로그램")
    parser.add_argument("--stockfish-path", type=str, default=None,
                        help="Stockfish 실행 파일 경로")
    parser.add_argument("--depth", type=int, default=18,
                        help="분석 깊이 (기본: 18)")
    parser.add_argument("--manual-select", action="store_true",
                        help="수동 보드 영역 선택")
    parser.add_argument("--calibrate", action="store_true",
                        help="보정 모드 (시작 위치 보드 필요)")
    parser.add_argument("--turn", choices=["w", "b"], default="w",
                        help="현재 차례 (w=백, b=흑)")
    parser.add_argument("--no-engine", action="store_true",
                        help="엔진 없이 보드 인식만")
    parser.add_argument("--debug", action="store_true",
                        help="디버그 출력 활성화")
    args = parser.parse_args()

    print("=" * 50)
    print("  Chess Vision - 화면 인식 체스 프로그램")
    print("=" * 50)
    print()

    # 컴포넌트 초기화
    detector = BoardDetector()
    recognizer = PieceRecognizer()
    fen_gen = FenGenerator()

    # 설정 로드
    config = load_config()
    if "board_rect" in config:
        detector.cached_rect = tuple(config["board_rect"])
        print(f"[i] 저장된 보드 위치 사용: {detector.cached_rect}")
    if "orientation" in config:
        detector.orientation = config["orientation"]

    # Stockfish 초기화
    advisor = None
    if not args.no_engine:
        sf_path = find_stockfish(args.stockfish_path)
        if sf_path:
            advisor = ChessAdvisor(sf_path, args.depth)
            advisor.start()
        else:
            print("[!] Stockfish를 찾을 수 없습니다.")
            print("    --stockfish-path 옵션으로 경로를 지정하세요.")
            print("    또는 --no-engine 옵션으로 엔진 없이 실행하세요.")
            if not args.calibrate:
                return

    # 보정 모드
    if args.calibrate:
        print("\n[보정 모드]")
        print("체스 보드를 시작 위치로 놓고 화면에 표시하세요.")
        input("준비되면 Enter를 누르세요...")

        screenshot = capture_screen()
        board_rect = detector.detect_board(screenshot, manual=True)
        if board_rect is None:
            print("[!] 보드를 찾을 수 없습니다.")
            return

        x, y, w, h = board_rect
        board_img = screenshot[y:y + h, x:x + w]

        # 템플릿 추출
        recognizer.calibrate(board_img)

        # 기물 인식 결과로 방향 감지
        pieces = recognizer.recognize(board_img)
        detector.detect_orientation_from_pieces(pieces)
        print(f"[i] 보드 방향: {detector.orientation}")

        # 설정 저장
        config["board_rect"] = list(board_rect)
        config["orientation"] = detector.orientation
        save_config(config)
        print("[+] 설정이 저장되었습니다.")

        if advisor:
            advisor.stop()
        return

    # 분석 모드
    turn = "w"

    print(f"\n[분석 모드]")
    print(f"  F7:  수동 FEN 입력")
    print(f"  F8:  백 진영 분석")
    print(f"  F9:  흑 진영 분석")
    print(f"  F6:  보드 위치 재설정")
    print(f"  F10: 종료")

    if not recognizer.has_templates():
        print("\n[!] 템플릿이 없습니다. --calibrate로 먼저 보정하세요.")
        print("    또는 색상 기반 폴백을 사용합니다 (정확도 낮음).")

    print("\n대기 중... (F7: 수동 FEN / F8: 백 분석 / F9: 흑 분석)")

    running = True

    def on_quit():
        nonlocal running
        running = False

    def on_reset():
        detector.cached_rect = None
        print("[i] 보드 위치가 초기화되었습니다.")

    keyboard.add_hotkey("f10", on_quit)
    keyboard.add_hotkey("f6", on_reset)

    def analyze_fen(fen):
        """FEN을 엔진으로 분석하여 최선의 수를 출력합니다."""
        if not advisor:
            print(f"  FEN: {fen}")
            return

        best_move, score, lines = advisor.get_best_move(fen)
        if not best_move:
            print("[!] 분석 실패. FEN을 확인하세요.")
            return

        board_obj = chess.Board(fen)
        san = board_obj.san(best_move)
        uci = best_move.uci()
        print(f"\n  ★ {san} ({uci})  {format_score(score)}")
        if len(lines) > 1:
            for i, (mv, sc) in enumerate(lines[:3]):
                mv_san = board_obj.san(mv)
                print(f"    {i+1}. {mv_san} ({mv.uci()})  {format_score(sc)}")
        print()

    def on_analyze(side):
        nonlocal turn

        turn = "w" if side == "white" else "b"
        label = "백" if side == "white" else "흑"
        print(f"\n[*] 분석 중... ({label})")

        screenshot = capture_screen()

        board_rect = detector.detect_board(screenshot, manual=args.manual_select)
        if board_rect is None:
            print("[!] 보드를 찾을 수 없습니다.")
            return

        x, y, w, h = board_rect
        board_img = screenshot[y:y + h, x:x + w]

        config["board_rect"] = list(board_rect)
        save_config(config)

        pieces = recognizer.recognize(board_img)

        detector.orientation = side
        fen = fen_gen.board_to_fen(pieces, detector.orientation, turn)

        if args.debug:
            total = sum(1 for r in pieces for c in r if c is not None)
            print(f"  [DEBUG] 감지: {total}개, rect={board_rect}")
            print(f"  FEN: {fen}")
            print_board(pieces)

        analyze_fen(fen)

    def on_manual_fen():
        """수동 FEN 입력 모드."""
        print("\n[수동 입력 모드]")
        fen = input("  FEN 입력: ").strip()
        if not fen:
            print("[!] FEN이 비어있습니다.")
            return
        try:
            chess.Board(fen)
        except ValueError as e:
            print(f"[!] 잘못된 FEN: {e}")
            return
        print(f"  FEN: {fen}")
        analyze_fen(fen)

    keyboard.add_hotkey("f7", on_manual_fen)
    keyboard.add_hotkey("f8", lambda: on_analyze("white"))
    keyboard.add_hotkey("f9", lambda: on_analyze("black"))

    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if advisor:
            advisor.stop()
        keyboard.unhook_all()
        print("\n[+] 종료.")


if __name__ == "__main__":
    main()
