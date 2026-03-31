"""
Chess Vision - 화면 인식 체스 프로그램
화면에서 체스 보드를 인식하고 Stockfish로 최선의 수를 찾아 자동으로 둡니다.

사용법:
  python chess_vision.py --calibrate          # 초기 보정 (시작 위치 보드 필요)
  python chess_vision.py                      # 분석 모드 (F9로 트리거)
  python chess_vision.py --auto-play          # 자동 플레이 모드
  python chess_vision.py --manual-select      # 수동 보드 영역 선택
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import keyboard
import mss
import numpy as np
import pyautogui
import chess
import chess.engine
from PIL import Image

# ─── 설정 ───────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
TEMPLATES_DIR = SCRIPT_DIR / "templates"
CONFIG_FILE = SCRIPT_DIR / "config.json"

SQUARE_SIZE = 64  # 템플릿 정규화 크기
MATCH_THRESHOLD = 0.15  # 템플릿 매칭 신뢰도 임계값 (낮춤)
EMPTY_VARIANCE_THRESHOLD = 800  # 빈 칸 분산 임계값

PIECE_NAMES = {
    "wK": "K", "wQ": "Q", "wR": "R", "wB": "B", "wN": "N", "wP": "P",
    "bK": "k", "bQ": "q", "bR": "r", "bB": "b", "bN": "n", "bP": "p",
}

# Stockfish 기본 경로 후보
STOCKFISH_PATHS = [
    r"C:\stockfish\stockfish-windows-x86-64-avx2.exe",
    r"C:\stockfish\stockfish.exe",
    r"C:\Program Files\Stockfish\stockfish.exe",
    r"C:\Program Files (x86)\Stockfish\stockfish.exe",
    r"stockfish.exe",
    r"stockfish",
]

# ─── 유틸 ───────────────────────────────────────────────────────────


def load_config():
    """저장된 설정(보드 위치 등)을 로드합니다."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def save_config(cfg):
    """설정을 파일에 저장합니다."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def capture_screen():
    """전체 화면을 캡처하여 numpy 배열로 반환합니다."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 주 모니터
        img = sct.grab(monitor)
        frame = np.array(img)
        # BGRA -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame


def find_stockfish(user_path=None):
    """Stockfish 실행 파일을 찾습니다."""
    if user_path and os.path.isfile(user_path):
        return user_path
    for p in STOCKFISH_PATHS:
        if os.path.isfile(p):
            return p
    # PATH에서 검색
    import shutil
    found = shutil.which("stockfish")
    if found:
        return found
    return None


# ─── 보드 감지 ──────────────────────────────────────────────────────


class BoardDetector:
    """화면에서 체스 보드 영역을 감지합니다."""

    def __init__(self):
        self.cached_rect = None
        self.orientation = "white"  # "white" = 백이 아래

    def detect_board(self, screenshot, manual=False):
        """
        체스 보드 영역을 감지합니다.
        Returns: (x, y, w, h) 또는 None
        """
        if manual:
            return self._manual_select(screenshot)

        # 캐시된 위치가 있으면 사용
        if self.cached_rect:
            return self.cached_rect

        # 자동 감지 시도
        rect = self._auto_detect(screenshot)
        if rect:
            self.cached_rect = rect
            return rect

        # 실패시 수동 선택
        print("[!] 자동 감지 실패. 수동으로 보드 영역을 선택하세요.")
        return self._manual_select(screenshot)

    def _auto_detect(self, screenshot):
        """색상 분석으로 체스 보드를 자동 감지합니다."""
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # 여러 체스 사이트의 보드 색상 패턴 시도
        candidates = []

        # chess.com 녹색 테마: 녹색 칸 감지
        green_mask = cv2.inRange(hsv, (35, 40, 80), (85, 255, 200))
        candidates.append(self._find_board_from_mask(green_mask, screenshot))

        # lichess 갈색 테마
        brown_mask = cv2.inRange(hsv, (10, 30, 80), (30, 180, 200))
        candidates.append(self._find_board_from_mask(brown_mask, screenshot))

        # 일반적인 방법: 에지 검출 + 직선 감지
        candidates.append(self._find_board_by_lines(gray))

        # 가장 큰 정사각형 후보 선택
        best = None
        best_area = 0
        for c in candidates:
            if c is None:
                continue
            x, y, w, h = c
            # 정사각형에 가까운지 확인
            ratio = min(w, h) / max(w, h)
            if ratio > 0.85 and w * h > best_area and w > 100:
                best = c
                best_area = w * h

        return best

    def _find_board_from_mask(self, mask, screenshot):
        """색상 마스크에서 보드 영역을 찾습니다."""
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 면적이 가장 큰 contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # 최소 크기 확인
        if w < 100 or h < 100:
            return None

        # 정사각형으로 보정
        size = max(w, h)
        return (x, y, size, size)

    def _find_board_by_lines(self, gray):
        """직선 검출로 보드를 찾습니다."""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return None

        # 수평/수직 직선 필터링
        h_lines = []
        v_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 5 or angle > 175:
                h_lines.append((min(y1, y2), max(y1, y2)))
            elif 85 < angle < 95:
                v_lines.append((min(x1, x2), max(x1, x2)))

        if len(h_lines) < 4 or len(v_lines) < 4:
            return None

        # 직선들의 범위로 보드 영역 추정
        h_coords = [y for pair in h_lines for y in pair]
        v_coords = [x for pair in v_lines for x in pair]

        y_min, y_max = min(h_coords), max(h_coords)
        x_min, x_max = min(v_coords), max(v_coords)

        w = x_max - x_min
        h = y_max - y_min
        if w < 100 or h < 100:
            return None

        size = max(w, h)
        return (x_min, y_min, size, size)

    def _manual_select(self, screenshot):
        """사용자가 마우스로 보드 영역을 선택합니다."""
        print("[i] 체스 보드 영역을 드래그하여 선택하세요. Enter로 확인, C로 취소.")

        # 화면을 축소하여 표시 (고해상도 대응)
        h, w = screenshot.shape[:2]
        scale = min(1.0, 1200 / w, 800 / h)
        display = cv2.resize(screenshot, None, fx=scale, fy=scale)

        roi = cv2.selectROI("체스 보드 선택", display, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi[2] == 0 or roi[3] == 0:
            return None

        # 원본 크기로 변환
        x = int(roi[0] / scale)
        y = int(roi[1] / scale)
        w = int(roi[2] / scale)
        h = int(roi[3] / scale)

        size = max(w, h)
        self.cached_rect = (x, y, size, size)
        return self.cached_rect

    def detect_orientation(self, board_img):
        """보드 방향을 감지합니다 (백이 아래인지 위인지)."""
        # 보드 하단 1/4과 상단 1/4의 기물 밀도 비교
        h = board_img.shape[0]
        sq = h // 8

        # 2번째 랭크 (아래에서 두 번째 줄)의 밝기 분석
        bottom_rank2 = board_img[6 * sq:7 * sq, :]
        top_rank2 = board_img[sq:2 * sq, :]

        # 백 폰은 밝은 색, 흑 폰은 어두운 색
        bottom_bright = np.mean(bottom_rank2)
        top_bright = np.mean(top_rank2)

        # 밝은 기물(백)이 아래에 있으면 white 방향
        self.orientation = "white" if bottom_bright >= top_bright else "black"
        return self.orientation

    def detect_orientation_from_pieces(self, pieces_8x8):
        """인식된 기물 배열을 기반으로 보드 방향을 감지합니다.
        아래쪽(row 6-7)에 백 기물이 많으면 white, 흑 기물이 많으면 black."""
        bottom_white = 0
        bottom_black = 0
        top_white = 0
        top_black = 0

        for row in range(6, 8):
            for col in range(8):
                p = pieces_8x8[row][col]
                if p and p.isupper():
                    bottom_white += 1
                elif p and p.islower():
                    bottom_black += 1

        for row in range(0, 2):
            for col in range(8):
                p = pieces_8x8[row][col]
                if p and p.isupper():
                    top_white += 1
                elif p and p.islower():
                    top_black += 1

        white_score = bottom_white - top_white
        black_score = bottom_black - top_black

        if white_score != black_score:
            self.orientation = "white" if white_score > black_score else "black"
        # 동점이면 기존 orientation 유지

        return self.orientation


# ─── 기물 인식 ──────────────────────────────────────────────────────


class PieceRecognizer:
    """
    체스 기물을 인식합니다.

    2단계 인식:
    1. 형태 매칭: 배경색 제거 후 실루엣으로 기물 종류(K,Q,R,B,N,P) 판별
    2. 색상 판별: 기물 픽셀의 절대 밝기로 백/흑 판별
    """

    # 기물 종류 (색상 무시) 매핑: wK/bK -> "K", wQ/bQ -> "Q", ...
    PIECE_TYPES = {}
    for _name in PIECE_NAMES:
        PIECE_TYPES[_name] = _name[1]  # "wK" -> "K", "bP" -> "P"

    def __init__(self):
        self.type_templates = {}  # piece_type ("K","Q",...) -> normalized template
        self.color_refs = {}  # "white"/"black" -> average piece brightness
        self._load_templates()

    @staticmethod
    def _get_background_color(gray_square):
        """칸의 모서리 픽셀에서 배경색을 추출합니다."""
        h, w = gray_square.shape
        margin = max(2, h // 10)
        corners = np.concatenate([
            gray_square[:margin, :margin].flatten(),
            gray_square[:margin, -margin:].flatten(),
            gray_square[-margin:, :margin].flatten(),
            gray_square[-margin:, -margin:].flatten(),
        ])
        return float(np.median(corners))

    @staticmethod
    def _normalize_square(gray_square):
        """
        엣지 기반 정규화: 기물 윤곽선만 추출합니다.
        밝기/배경에 무관하게 안정적인 매칭이 가능합니다.
        """
        blurred = cv2.GaussianBlur(gray_square, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    @staticmethod
    def _has_piece(gray_square):
        """칸에 기물이 있는지 판별합니다."""
        h, w = gray_square.shape
        center = gray_square[h // 4:3 * h // 4, w // 4:3 * w // 4]

        # 기물은 윤곽/그림자로 인해 픽셀 변동이 큼
        if np.std(center) > 15:
            return True

        # 배경 대비 차이 검사 (폴백)
        bg = PieceRecognizer._get_background_color(gray_square)
        diff = np.abs(center.astype(float) - bg)
        significant_pixels = np.sum(diff > 25) / diff.size
        return significant_pixels > 0.08

    def _detect_piece_color(self, gray_square, color_square=None):
        """
        기물의 색상으로 백/흑을 판별합니다.
        컬러 이미지와 캘리브레이션 데이터가 있으면 BGR 색상 거리를 사용합니다.
        """
        bg = PieceRecognizer._get_background_color(gray_square)
        h, w = gray_square.shape
        center_gray = gray_square[h // 4:3 * h // 4, w // 4:3 * w // 4]

        # 배경과 다른 픽셀 (기물 픽셀) 마스크
        diff = np.abs(center_gray.astype(float) - bg)
        piece_mask = diff > 20

        # 윤곽/그림자 제거: 마스크를 침식하여 기물 몸통만 추출
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        body_mask = cv2.erode(piece_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
        # 몸통 픽셀이 너무 적으면 원래 마스크 사용
        if np.sum(body_mask) < 10:
            body_mask = piece_mask

        # 컬러 이미지 + 캘리브레이션 참조가 있으면 BGR 색상 거리로 판별
        if color_square is not None and self.color_refs and "white_bgr" in self.color_refs:
            ch, cw = color_square.shape[:2]
            center_color = color_square[ch // 4:3 * ch // 4, cw // 4:3 * cw // 4]

            if np.sum(body_mask) >= 10:
                piece_bgr = center_color[body_mask].mean(axis=0)
            else:
                piece_bgr = center_color.reshape(-1, 3).mean(axis=0).astype(float)

            white_ref = np.array(self.color_refs["white_bgr"])
            black_ref = np.array(self.color_refs["black_bgr"])
            dist_white = np.linalg.norm(piece_bgr - white_ref)
            dist_black = np.linalg.norm(piece_bgr - black_ref)
            return "w" if dist_white <= dist_black else "b"

        # 폴백: 밝기 기반
        if np.sum(body_mask) < 10:
            piece_brightness = float(np.mean(center_gray))
        else:
            piece_brightness = float(np.mean(center_gray[body_mask]))

        if self.color_refs and "threshold" in self.color_refs:
            threshold = self.color_refs["threshold"]
            return "w" if piece_brightness >= threshold else "b"

        if piece_brightness > 160:
            return "w"
        elif piece_brightness < 100:
            return "b"
        else:
            return "w" if piece_brightness > bg else "b"

    def _load_templates(self):
        """저장된 형태 템플릿과 색상 참조값을 로드합니다."""
        self.type_templates = {}

        # 종류별 정규화 템플릿 로드 (K, Q, R, B, N, P)
        for piece_type in ["K", "Q", "R", "B", "N", "P"]:
            path = TEMPLATES_DIR / f"type_{piece_type}_norm.png"
            if path.exists():
                tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    tmpl = cv2.resize(tmpl, (SQUARE_SIZE, SQUARE_SIZE))
                    self.type_templates[piece_type] = tmpl

        # 색상 참조값 로드
        color_path = TEMPLATES_DIR / "color_refs.json"
        if color_path.exists():
            with open(color_path, "r") as f:
                self.color_refs = json.load(f)

    def has_templates(self):
        """템플릿이 로드되었는지 확인합니다."""
        return len(self.type_templates) == 6

    def calibrate(self, board_img):
        """
        시작 위치 보드에서 템플릿을 추출합니다.
        형태 템플릿: 백/흑 같은 종류 기물의 정규화 실루엣을 합쳐서 저장
        색상 참조: 백/흑 기물의 평균 밝기를 기록
        """
        print("[i] 시작 위치에서 기물 템플릿을 추출합니다...")
        h, w = board_img.shape[:2]
        sq_h = h // 8
        sq_w = w // 8

        start_position = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
        ]

        TEMPLATES_DIR.mkdir(exist_ok=True)

        # 종류별 정규화 샘플 수집 + 색상별 밝기/BGR 수집
        type_samples = {}   # "K" -> [normalized images]
        white_brightnesses = []
        black_brightnesses = []
        white_bgrs = []
        black_bgrs = []

        for row in range(8):
            for col in range(8):
                piece = start_position[row][col]
                if piece is None:
                    continue

                y1 = row * sq_h
                y2 = (row + 1) * sq_h
                x1 = col * sq_w
                x2 = (col + 1) * sq_w
                square = board_img[y1:y2, x1:x2]

                color_resized = cv2.resize(square, (SQUARE_SIZE, SQUARE_SIZE))
                gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (SQUARE_SIZE, SQUARE_SIZE))

                # 원본 저장 (디버깅용)
                cv2.imwrite(str(TEMPLATES_DIR / f"{piece}.png"), gray)

                # 정규화 실루엣
                norm = self._normalize_square(gray)
                piece_type = piece[1]  # "K","Q","R","B","N","P"
                if piece_type not in type_samples:
                    type_samples[piece_type] = []
                type_samples[piece_type].append(norm)

                # 기물 몸통 픽셀의 밝기 + BGR 색상 수집
                bg = self._get_background_color(gray)
                center_gray = gray[SQUARE_SIZE // 4:3 * SQUARE_SIZE // 4,
                                   SQUARE_SIZE // 4:3 * SQUARE_SIZE // 4]
                center_color = color_resized[SQUARE_SIZE // 4:3 * SQUARE_SIZE // 4,
                                             SQUARE_SIZE // 4:3 * SQUARE_SIZE // 4]
                diff = np.abs(center_gray.astype(float) - bg)
                piece_mask = diff > 20

                # 윤곽/그림자 제거: 몸통만 추출
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                body_mask = cv2.erode(piece_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
                if np.sum(body_mask) < 10:
                    body_mask = piece_mask

                if np.sum(body_mask) > 10:
                    brightness = float(np.mean(center_gray[body_mask]))
                    piece_bgr = center_color[body_mask].mean(axis=0).tolist()
                else:
                    brightness = float(np.mean(center_gray))
                    piece_bgr = center_color.reshape(-1, 3).mean(axis=0).tolist()

                if piece.startswith("w"):
                    white_brightnesses.append(brightness)
                    white_bgrs.append(piece_bgr)
                else:
                    black_brightnesses.append(brightness)
                    black_bgrs.append(piece_bgr)

        # 종류별 평균 정규화 템플릿 저장
        for piece_type, samples in type_samples.items():
            avg = np.mean(samples, axis=0).astype(np.uint8)
            path = TEMPLATES_DIR / f"type_{piece_type}_norm.png"
            cv2.imwrite(str(path), avg)
            print(f"  [+] {piece_type} 형태 템플릿 저장 ({len(samples)}개 샘플)")

        # 색상 참조값 저장
        white_bgr_mean = np.mean(white_bgrs, axis=0).tolist()
        black_bgr_mean = np.mean(black_bgrs, axis=0).tolist()
        color_refs = {
            "white_mean": float(np.mean(white_brightnesses)),
            "black_mean": float(np.mean(black_brightnesses)),
            "threshold": float((np.mean(white_brightnesses) + np.mean(black_brightnesses)) / 2),
            "white_bgr": white_bgr_mean,
            "black_bgr": black_bgr_mean,
        }
        with open(TEMPLATES_DIR / "color_refs.json", "w") as f:
            json.dump(color_refs, f, indent=2)
        self.color_refs = color_refs
        print(f"  [+] 색상 참조: 백={color_refs['white_mean']:.0f}, "
              f"흑={color_refs['black_mean']:.0f}, "
              f"임계={color_refs['threshold']:.0f}")
        print(f"  [+] BGR 참조: 백={[f'{v:.0f}' for v in white_bgr_mean]}, "
              f"흑={[f'{v:.0f}' for v in black_bgr_mean]}")

        # 빈 칸 저장
        for row, col, name in [(3, 0, "empty_dark"), (3, 1, "empty_light")]:
            y1 = row * sq_h
            y2 = (row + 1) * sq_h
            x1 = col * sq_w
            x2 = (col + 1) * sq_w
            square = board_img[y1:y2, x1:x2]
            gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (SQUARE_SIZE, SQUARE_SIZE))
            cv2.imwrite(str(TEMPLATES_DIR / f"{name}.png"), gray)

        self._load_templates()
        print(f"[+] 보정 완료! {len(self.type_templates)}개 형태 템플릿 로드됨.")

    def recognize(self, board_img):
        """
        보드 이미지에서 각 칸의 기물을 인식합니다.
        Returns: 8x8 리스트 (FEN 문자, 빈 칸은 None)
        """
        h, w = board_img.shape[:2]
        sq_h = h // 8
        sq_w = w // 8

        board = [[None] * 8 for _ in range(8)]

        for row in range(8):
            for col in range(8):
                y1 = row * sq_h
                y2 = (row + 1) * sq_h
                x1 = col * sq_w
                x2 = (col + 1) * sq_w
                square = board_img[y1:y2, x1:x2]

                color_resized = cv2.resize(square, (SQUARE_SIZE, SQUARE_SIZE))
                gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (SQUARE_SIZE, SQUARE_SIZE))

                piece = self._match_piece(gray, color_resized)
                board[row][col] = piece

        return board

    def _match_piece(self, square_gray, color_square=None):
        """
        2단계 인식:
        1. 형태 매칭 → 기물 종류 (K, Q, R, B, N, P)
        2. 색상 판별 → 백/흑 (w/b)
        """
        # 1단계: 기물 유무 판별
        if not self._has_piece(square_gray):
            return None

        # 2단계: 형태 매칭으로 기물 종류 판별
        if not self.type_templates:
            return self._match_by_color_only(square_gray, color_square)

        normalized = self._normalize_square(square_gray)

        best_type = None
        best_score = -1

        for piece_type, tmpl in self.type_templates.items():
            result = cv2.matchTemplate(normalized, tmpl, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            if score > best_score:
                best_score = score
                best_type = piece_type

        if best_score < MATCH_THRESHOLD:
            return None

        # 3단계: 색상 판별 → 백/흑
        color = self._detect_piece_color(square_gray, color_square)

        # FEN 문자: 백=대문자, 흑=소문자
        if color == "w":
            return best_type.upper()
        else:
            return best_type.lower()

    def _match_by_color_only(self, square_gray, color_square=None):
        """템플릿 없이 색상만으로 판별 (폴백). P/p로만 표시."""
        color = self._detect_piece_color(square_gray, color_square)
        return "P" if color == "w" else "p"


# ─── FEN 생성 ───────────────────────────────────────────────────────


class FenGenerator:
    """8x8 기물 배열을 FEN 문자열로 변환합니다."""

    @staticmethod
    def board_to_fen(pieces_8x8, orientation="white", turn="w"):
        """
        pieces_8x8: 8x8 리스트, 각 원소는 FEN 문자(K,Q,R,B,N,P,k,q,r,b,n,p) 또는 None
        orientation: "white" = 백이 아래 (rank 8이 row 0)
        turn: "w" 또는 "b"
        Returns: FEN 문자열
        """
        rows = pieces_8x8
        if orientation == "black":
            # 보드를 180도 회전
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

        # 캐슬링 권리 추론 (간단한 휴리스틱)
        castling = FenGenerator._infer_castling(pieces_8x8, orientation)

        return f"{fen} {turn} {castling} - 0 1"

    @staticmethod
    def _infer_castling(pieces_8x8, orientation):
        """킹과 룩의 위치로 캐슬링 가능 여부를 추론합니다."""
        castling = ""

        if orientation == "white":
            # rank 1 = row 7, rank 8 = row 0
            # 백 킹사이드: e1에 킹, h1에 룩
            if (pieces_8x8[7][4] == "K" and pieces_8x8[7][7] == "R"):
                castling += "K"
            if (pieces_8x8[7][4] == "K" and pieces_8x8[7][0] == "R"):
                castling += "Q"
            if (pieces_8x8[0][4] == "k" and pieces_8x8[0][7] == "r"):
                castling += "k"
            if (pieces_8x8[0][4] == "k" and pieces_8x8[0][0] == "r"):
                castling += "q"
        else:
            # 흑 방향이면 뒤집힌 상태
            if (pieces_8x8[0][3] == "K" and pieces_8x8[0][0] == "R"):
                castling += "K"
            if (pieces_8x8[0][3] == "K" and pieces_8x8[0][7] == "R"):
                castling += "Q"
            if (pieces_8x8[7][3] == "k" and pieces_8x8[7][0] == "r"):
                castling += "k"
            if (pieces_8x8[7][3] == "k" and pieces_8x8[7][7] == "r"):
                castling += "q"

        return castling if castling else "-"


# ─── 체스 엔진 ──────────────────────────────────────────────────────


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
            # 여러 라인 분석
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
            # 엔진이 죽었으면 재시작
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


# ─── 자동 플레이 ────────────────────────────────────────────────────


class MoveExecutor:
    """마우스 클릭으로 수를 실행합니다."""

    def __init__(self, board_rect, orientation="white"):
        self.board_rect = board_rect  # (x, y, w, h)
        self.orientation = orientation

    def square_to_pixel(self, square_name):
        """
        체스 칸 이름 (예: 'e2')을 화면 픽셀 좌표로 변환합니다.
        """
        x, y, w, h = self.board_rect
        sq_size = w / 8

        file_idx = ord(square_name[0]) - ord('a')  # 0-7
        rank_idx = int(square_name[1]) - 1  # 0-7

        if self.orientation == "white":
            px = x + (file_idx + 0.5) * sq_size
            py = y + (7 - rank_idx + 0.5) * sq_size
        else:
            px = x + (7 - file_idx + 0.5) * sq_size
            py = y + (rank_idx + 0.5) * sq_size

        return int(px), int(py)

    def execute_move(self, move, delay=0.15):
        """
        수를 클릭으로 실행합니다.
        move: chess.Move 객체
        """
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        from_px, from_py = self.square_to_pixel(from_sq)
        to_px, to_py = self.square_to_pixel(to_sq)

        print(f"  [>] 클릭: {from_sq}({from_px},{from_py}) -> {to_sq}({to_px},{to_py})")

        pyautogui.click(from_px, from_py)
        time.sleep(delay)
        pyautogui.click(to_px, to_py)

        # 프로모션 처리
        if move.promotion:
            time.sleep(delay)
            # 퀸 프로모션이 기본 (보통 첫 번째 옵션)
            pyautogui.click(to_px, to_py)


# ─── 화면 오버레이 표시 ─────────────────────────────────────────────


def show_analysis(screenshot, board_rect, pieces, fen, best_move, score, lines):
    """분석 결과를 화면에 오버레이로 표시합니다."""
    display = screenshot.copy()
    x, y, w, h = board_rect

    # 보드 영역 표시
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 정보 텍스트
    info_x = x + w + 20
    info_y = y + 30
    line_gap = 30

    texts = [
        f"FEN: {fen[:50]}...",
        f"Best: {best_move}",
        f"Score: {score}",
        "---",
    ]

    for i, line_info in enumerate(lines[:3]):
        move, sc = line_info
        texts.append(f"  {i+1}. {move} ({sc})")

    for i, text in enumerate(texts):
        cv2.putText(display, text, (info_x, info_y + i * line_gap),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 최선의 수 화살표 표시
    if best_move:
        sq_size = w / 8
        from_sq = chess.square_name(best_move.from_square)
        to_sq = chess.square_name(best_move.to_square)

        def sq_to_px(sq_name):
            f = ord(sq_name[0]) - ord('a')
            r = int(sq_name[1]) - 1
            px = int(x + (f + 0.5) * sq_size)
            py = int(y + (7 - r + 0.5) * sq_size)
            return (px, py)

        pt1 = sq_to_px(from_sq)
        pt2 = sq_to_px(to_sq)
        cv2.arrowedLine(display, pt1, pt2, (0, 0, 255), 4, tipLength=0.3)

    # 축소 표시
    scale = min(1.0, 1200 / display.shape[1], 800 / display.shape[0])
    display = cv2.resize(display, None, fx=scale, fy=scale)

    cv2.imshow("Chess Vision - Analysis", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── 메인 루프 ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Chess Vision - 화면 인식 체스 프로그램")
    parser.add_argument("--stockfish-path", type=str, default=None,
                        help="Stockfish 실행 파일 경로")
    parser.add_argument("--depth", type=int, default=18,
                        help="분석 깊이 (기본: 18)")
    parser.add_argument("--auto-play", action="store_true",
                        help="자동 플레이 모드")
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

        # 밝기 기반 방향 초기 감지
        detector.detect_orientation(board_img)

        # 템플릿 추출
        recognizer.calibrate(board_img)

        # 기물 인식 결과로 방향 재감지
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
        print(f"\n  ★ {san}  ({score})")
        if len(lines) > 1:
            for i, (mv, sc) in enumerate(lines[:3]):
                mv_san = board_obj.san(mv)
                print(f"    {i+1}. {mv_san} ({sc})")
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
            board = chess.Board(fen)
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
                # 밝은/어두운 칸 표시
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


if __name__ == "__main__":
    main()
