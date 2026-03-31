"""체스 기물을 인식합니다."""

import json

import cv2
import numpy as np

from config import TEMPLATES_DIR, SQUARE_SIZE, MATCH_THRESHOLD, PIECE_NAMES


class PieceRecognizer:
    """
    체스 기물을 인식합니다.

    2단계 인식:
    1. 형태 매칭: 배경색 제거 후 실루엣으로 기물 종류(K,Q,R,B,N,P) 판별
    2. 색상 판별: 기물 픽셀의 절대 밝기로 백/흑 판별
    """

    PIECE_TYPES = {}
    for _name in PIECE_NAMES:
        PIECE_TYPES[_name] = _name[1]

    def __init__(self):
        self.type_templates = {}
        self.color_refs = {}
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
        """엣지 기반 정규화: 기물 윤곽선만 추출합니다."""
        blurred = cv2.GaussianBlur(gray_square, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    @staticmethod
    def _has_piece(gray_square):
        """칸에 기물이 있는지 판별합니다."""
        h, w = gray_square.shape
        center = gray_square[h // 4:3 * h // 4, w // 4:3 * w // 4]

        if np.std(center) > 15:
            return True

        bg = PieceRecognizer._get_background_color(gray_square)
        diff = np.abs(center.astype(float) - bg)
        significant_pixels = np.sum(diff > 25) / diff.size
        return significant_pixels > 0.08

    def _detect_piece_color(self, gray_square, color_square=None):
        """기물의 색상으로 백/흑을 판별합니다."""
        bg = PieceRecognizer._get_background_color(gray_square)
        h, w = gray_square.shape
        center_gray = gray_square[h // 4:3 * h // 4, w // 4:3 * w // 4]

        diff = np.abs(center_gray.astype(float) - bg)
        piece_mask = diff > 20

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        body_mask = cv2.erode(piece_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
        if np.sum(body_mask) < 10:
            body_mask = piece_mask

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

        for piece_type in ["K", "Q", "R", "B", "N", "P"]:
            path = TEMPLATES_DIR / f"type_{piece_type}_norm.png"
            if path.exists():
                tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    tmpl = cv2.resize(tmpl, (SQUARE_SIZE, SQUARE_SIZE))
                    self.type_templates[piece_type] = tmpl

        color_path = TEMPLATES_DIR / "color_refs.json"
        if color_path.exists():
            with open(color_path, "r") as f:
                self.color_refs = json.load(f)

    def has_templates(self):
        """템플릿이 로드되었는지 확인합니다."""
        return len(self.type_templates) == 6

    def calibrate(self, board_img):
        """시작 위치 보드에서 템플릿을 추출합니다."""
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

        type_samples = {}
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

                cv2.imwrite(str(TEMPLATES_DIR / f"{piece}.png"), gray)

                norm = self._normalize_square(gray)
                piece_type = piece[1]
                if piece_type not in type_samples:
                    type_samples[piece_type] = []
                type_samples[piece_type].append(norm)

                bg = self._get_background_color(gray)
                center_gray = gray[SQUARE_SIZE // 4:3 * SQUARE_SIZE // 4,
                                   SQUARE_SIZE // 4:3 * SQUARE_SIZE // 4]
                center_color = color_resized[SQUARE_SIZE // 4:3 * SQUARE_SIZE // 4,
                                             SQUARE_SIZE // 4:3 * SQUARE_SIZE // 4]
                diff = np.abs(center_gray.astype(float) - bg)
                piece_mask = diff > 20

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

        for piece_type, samples in type_samples.items():
            avg = np.mean(samples, axis=0).astype(np.uint8)
            path = TEMPLATES_DIR / f"type_{piece_type}_norm.png"
            cv2.imwrite(str(path), avg)
            print(f"  [+] {piece_type} 형태 템플릿 저장 ({len(samples)}개 샘플)")

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
        """보드 이미지에서 각 칸의 기물을 인식합니다."""
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
        """2단계 인식: 형태 매칭 → 색상 판별."""
        if not self._has_piece(square_gray):
            return None

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

        color = self._detect_piece_color(square_gray, color_square)

        if color == "w":
            return best_type.upper()
        else:
            return best_type.lower()

    def _match_by_color_only(self, square_gray, color_square=None):
        """템플릿 없이 색상만으로 판별 (폴백). P/p로만 표시."""
        color = self._detect_piece_color(square_gray, color_square)
        return "P" if color == "w" else "p"
