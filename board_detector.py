"""화면에서 체스 보드 영역을 감지합니다."""

import cv2
import numpy as np


class BoardDetector:
    """화면에서 체스 보드 영역을 감지합니다."""

    def __init__(self):
        self.cached_rect = None
        self.orientation = "white"

    def detect_board(self, screenshot, manual=False):
        """
        체스 보드 영역을 감지합니다.
        Returns: (x, y, w, h) 또는 None
        """
        if manual:
            return self._manual_select(screenshot)

        if self.cached_rect:
            return self.cached_rect

        rect = self._auto_detect(screenshot)
        if rect:
            self.cached_rect = rect
            return rect

        print("[!] 자동 감지 실패. 수동으로 보드 영역을 선택하세요.")
        return self._manual_select(screenshot)

    def _auto_detect(self, screenshot):
        """색상 분석으로 체스 보드를 자동 감지합니다."""
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        candidates = []

        green_mask = cv2.inRange(hsv, (35, 40, 80), (85, 255, 200))
        candidates.append(self._find_board_from_mask(green_mask, screenshot))

        brown_mask = cv2.inRange(hsv, (10, 30, 80), (30, 180, 200))
        candidates.append(self._find_board_from_mask(brown_mask, screenshot))

        candidates.append(self._find_board_by_lines(gray))

        best = None
        best_area = 0
        for c in candidates:
            if c is None:
                continue
            x, y, w, h = c
            ratio = min(w, h) / max(w, h)
            if ratio > 0.85 and w * h > best_area and w > 100:
                best = c
                best_area = w * h

        return best

    def _find_board_from_mask(self, mask, screenshot):
        """색상 마스크에서 보드 영역을 찾습니다."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        if w < 100 or h < 100:
            return None

        size = max(w, h)
        return (x, y, size, size)

    def _find_board_by_lines(self, gray):
        """직선 검출로 보드를 찾습니다."""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return None

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

        h, w = screenshot.shape[:2]
        scale = min(1.0, 1200 / w, 800 / h)
        display = cv2.resize(screenshot, None, fx=scale, fy=scale)

        roi = cv2.selectROI("체스 보드 선택", display, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi[2] == 0 or roi[3] == 0:
            return None

        x = int(roi[0] / scale)
        y = int(roi[1] / scale)
        w = int(roi[2] / scale)
        h = int(roi[3] / scale)

        size = max(w, h)
        self.cached_rect = (x, y, size, size)
        return self.cached_rect

    def detect_orientation_from_pieces(self, pieces_8x8):
        """인식된 기물 배열을 기반으로 보드 방향을 감지합니다."""
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

        return self.orientation
