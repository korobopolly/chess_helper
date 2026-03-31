"""설정 상수 및 유틸리티 함수."""

import json
import os
import shutil
from pathlib import Path

import cv2
import mss
import numpy as np

SCRIPT_DIR = Path(__file__).parent
TEMPLATES_DIR = SCRIPT_DIR / "templates"
CONFIG_FILE = SCRIPT_DIR / "config.json"

SQUARE_SIZE = 64  # 템플릿 정규화 크기
MATCH_THRESHOLD = 0.15  # 템플릿 매칭 신뢰도 임계값

PIECE_NAMES = {
    "wK": "K", "wQ": "Q", "wR": "R", "wB": "B", "wN": "N", "wP": "P",
    "bK": "k", "bQ": "q", "bR": "r", "bB": "b", "bN": "n", "bP": "p",
}

STOCKFISH_PATHS = [
    r"C:\stockfish\stockfish-windows-x86-64-avx2.exe",
    r"C:\stockfish\stockfish.exe",
    r"C:\Program Files\Stockfish\stockfish.exe",
    r"C:\Program Files (x86)\Stockfish\stockfish.exe",
    r"stockfish.exe",
    r"stockfish",
]


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
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame


def find_stockfish(user_path=None):
    """Stockfish 실행 파일을 찾습니다."""
    if user_path and os.path.isfile(user_path):
        return user_path
    for p in STOCKFISH_PATHS:
        if os.path.isfile(p):
            return p
    found = shutil.which("stockfish")
    if found:
        return found
    return None
