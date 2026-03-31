# Chess Vision

화면에서 체스 보드를 인식하고 Stockfish로 최선의 수를 찾아주는 프로그램.

## 요구사항

- 기물 : NEO
- 보드 : Chess The Musical

- Windows Python 3.10+ (WSL이 아닌 Windows에서 실행)
- Stockfish 엔진 ([다운로드](https://stockfishchess.org/download/))

## 설치

```bash
pip install -r requirements.txt
```

Stockfish를 `C:\stockfish\` 에 배치하거나 `--stockfish-path`로 경로 지정.

## 사용법

### 1. 초기 보정 (최초 1회)

체스 사이트에서 **시작 위치** 보드를 화면에 띄운 상태로:

```bash
python chess_vision.py --calibrate
```

- 보드 영역을 드래그로 선택 → 기물 템플릿 자동 추출
- 보드 테마가 바뀌면 재보정 필요

### 2. 분석 모드

```bash
python chess_vision.py                         # 수동 (F9로 분석)
python chess_vision.py --watch                 # 자동 감시 (보드 변화시 자동 분석)
python chess_vision.py --watch --auto-play     # 완전 자동 (감시 + 자동 클릭)
python chess_vision.py --turn b                # 흑 차례부터
python chess_vision.py --depth 22              # 분석 깊이 조정
python chess_vision.py --watch-interval 2.0    # 감시 간격 조절 (기본 1초)
python chess_vision.py --manual-select         # 수동 보드 선택
python chess_vision.py --no-engine             # 엔진 없이 FEN만 출력
```

### 핫키

| 키    | 기능                       |
| ----- | -------------------------- |
| `F9`  | 화면 분석 → 최선의 수 표시 |
| `F8`  | 자동 감시 켜기/끄기        |
| `F7`  | 차례 전환 (백↔흑)          |
| `F6`  | 보드 위치 재설정           |
| `F10` | 종료                       |

## 동작 원리

1. **화면 캡처** - mss로 모니터 캡처
2. **보드 감지** - HSV 색상 분석 또는 수동 선택
3. **기물 인식** - 배경색 정규화 후 형태 템플릿 매칭 + 색상 판별
4. **FEN 생성** - 8x8 배열을 FEN 문자열로 변환
5. **Stockfish 분석** - 상위 3개 수 + 평가점수
6. **결과 표시** - 화면 오버레이 (빨간 화살표) + 콘솔 출력

## 지원 사이트

chess.com, lichess.org 등 대부분의 웹/데스크톱 체스 보드에서 동작.
보드 테마가 바뀌면 `--calibrate`로 재보정.
