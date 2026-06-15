# Báo cáo Đồ án: Gomoku AI Agent

> **Môn học:** Trí tuệ nhân tạo (AI)
> **Nền tảng:** Python 3.x, Pygame, NumPy, pandas, PyTorch, pytest
> **Board:** 9x9

---

## Mục lục

1. [Tổng quan project](#1-tổng-quan-project)
2. [Cấu trúc cây thư mục](#2-cấu-trúc-cây-thư-mục)
3. [Game Engine](#3-game-engine)
4. [Thuật toán AI](#4-thuật-toán-ai)
   - 4.1 [Heuristic Evaluation](#41-heuristic-evaluation)
   - 4.2 [Minimax + Alpha-Beta Pruning](#42-minimax--alpha-beta-pruning)
   - 4.3 [RL Agent (Legacy)](#43-rl-agent-legacy)
   - 4.4 [AlphaZero (ResNet + MCTS)](#44-alphazero-resnet--mcts)
5. [Giao diện người dùng (UI)](#5-giao-diện-người-dùng-ui)
6. [Tiện ích (Utils)](#6-tiện-ích-utils)
7. [Scripts](#7-scripts)
8. [Kết quả Tests](#8-kết-quả-tests)
9. [Kết quả So sánh Agents](#9-kết-quả-so-sánh-agents)
10. [Kết quả Training RL](#10-kết-quả-training-rl)
11. [Bug Fixes](#11-bug-fixes)
12. [Kết luận](#12-kết-luận)

---

## 1. Tổng quan project

**Gomoku AI Agent** là đồ án môn học Trí tuệ nhân tạo, so sánh hai hướng tiếp cận cho game **Caro (Gomoku)** trên bàn cờ **9x9**:

1. **Minimax + Alpha-Beta Pruning** — tìm kiếm có độ sâu giới hạn với các kỹ thuật tối ưu (Zobrist hashing, Transposition Table, candidate pruning, move ordering)
2. **Reinforcement Learning (Self-Play)** — huấn luyện agent tự chơi theo phong cách AlphaZero: ResNet + Monte Carlo Tree Search (MCTS)

Project bao gồm:
- GUI bằng Pygame (chơi người-vs-người, người-vs-AI, AI-vs-AI)
- Script huấn luyện RL self-play
- Script so sánh các agents
- Bộ test tự động (47 tests) cho game logic, heuristic, agents
- Hỗ trợ training trên Google Colab qua `colab_train.py`

### Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| Python 3.x | Ngôn ngữ lập trình |
| Pygame 2.5+ | Giao diện đồ họa (GUI) |
| NumPy 1.24+ | Xử lý ma trận bàn cờ, tính toán số học |
| pandas 2.0+ | Xử lý dữ liệu log |
| PyTorch 2.0+ | Mạng neural cho RL / AlphaZero |
| pytest 7.0+ | Kiểm thử tự động |

---

## 2. Cấu trúc cây thư mục

```
Gomoku-AI-agent-main/
│
├── .gitignore                     # Quy tắc ignore cho Git
├── .ruff.toml                     # Cấu hình linter Ruff (Python 3.11, line length 100)
├── AGENTS.md                      # Hướng dẫn cho AI agents (dành cho 개발)
├── README.md                      # Hướng dẫn chính (tiếng Việt)
├── README_COLAB.md                # Hướng dẫn training trên Google Colab
├── colab_train.py                 # Script self-contained cho Colab (787 dòng)
├── requirements.txt               # Danh sách thư viện: pygame, numpy, pandas, torch, pytest
├── test_output.txt                # Kết quả test (47/47 passed)
├── test_results.txt               # Kết quả test (47/47 passed)
├── test_results_optimized.txt     # Kết quả test (47/47 passed)
│
├── docs/
│   ├── Gomoku AI Agent.docx       # Tài liệu Word
│   ├── implementation_plan.md     # Kế hoạch triển khai (601 dòng, tiếng Việt)
│   └── bao_cao.md                 # Báo cáo đồ án (file này)
│
├── logs/
│   └── matches.csv                # Kết quả 20 trận Minimax vs RL Agent
│
├── models/
│   ├── checkpoint.json            # Metadata training: 2000 episodes
│   ├── rl_agent.pth               # Model weights đã train
│   └── rl_agent_buffer.npz        # Replay buffer data
│
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point — CLI args + khởi động GUI (95 dòng)
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract class Agent (8 dòng)
│   │   ├── heuristic.py           # Heuristic evaluation function (174 dòng)
│   │   ├── minimax.py             # Minimax + Alpha-Beta Pruning (244 dòng)
│   │   ├── mcts.py                # MCTS + PUCT (206 dòng)
│   │   └── rl_agent.py            # RLAgent + AlphaZeroNet + AlphaZeroAgent (335 dòng)
│   │
│   ├── game/
│   │   ├── __init__.py
│   │   ├── constants.py           # Hằng số: BOARD_SIZE=9, EMPTY=0, X=1, O=2 (4 dòng)
│   │   ├── board.py               # Class Board — bàn cờ 9x9 (50 dòng)
│   │   └── rules.py               # Luật chơi: is_win, is_draw, is_game_over, valid_moves (85 dòng)
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── gui.py                 # Pygame game loop (113 dòng)
│   │   └── renderer.py            # Vẽ bàn cờ, quân cờ, status bar (66 dòng)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py              # Ghi log match + replay (89 dòng)
│   │   └── replay_buffer.py       # Experience replay buffer (55 dòng)
│   │
│   └── scripts/
│       ├── train_rl.py            # Self-play training loop (211 dòng)
│       └── compare.py             # So sánh agents (138 dòng)
│
└── tests/
    ├── __init__.py
    ├── test_augmentation.py       # 1 test — data augmentation 8 biến thể
    ├── test_heuristic.py          # 4 tests — heuristic evaluation
    ├── test_integration.py        # 7 tests — full game scenarios
    ├── test_logger.py             # 8 tests — match + replay logging
    ├── test_mcts.py               # 2 tests — MCTS policy + winning move
    ├── test_minimax.py            # 4 tests — minimax valid moves, blocking, winning
    ├── test_replay_buffer.py      # 4 tests — push, sample, capacity, save/load
    ├── test_rl_agent.py           # 2 tests — network shapes, valid move
    ├── test_rules.py              # 11 tests — win/draw/game_over/valid_moves
    ├── test_threats.py            # 3 tests — split four, double three threats
    └── verify_candidates.py       # 1 test (standalone) — candidate expansion distance
```

---

## 3. Game Engine

### 3.1 Constants (`src/game/constants.py`)

```python
BOARD_SIZE: int = 9
EMPTY: int = 0
X: int = 1
O: int = 2
```

Board 9x9, 3 giá trị: EMPTY (0), X (1), O (2).

### 3.2 Board (`src/game/board.py`)

Class `Board` quản lý trạng thái bàn cờ:

- `grid`: numpy array 9x9 kiểu int
- `move_count`: đếm số nước đã đi
- `reset()`: xóa bàn cờ
- `place(r, c, player)`: đặt quân (trả về False nếu ô đã có quân)
- `undo(r, c)`: undo nước đi (có guard chống corrupt move_count)
- `get_valid_moves()`: trả về list các ô trống
- `is_full()`: kiểm tra bàn cờ đầy
- `copy()`: tạo bản sao độc lập

### 3.3 Rules (`src/game/rules.py`)

Các hàm xử lý luật chơi:

- **`is_win(grid, player, last_move=None)`**: Kiểm tra thắng (5 quân liên tiếp). Hỗ trợ 2 mode:
  - **Optimized** (có `last_move`): chỉ kiểm tra 4 hướng từ ô cuối cùng
  - **Full board scan** (không có `last_move`): dùng numpy shift để kiểm tra toàn bộ board

- **`is_draw(grid)`**: Board đầy (không còn ô EMPTY)

- **`is_game_over(grid, last_move=None)`**: Trả về `(bool, winner_or_None)`

- **`valid_moves(grid)`**: Trả về list các ô trống

---

## 4. Thuật toán AI

### 4.1 Heuristic Evaluation (`src/ai/heuristic.py`)

Hàm `evaluate(grid, player)` đánh giá điểm số của bàn cờ cho một người chơi:

**Pattern scores:**

| Pattern | Score | Mô tả |
|---------|-------|-------|
| FIVE (11111) | 1,000,000 | Thắng ngay lập tức |
| FOUR_OPEN (011110) | 100,000 | 4 quân hở 2 đầu |
| FOUR_BLOCKED | 10,000 | 4 quân bị chặn 1 đầu / split four |
| THREE_OPEN | 5,000 | 3 quân hở 2 đầu |
| THREE_BLOCKED | 500 | 3 quân bị chặn |
| TWO_OPEN | 100 | 2 quân hở |
| TWO_BLOCKED | 10 | 2 quân bị chặn |

**Cơ chế hoạt động:**

1. Xây dựng chuỗi pattern từ 4 hướng (hàng, cột, chéo chính, chéo phụ) dùng lookup table
2. Đếm số pattern xuất hiện (non-overlapping)
3. Double-threat bonus: nếu có ≥2 open-three hoặc 1 open-three + 1 four
4. Center-proximity bonus: thưởng quân gần tâm bàn cờ
5. Kết quả = player_score - opponent_score + positional_bonus

### 4.2 Minimax + Alpha-Beta Pruning (`src/ai/minimax.py`)

Class `MinimaxAgent` cài đặt thuật toán Minimax với nhiều kỹ thuật tối ưu:

**Tham số:**
- `depth`: độ sâu tìm kiếm tối đa (mặc định 10)
- `time_limit`: giới hạn thời gian (mặc định 1.8 giây)

**Kỹ thuật tối ưu:**

| Kỹ thuật | Mô tả |
|----------|-------|
| **Iterative Deepening** | Tăng dần depth từ 1 đến max_depth, nếu hết thời gian thì dùng kết quả depth trước |
| **Alpha-Beta Pruning** | Cắt tỉa nhánh không cần thiết với alpha/beta |
| **Zobrist Hashing** | Hash bàn cờ 64-bit dùng bảng ngẫu nhiên, vectorized gather |
| **Transposition Table** | Cache kết quả đã tính, 3 flags: EXACT / LOWERBOUND / UPPERBOUND |
| **Candidate Move Pruning** | Chỉ xét các ô cách quân hiện tại ≤2 (distance-2 neighbors) |
| **Move Ordering** | Ưu tiên nước đi tốt nhất từ depth trước + heuristic scoring |
| **Time-limited search** | Raise TimeoutError khi hết thời gian, try/finally đảm bảo restore grid |

**Quy trình `get_move()`:**

1. Lấy candidate moves (distance-2 neighbors)
2. Vòng lặp iterative deepening:
   - Sắp xếp nước đi (best move từ depth trước được ưu tiên tối đa)
   - Với mỗi nước đi: đặt quân → gọi `_minimax()` → restore grid (try/finally)
   - Kiểm tra timeout
   - Nếu tìm được nước thắng (score ≥ 500000) → dừng sớm

**`_minimax()` chi tiết:**
1. Kiểm tra Transposition Table
2. Kiểm tra game over (win/draw)
3. Depth == 0 → gọi `evaluate()`
4. Sinh candidate moves + move ordering
5. Dùng `orig_alpha`/`orig_beta` để xác định đúng TT flag
6. Restore grid bằng `try/finally` tránh phantom stones

### 4.3 RL Agent (Legacy) (`src/ai/rl_agent.py`)

**`RLAgent`** — agent học tăng cường kiểu cũ (không dùng MCTS).

**Mạng neural:** `AlphaZeroNet` (ResNet, 5 residual blocks, 64 channels) — hoặc `GomokuNetLegacy` (CNN 3 lớp, giữ lại để so sánh).

**`get_move()`:**
1. Encode board thành 3 channels (player, opponent, empty)
2. Forward qua network → policy logits
3. Softmax + mask invalid moves
4. Sampling (deterministic: argmax, stochastic: random theo phân bố)

**`train_step()`:**
- Sample batch từ replay buffer
- Policy loss: cross-entropy với target policy
- Value loss: MSE với target value
- Optimizer: Adam + StepLR scheduler

### 4.4 AlphaZero (ResNet + MCTS) (`src/ai/rl_agent.py` + `src/ai/mcts.py`)

**`AlphaZeroAgent`** — agent AlphaZero-style với ResNet + MCTS.

#### 4.4.1 AlphaZeroNet (`src/ai/rl_agent.py`)

Kiến trúc ResNet:

```
Input (3, 9, 9)
  └─ Input Conv: Conv2D(3→64, 3x3) → BN → ReLU
       └─ 5× ResidualBlock(64):
            └─ Conv2D(64→64, 3x3) → BN → ReLU → Conv2D(64→64, 3x3) → BN → +residual → ReLU
              ├─ Policy Head: Conv2D(64→2, 1x1) → BN → ReLU → Flatten → Linear(162→81)
              └─ Value Head: Conv2D(64→1, 1x1) → BN → ReLU → Flatten → Linear(81→64) → ReLU → Linear(64→1) → Tanh
```

- Policy output: shape (81,) — xác suất từng ô
- Value output: shape (1,) — giá trị [-1, 1]

#### 4.4.2 Data Augmentation (`src/ai/rl_agent.py`)

Hàm `augment_data(state, policy)` sinh **8 biến thể đối xứng** từ 1 cặp (state, policy):

- 4 phép xoay: 0°, 90°, 180°, 270°
- Kết hợp lật ngang cho mỗi phép xoay

#### 4.4.3 MCTS — Monte Carlo Tree Search (`src/ai/mcts.py`)

**Class `MCTS`** — AlphaZero-style batch MCTS:

**PUCT Selection:**

```
PUCT(s, a) = -Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Trong đó:
- `Q(s,a)` = value_sum / visit_count (giá trị trung bình, đảo dấu cho đối thủ)
- `P(s,a)` = prior probability từ policy head
- `N(s)` = tổng lượt thăm node cha
- `c_puct` = 1.5 (hệ số exploration)

**Quy trình `search()`:**

1. **Root expansion**: gọi network để đánh giá root node
2. **Batch simulation loop** (mỗi batch = 16 simulations):
   - **SELECT**: từ root xuống lá theo PUCT (có virtual loss)
   - **EVALUATE & EXPAND**: kiểm tra terminal → gọi network batch → expand children
   - **BACKPROPAGATE**: cập nhật visit_count, value_sum ngược lên root
3. **Build policy**: từ visit counts, áp dụng temperature:
   - temperature > 0: softmax theo N^(1/T)
   - temperature = 0: argmax

**Các kỹ thuật trong MCTS:**
- **Virtual Loss** (+3 khi select, -3 khi backprop) — ngăn chọn cùng leaf trong 1 batch
- **Batch prediction** — gộp nhiều leaf states thành 1 batch forward qua network
- **Temperature schedule** — exploration_moves=12 nước đầu temperature=1.0

#### 4.4.4 AlphaZeroAgent (`src/ai/rl_agent.py`)

Kết hợp ResNet + MCTS:

```python
class AlphaZeroAgent(Agent):
    def get_move(self, grid, player, deterministic=True):
        temperature = 0.0 if deterministic else 1.0
        pi = self.mcts.search(grid, player, temperature)
        # Mask invalid moves, argmax
        return move_idx // BOARD_SIZE, move_idx % BOARD_SIZE
```

---

## 5. Giao diện người dùng (UI)

### 5.1 GUI (`src/ui/gui.py`)

Game loop Pygame với các chức năng:

- **Click chuột**: đặt quân (cho human)
- **AI move**: delay 500ms trước khi AI đi
- **Win/Draw detection**: hiển thị thông báo và chờ nhấn R để restart
- **Hỗ trợ 3 mode**: hvh, hvai, aivai

### 5.2 Renderer (`src/ui/renderer.py`)

- `CELL_SIZE = 60`, `MARGIN = 40`
- Vẽ grid, quân cờ (X=đen, O=trắng), highlight nước đi cuối
- Status bar ở đáy màn hình (cached font)

---

## 6. Tiện ích (Utils)

### 6.1 Logger (`src/utils/logger.py`)

- `log_match()`: ghi kết quả match vào `logs/matches.csv` (winner, move_count, avg_time theo stage)
- `log_game_replay()`: ghi lịch sử nước đi vào `logs/replays.jsonl`
- `load_logs()`, `load_replays()`: đọc dữ liệu

### 6.2 Replay Buffer (`src/utils/replay_buffer.py`)

- Lưu states, policies, rewards trong `deque` với dung lượng giới hạn
- `push()`, `extend()`, `sample()`, `save()` (npz), `load()` (npz)

---

## 7. Scripts

### 7.1 Train RL (`src/scripts/train_rl.py`)

Self-play training loop cho AlphaZero:

1. **Self-play game**: dùng MCTS với temperature schedule (12 exploration moves)
2. **Data augmentation** (8x)
3. **Replay buffer**: push augmented experiences
4. **Training step**: policy cross-entropy + value MSE
5. **Checkpoint**: save mỗi N episodes
6. **Early stopping**: tùy chọn patience

### 7.2 Compare Agents (`src/scripts/compare.py`)

So sánh Minimax(depth=3) vs các agent khác:

- Hỗ trợ 3 loại agent 2: minimax, rl, alphazero
- Alternating colors (mỗi match đổi bên)
- Ghi log kết quả: winner, move_count, avg_time theo stage (early/mid/end)
- Tùy chọn log game replay

---

## 8. Kết quả Tests

### Tổng quan: **47/47 tests passed** ✅

| Test file | Số tests | Mô tả |
|-----------|:--------:|-------|
| `test_rules.py` | 11 | Win detection ngang/dọc/chéo/phản chéo, draw, game_over, valid_moves |
| `test_integration.py` | 7 | Full game scenarios (win horizontal/vertical, draw, overwrite, board reset/copy) |
| `test_logger.py` | 8 | Match log (csv, draw, empty stages), replay log (jsonl, draw, numpy int types), load empty |
| `test_heuristic.py` | 4 | Evaluate empty board, winning position, blocking, open four |
| `test_minimax.py` | 4 | Valid move on empty/partial board, blocks threat, wins if possible |
| `test_replay_buffer.py` | 4 | Push+len, sample, capacity overflow, save+load |
| `test_mcts.py` | 2 | Valid policy (center bias, masks occupied), finds immediate winning move |
| `test_rl_agent.py` | 2 | AlphaZeroNet forward shapes, AlphaZeroAgent returns valid move |
| `test_threats.py` | 3 | Split four, split four center, double three threats |
| `test_augmentation.py` | 1 | 8 aligned pairs generated, piece positions match |
| `verify_candidates.py` | 1 (standalone) | Candidate expansion distance-2 |

### Chi tiết các test quan trọng:

#### test_rules.py (11 tests)
- `test_is_win_horizontal/vertical/diagonal/anti_diagonal`: Kiểm tra 5 quân liên tiếp
- `test_is_win_not_five`: 4 quân không được tính là thắng
- `test_is_draw_full_board/not_full`: Kiểm tra hòa
- `test_is_game_over_win/draw/not_over`: is_game_over() trả về đúng
- `test_valid_moves/partial`: valid_moves() đếm đúng số ô trống

#### test_minimax.py (4 tests)
- `test_minimax_blocks_immediate_threat`: X có 4 quân (0,0)-(0,3), O phải chặn ở (0,4)
- `test_minimax_wins_if_possible`: O có 4 quân (0,0)-(0,3), O phải thắng ở (0,4)

#### test_mcts.py (2 tests)
- `test_mcts_returns_valid_pi_and_masks_occupied`: Center-biased network → trung tâm có pi cao nhất; ô đã có quân bị mask
- `test_mcts_finds_immediate_winning_move`: X có 4 quân hàng 0, MCTS chọn (0,4)

#### test_threats.py (3 tests)
- `test_split_four_threat`: Pattern X.XXX (split four) cho score ≥ 10,000
- `test_split_four_center_threat`: Pattern XX.XX (split four center) cho score ≥ 10,000
- `test_double_three_threat`: 2 đường three hở cho score ≥ 80,000

---

## 9. Kết quả So sánh Agents

### Minimax(d=3) vs RL Agent — 20 matches

| Kết quả | Số trận |
|---------|:-------:|
| Minimax(d=3) thắng | **20** |
| RL Agent thắng | **0** |
| Hòa | **0** |

### Chi tiết từng match:

| Match | X | O | Winner | Số nước | Early avg(s) | Mid avg(s) |
|:-----:|---|---|--------|:-------:|:------------:|:----------:|
| 1 | Minimax | RL | Minimax | 9 | 0.6873 | 0.0000 |
| 2 | RL | Minimax | Minimax | 12 | 0.9111 | 0.0378 |
| 3 | Minimax | RL | Minimax | 11 | 0.7297 | 0.0847 |
| 4 | RL | Minimax | Minimax | 12 | 0.9102 | 0.0536 |
| 5 | Minimax | RL | Minimax | 13 | 0.5736 | 0.6453 |
| 6 | RL | Minimax | Minimax | 12 | 0.8281 | 0.0740 |
| 7 | Minimax | RL | Minimax | 11 | 0.5549 | 0.0936 |
| 8 | RL | Minimax | Minimax | 12 | 0.7371 | 0.0648 |
| 9 | Minimax | RL | Minimax | 11 | 0.5596 | 0.0963 |
| 10 | RL | Minimax | Minimax | 12 | 0.7329 | 0.0650 |
| 11 | Minimax | RL | Minimax | 11 | 0.4506 | 0.0959 |
| 12 | RL | Minimax | Minimax | 10 | 0.5648 | 0.0000 |
| 13 | Minimax | RL | Minimax | 11 | 0.2098 | 0.0923 |
| 14 | RL | Minimax | Minimax | 10 | 0.4734 | 0.0000 |
| 15 | Minimax | RL | Minimax | 11 | 0.2147 | 0.0949 |
| 16 | RL | Minimax | Minimax | 12 | 0.5670 | 0.0549 |
| 17 | Minimax | RL | Minimax | 11 | 0.3934 | 0.0978 |
| 18 | RL | Minimax | Minimax | 12 | 0.5675 | 0.0652 |
| 19 | Minimax | RL | Minimax | 11 | 0.3913 | 0.1149 |
| 20 | RL | Minimax | Minimax | 10 | 0.3992 | 0.0000 |

**Nhận xét:**
- Minimax(d=3) thắng tuyệt đối **20-0**, RL Agent không thắng được trận nào
- Hầu hết trận kết thúc ở early game (9-13 nước), RL hầu như không vào được mid-game
- Thời gian suy nghĩ trung bình của Minimax: ~0.2-0.9s (early), ~0.04-0.6s (mid)
- Nguyên nhân: RL Agent (bản legacy) không dùng MCTS, chỉ dựa trên policy network thuần, yếu hơn nhiều so với Minimax có search depth=3

---

## 10. Kết quả Training RL

### Checkpoint hiện tại (`models/checkpoint.json`)

```json
{
  "episodes_done": 2000,
  "win_counts": {"X": 1333, "O": 667, "Draw": 0},
  "mcts_sims": 80,
  "c_puct": 1.4
}
```

### Thống kê

| Tham số | Giá trị |
|---------|:-------:|
| Số episodes | 2000 |
| X thắng (đi trước) | 1333 (66.7%) |
| O thắng (đi sau) | 667 (33.3%) |
| Hòa | 0 (0%) |
| MCTS simulations | 80 |
| c_puct | 1.4 |

**Nhận xét:**
- Người đi trước (X) có lợi thế rõ rệt (tỉ lệ ~2:1)
- Không có ván hòa nào trong 2000 episodes — điều này phổ biến với MCTS self-play khi agent luôn chọn nước mạnh nhất
- Với 80 simulations, chất lượng MCTS còn hạn chế (so với 200-400 simulations thông thường)

---

## 11. Bug Fixes

Trong quá trình code review, đã phát hiện và sửa **8 bugs** (7 critical/medium, 1 low):

| # | Bug | File | Severity | Mô tả |
|:-:|-----|:----:|:--------:|-------|
| 5.1 | TT flag sai | `minimax.py` | 🔴 CRITICAL | alpha/beta bị mutate trước khi xác định flag → flag luôn là UPPERBOUND |
| 5.2 | Grid corruption khi timeout | `minimax.py` | 🔴 CRITICAL | TimeoutError raise trước khi restore grid → phantom stones |
| 5.3 | Uninitialized `winner` | `compare.py` | 🔴 CRITICAL | Nếu `get_move()` trả về None → UnboundLocalError |
| 5.4 | Status text sai vị trí | `renderer.py` | 🟠 MEDIUM | Text render trong grid thay vì status bar 40px |
| 5.5 | Magic numbers + Banker's rounding | `gui.py` | 🟡 LOW | Hardcode MARGIN/CELL_SIZE, round() sai ở biên ô |
| 5.6 | Thiếu THREE_BLOCKED patterns | `heuristic.py` | 🟠 MEDIUM | `21110`, `01112`, `1101`, `1011` bị thiếu |
| 5.7 | Undo không kiểm tra ô trống | `board.py` | 🟠 MEDIUM | `move_count` bị corrupt khi undo ô đã EMPTY |
| 5.8 | is_game_over() không dùng last_move | `rules.py` | 🟠 MEDIUM | Luôn scan full board, không dùng optimization |

---

## 12. Kết luận

### Những gì đã làm được

1. **Game Engine** hoàn chỉnh: Board 9x9, rules (win/draw), valid moves, copy/undo/reset
2. **Minimax Agent** mạnh mẽ: Iterative deepening, alpha-beta pruning, Zobrist hashing, transposition table, candidate pruning, move ordering, time-limited
3. **Heuristic Evaluation** chi tiết: Pattern scoring 5 levels, double-threat bonus, center bonus
4. **AlphaZero Agent** (ResNet + MCTS): 5 residual blocks, PUCT selection, batch evaluation, temperature schedule, data augmentation (8x)
5. **RL Training pipeline**: Self-play loop, replay buffer, augmentation, checkpointing, Colab support
6. **GUI**: Pygame, 3 modes (hvh, hvai, aivai), highlight last move, restart
7. **47/47 tests passed** — full coverage cho rules, heuristic, minimax, mcts, rl_agent, logger, replay_buffer, integration, augmentation, threats
8. **Compare script**: So sánh agents, log kết quả
9. **8 bugs fixed**: TT flag, grid corruption, uninitialized winner, UI rendering, heuristic patterns, board undo

### Hạn chế và hướng phát triển

1. **RL Agent hiện tại yếu**: Bị Minimax(d=3) đánh bại 20-0. Cần:
   - Train thêm với số episodes lớn hơn (>10000)
   - Tăng MCTS simulations (200-400)
   - Dùng `AlphaZeroAgent` thay vì `RLAgent` trong `compare.py` (hiện tại compare.py dùng `RLAgent` kế thừa từ code cũ)

2. **Chưa có so sánh AlphaZeroAgent thực tế**: Mặc dù đã cài đặt `AlphaZeroAgent` + `MCTS`, script `compare.py` mặc định dùng `RLAgent`. Cần chạy so sánh với `--agent-type alphazero`.

3. **Draw rate = 0**: 2000 self-play episodes không có ván hòa nào — có thể do exploration chưa đủ hoặc cần điều chỉnh temperature schedule.

4. **Chưa có validation vs random/weak opponent**: Training hiện tại không có đối thủ validation để đánh giá progress.

5. **Thiếu model fine-tuned cho 9x9**: Model hiện tại train với 80 sims, cần train thêm với config mạnh hơn.

6. **Log file handle leak**: `train_rl.py` mở file log ở đầu và không đóng đúng cách nếu có exception (minor).

### Tổng kết

Project đã cài đặt thành công 2 approaches cho Gomoku 9x9:
- **Minimax + Alpha-Beta**: hoạt động tốt, thắng RL Agent hiện tại
- **AlphaZero (ResNet + MCTS)**: đã cài đặt đầy đủ, cần train thêm để đạt sức mạnh tương đương Minimax

Kiến trúc module rõ ràng, test coverage cao (47 tests), code quality tốt (8 bugs đã fix trong code review), hỗ trợ Colab training và GUI tương tác.
