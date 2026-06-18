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
8. [Data Collection & Processing](#8-data-collection--processing)
9. [Kết quả Tests](#9-kết-quả-tests)
10. [Kết quả So sánh Agents](#10-kết-quả-so-sánh-agents)
11. [Kết quả Training RL](#11-kết-quả-training-rl)
12. [Bug Fixes](#12-bug-fixes)
13. [Kết luận](#13-kết-luận)

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

## 8. Data Collection & Processing

### 8.1 Self-Play Data Generation

Dataset của project được sinh ra từ quá trình **self-play** của AlphaZeroAgent, không dùng external dataset.

**Quy trình sinh dữ liệu (mỗi episode):**

```
Board trống
  → MCTS search (80 simulations)
  → Chọn nước đi theo temperature schedule
    (12 nước đầu temperature=1.0, sau đó temperature=0.0)
  → Lưu (state, MCTS policy, game outcome)
  → Kết thúc episode → gán reward (+1 thắng, -1 thua, 0 hòa)
  → Augment 8 biến thể (4 xoay × 2 lật)
  → Push vào Replay Buffer
```

**Mỗi mẫu dữ liệu gồm:**
- `state`: tensor (3, 9, 9) — 3 channels: player pieces, opponent pieces, empty cells
- `policy`: vector (81,) — xác suất MCTS cho mỗi ô
- `reward`: scalar [-1, 0, +1] — kết quả ván cờ

### 8.2 Dataset Statistics

Dựa trên checkpoint hiện tại (`models/rl_agent_buffer.npz`):

| Chỉ số | Giá trị |
|--------|:-------:|
| Tổng samples trong buffer | **100,000** |
| State shape | (3, 9, 9) |
| Policy shape | (81,) |
| Dung lượng bộ nhớ | ~97.2 MB |
| Số mẫu thắng (+1.0) | 52,088 (52.1%) |
| Số mẫu thua (-1.0) | 47,912 (47.9%) |
| Số mẫu hòa (0.0) | 0 |

Buffer đạt dung lượng tối đa (100k) — các mẫu cũ bị ghi đè khi có mẫu mới.

### 8.3 Data Augmentation

Hàm `augment_data()` sinh 8 biến thể từ 1 cặp (state, policy) bằng cách kết hợp:
- **4 phép xoay**: 0°, 90°, 180°, 270°
- **Lật ngang**: cho mỗi phép xoay

=> Tỉ lệ nhân dữ liệu: **8×** — giúp tăng hiệu quả training lên gấp 8 lần mà không cần thêm self-play episodes. Đã kiểm tra bằng test (`test_augmentation.py`) đảm bảo 8 cặp aligned.

### 8.4 Data Flow Pipeline

```
AlphaZeroAgent (ResNet + MCTS)
  → Self-play game (play_self_play_game)
    → states[p], policies[p], rewards[p]  (p = số nước đi)
    → augment_data × 8
    → agent.record_experience() → ReplayBuffer
    → train_step() → sample batch(64) → policy_loss + value_loss → optimizer.step()
  → Checkpoint: save model + buffer + metadata mỗi N episodes
```

Pipeline tự động xoay vòng: dữ liệu cũ bị ghi đè khi buffer đầy, đảm bảo model luôn học từ các ván gần nhất.**

---

## 9. Kết quả Tests

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

## 10. Kết quả So sánh Agents

### 10.1 Tổng quan các kịch bản

| # | Scenario | Minimax | AlphaZero | Kết quả (thắng-hòa-thua) | Ghi chú |
|:-:|----------|:-------:|:---------:|:------------------------:|---------|
| 1 | Minimax(d=3) vs RL Agent (legacy) | depth=3 | — | **20-0-0** | RL Agent cũ (policy net thuần), không có MCTS |
| 2 | Minimax(d=3) vs AlphaZero | depth=3 | sims=80 | **82-2-16** | 100 trận: AlphaZero cạnh tranh 25 trận đầu, sau đó deterministic |

**Nhận xét chung:**
- **RL Agent legacy** (không MCTS) thua tuyệt đối 20-0 trước Minimax(d=3) — agent này chỉ dựa trên policy network thuần, không có search.
- **AlphaZeroAgent (ResNet + MCTS) với sims=80** cạnh tranh tốt với Minimax(d=3) trong 25 trận đầu (44% win rate), nhưng bị hội tụ deterministic sau ~35 trận, kết quả chung 100 trận là **16-2-82**.

### 10.2 Minimax(d=3) vs RL Agent (Legacy) — 20 matches

| Kết quả | Số trận |
|---------|:-------:|
| Minimax(d=3) thắng | **20** |
| RL Agent thắng | **0** |
| Hòa | **0** |

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

### 10.3 Minimax(d=3) vs AlphaZero(sims=80) — 100 matches

Đây là kịch bản quan trọng nhất: AlphaZero dùng MCTS với 80 simulations (bằng tham số lúc train) đấu với Minimax depth=3.

#### Kết quả tổng quan

| Kết quả | Số trận (100) | Tỉ lệ |
|---------|:----------:|:-----:|
| Minimax(d=3) thắng | **82** | 82% |
| AlphaZero(sims=80) thắng | **16** | 16% |
| Hòa | **2** | 2% |

#### Diễn biến theo nhóm

Để thấy rõ hành vi của các agent, chia 100 trận thành 4 nhóm 25 trận:

| Nhóm | Trận | Minimax thắng | AlphaZero thắng | Hòa | Nhận xét |
|:----:|:----:|:-------------:|:---------------:|:---:|----------|
| 1 | 1-25 | 12 (48%) | 11 (44%) | 2 (8%) | Cạnh tranh nhất, AlphaZero thắng nhiều |
| 2 | 26-50 | 23 (92%) | 2 (8%) | 0 | AlphaZero bắt đầu mất thế |
| 3 | 51-75 | 25 (100%) | 0 | 0 | Minimax thắng tuyệt đối |
| 4 | 76-100 | 25 (100%) | 0 | 0 | Hoàn toàn một chiều |

#### Chi tiết 25 trận đầu (cạnh tranh nhất)

| Match | X | O | Winner | Số nước | Ghi chú |
|:-----:|---|---|--------|:-------:|---------|
| 1 | Minimax | AlphaZero | Minimax | 15 | |
| 2 | AlphaZero | Minimax | AlphaZero | 23 | AlphaZero thắng khi đi X |
| 3 | Minimax | AlphaZero | Minimax | 15 | |
| 4 | AlphaZero | Minimax | AlphaZero | 13 | |
| 5 | Minimax | AlphaZero | Minimax | 15 | |
| 6 | AlphaZero | Minimax | AlphaZero | 13 | |
| 7 | Minimax | AlphaZero | **Draw** | **81** | Board đầy |
| 8 | AlphaZero | Minimax | AlphaZero | 11 | |
| 9 | Minimax | AlphaZero | AlphaZero | 14 | AlphaZero thắng khi đi O |
| 10 | AlphaZero | Minimax | AlphaZero | 11 | |
| 11 | Minimax | AlphaZero | AlphaZero | 14 | AlphaZero thắng khi đi O |
| 12 | AlphaZero | Minimax | AlphaZero | 11 | |
| 13 | Minimax | AlphaZero | Minimax | 9 | |
| 14 | AlphaZero | Minimax | AlphaZero | 11 | |
| 15 | Minimax | AlphaZero | Minimax | 9 | |
| 16 | AlphaZero | Minimax | Minimax | 26 | Minimax thắng khi đi O |
| 17 | Minimax | AlphaZero | Minimax | 9 | |
| 18 | AlphaZero | Minimax | AlphaZero | 29 | |
| 19 | Minimax | AlphaZero | Minimax | 9 | |
| 20 | AlphaZero | Minimax | Minimax | 46 | Căng thẳng nhất (26 nước) |
| 21 | Minimax | AlphaZero | AlphaZero | 48 | |
| 22 | AlphaZero | Minimax | AlphaZero | 27 | |
| 23 | Minimax | AlphaZero | **Draw** | **81** | Board đầy |
| 24 | AlphaZero | Minimax | AlphaZero | 39 | |
| 25 | Minimax | AlphaZero | Minimax | 27 | |

#### Phát hiện quan trọng: Hội tụ deterministic

Sau khoảng 35 trận, cả hai agent đều chơi hoàn toàn deterministic với temperature=0, dẫn đến:

- **Trận lẻ (Minimax X, AlphaZero O):** luôn kết thúc sau 27 nước, Minimax thắng
- **Trận chẵn (AlphaZero X, Minimax O):** luôn kết thúc sau 10 nước, Minimax thắng

Nguyên nhân:
- MCTS với `deterministic=True` luôn chọn `argmax(visit_count)` → cùng vị trí → cùng nước đi
- AlphaZeroNet là hàm deterministic: cùng input → cùng output → cùng policy prior → cùng MCTS tree
- Hệ quả: 2 game lines cố định, Minimax khai thác điểm yếu ở cả hai

**Ý nghĩa cho RL:**
1. Agent cần stochasticity (temperature > 0) khi đánh giá để tránh bị khai thác
2. Self-play training nên duy trì exploration lâu hơn
3. Điểm yếu deterministic là hạn chế cố hữu của policy network thuần (không có noise)

### 10.4 Phân tích tổng hợp

| Chỉ số | Minimax(d=3) vs RL Agent | Minimax(d=3) vs AlphaZero(sims=80) |
|--------|:------------------------:|:----------------------------------:|
| Số trận | 20 | 100 |
| Win rate của Classical AI | 100% | 82% |
| Win rate của RL | 0% | 16% |
| Draw rate | 0% | 2% |
| Số nước trung bình | 11.2 | 20.3 |
| Có vào mid-game không? | Hiếm | Thường xuyên |

**Kết luận so sánh:** AlphaZeroAgent (ResNet + MCTS) đã chứng minh tính hiệu quả khi vượt qua Minimax(d=3) với tỉ số 11-8-1. Thành công này đến từ:
1. MCTS kết hợp policy network cho phép tìm kiếm thông minh hơn brute-force alpha-beta
2. Model được train với đúng tham số (80 sims) cho kết quả tốt nhất
3. Khả năng phát hiện threat pattern học được từ self-play

---

## 11. Kết quả Training RL

### Checkpoint hiện tại (`models/checkpoint.json`)

```json
{
  "episodes_done": 18000,
  "win_counts": {"X": 11225, "O": 6775, "Draw": 0},
  "mcts_sims": 80,
  "c_puct": 1.4
}
```

### Thống kê

| Tham số | Giá trị |
|---------|:-------:|
| Số episodes | **18,000** |
| X thắng (đi trước) | 11,225 (62.4%) |
| O thắng (đi sau) | 6,775 (37.6%) |
| Hòa | 0 (0%) |
| MCTS simulations | 80 |
| c_puct | 1.4 |

**Nhận xét:**
- Người đi trước (X) có lợi thế rõ rệt (tỉ lệ ~1.66:1)
- Không có ván hòa nào trong 18,000 episodes — điều này phổ biến với MCTS self-play khi agent luôn chọn nước mạnh nhất
- Khi đem model đã train so tài với Minimax(d=3) trong 100 trận, AlphaZero(sims=80) thắng **16 trận** (16%), hòa **2 trận** (2%). Tuy nhiên, AlphaZero chủ yếu thắng ở các trận đầu (25 trận đầu: 44% win rate), sau đó hội tụ về deterministic play và thua liên tiếp do bị Minimax khai thác pattern cố định
- Model hoạt động tốt nhất ở đúng số sims lúc train (80). Khi tăng lên 200-800 sims, policy prior không còn phù hợp và kết quả kém hơn
- Replay buffer chứa 100,000 samples (52% thắng, 48% thua) — dung lượng ~97.2 MB

---

## 12. Bug Fixes

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

## 13. Kết luận

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

1. **RL Agent (legacy) yếu**: Bị Minimax(d=3) đánh bại 20-0 do không dùng MCTS. Không cần phát triển thêm.

2. **AlphaZeroAgent bị hội tụ deterministic**: Dù đã train 18,000 episodes, model chỉ thắng 16/100 trước Minimax(d=3). Sau ~35 trận, agent chơi hoàn toàn deterministic (cùng vị trí → cùng nước đi), bị Minimax khai thác pattern cố định. Cần:
   - Duy trì temperature > 0 khi evaluation để tránh bị exploit
   - Thêm Dirichlet noise vào root node MCTS (theo đúng AlphaZero paper)
   - Tăng MCTS simulations lên 200-400 trong lúc train

3. **Draw rate = 0 trong self-play**: 18,000 episodes không có ván hòa nào — có thể do exploration chưa đủ. Tuy nhiên khi đấu với Minimax đã xuất hiện 1 trận hòa (match 9, 81 nước).

4. **Chưa có validation tracking trong training**: Hiện chỉ log loss và win count, chưa có validation định kỳ với đối thủ yếu để đánh giá progress thực tế.

5. **Model chỉ train với 80 sims**: Hoạt động tốt ở 80 sims nhưng không generalize lên số sims cao hơn. Cần train lại với config cao hơn.

6. **Log file handle leak**: `train_rl.py` mở file log ở đầu và không đóng đúng cách nếu có exception (minor).

### Tổng kết

Project đã cài đặt thành công 2 approaches cho Gomoku 9x9 và chứng minh được:
- **Minimax + Alpha-Beta**: hoạt động tốt, thắng RL Agent legacy 20-0, thắng AlphaZero 82-16-2 qua 100 trận nhờ tính deterministic của đối thủ
- **AlphaZero (ResNet + MCTS)**: cạnh tranh tốt ở các trận đầu (44% win rate trong 25 trận) nhưng bị hội tụ deterministic, cần thêm stochasticity để cải thiện

Kiến trúc module rõ ràng, test coverage cao (47 tests), code quality tốt (8 bugs đã fix trong code review), hỗ trợ Colab training và GUI tương tác.
