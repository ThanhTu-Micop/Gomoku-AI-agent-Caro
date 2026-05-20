# AGENTS.md

## Project

**Gomoku AI Agent** — group project (4 members), university AI course.  
Implements and compares **Minimax + Alpha-Beta Pruning** vs **Reinforcement Learning (self-play)** on a 9x9 Gomoku (Caro) board.  
Reference: [hesic73/gomoku_rl](https://github.com/hesic73/gomoku_rl)

## Tech stack

- **Language:** Python 3.x
- **Libraries:** Pygame (GUI), NumPy, pandas, PyTorch
- **Testing:** pytest

## Directory layout

```
src/
├── main.py              # entrypoint — launches GUI
├── game/
│   ├── board.py         # 9x9 board representation
│   ├── rules.py         # win/draw detection, move validation
│   └── constants.py     # BOARD_SIZE, EMPTY, X, O
├── ai/
│   ├── base.py          # abstract base agent
│   ├── minimax.py       # Minimax + Alpha-Beta
│   ├── heuristic.py     # heuristic evaluation function
│   └── rl_agent.py      # RL agent (self-play train + inference)
├── ui/
│   ├── gui.py           # Pygame main loop
│   └── renderer.py      # board rendering
├── utils/
│   ├── __init__.py
│   ├── logger.py        # match logging (win/draw, time/move by stage)
│   └── replay_buffer.py  # RL experience replay buffer (save/load)
└── scripts/
    ├── train_rl.py       # train RL agent
    └── compare.py        # pit Minimax vs RL for N matches, log results
```

## Commands

| Action | Command |
|---|---|
| Install deps | `pip install -r requirements.txt` |
| Run GUI | `python src/main.py` |
| Train RL | `python src/scripts/train_rl.py` |
| Compare agents | `python src/scripts/compare.py` |
| Run tests | `pytest tests/` |

## Code conventions

- **Style:** PEP 8 (`black` formatter, `ruff` linter)
- **Naming:** `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants
- **Typing:** type hints on all function signatures
- **Docstrings:** Google-style for public functions and classes

## Commit convention

- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code restructure
- `test:` — add/update tests
- `docs:` — documentation
- `train:` — RL training experiments

## Testing

- pytest with no special plugins
- Unit-test game logic (win conditions, move validation)
- Unit-test heuristic function (deterministic board states)
- RL agent tests are coverage-only (non-deterministic); meaningful comparison via `compare.py`
- Keep tests fast and GUI-independent

## Important notes

- Board is **9x9** (not 15x15 like the reference repo) — RL must be trained from scratch
- RL training requires GPU (CUDA) for practical speed; CPU fallback works but is slow
- Match logging records: winner, move count, avg thinking time by game stage (early / mid / end)
- Heuristic evaluation must balance offense (building patterns) and defense (blocking opponent threats)

## Google Colab Training Guide

### A. Tao Notebook

1. Vao [colab.research.google.com](https://colab.research.google.com), dang nhap.
2. **File -> New notebook**.
3. **Runtime -> Change runtime type** -> chon **Hardware accelerator: GPU** (T4/P100/L4).
4. Dat ten: `gomoku-rl-training`.

### B. Upload script

- Keo tha `colab_train.py` vao phan **Files** ben trai (hoac dung **Upload to session storage**).
- File se o duong dan `/content/colab_train.py`.

### C. Chay training

**Cell 1 — Cai thu vien:**
```python
!pip install numpy pandas torch --quiet
```

**Cell 2 — Chay lan dau:**
```python
%run /content/colab_train.py --episodes 2000 --save-every 200
```

**Cell 3 — Resume (upload `models/` tu session truoc):**
```python
%run /content/colab_train.py --episodes 2000 --save-every 200 --resume
```

### D. Tham so CLI

| Flag | Mac dinh | Mo ta |
|---|---|---|
| `--episodes` | 1000 | Tong so van self-play moi lan chay |
| `--batch-size` | 64 | So sample moi lan train |
| `--buffer-size` | 100_000 | Kich thuoc replay buffer |
| `--save-every` | 100 | Luu checkpoint sau moi N van |
| `--epsilon-start` | 0.5 | Epsilon bat dau |
| `--epsilon-end` | 0.01 | Epsilon ket thuc |
| `--model-path` | `models/rl_agent.pth` | Duong dan luu model |
| `--resume` | (off) | Tiep tuc tu checkpoint cuoi |

### E. Download ket qua

Sau khi train xong, file duoc luu trong:
- `/content/models/rl_agent.pth` — model weights
- `/content/models/rl_agent_buffer.npz` — replay buffer
- `/content/models/checkpoint.json` — metadata (episodes_done, win_counts)
- `/content/logs/replays.jsonl` — lich su van dau

Download `rl_agent.pth` ve may, dat vao `models/` trong project local.

### F. So sanh tren may local

```bash
python src/scripts/compare.py --matches 20 --rl-model models/rl_agent.pth
```

### G. Luu y

- **Session timeout:** Colab thuong gioi han theo session (thuong ~12h, tuy tai khoan). Nen dat `--episodes` phu hop.
- **Resume:** Upload `models/` (3 files: `.pth`, `_buffer.npz`, `checkpoint.json`) vao session moi, dung `--resume`.
- **Luu output:** Neu session sap reset, download checkpoint truoc khi ngat runtime.
- **Disk:** Session storage tam thoi, nen download model sau khi train.
