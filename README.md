# Gomoku AI Agent

Project mon hoc AI so sanh hai huong tiep can cho Gomoku 9x9:

- `Minimax + Alpha-Beta Pruning`
- `Reinforcement Learning (self-play)`

Project co GUI bang Pygame, script train RL, script compare hai agent, va bo test cho game logic.

## Tech Stack

- Python 3.x
- Pygame
- NumPy
- pandas
- PyTorch
- pytest

## Cau truc thu muc

```text
src/
  main.py
  game/
  ai/
  ui/
  utils/
  scripts/
tests/
colab_train.py
README_COLAB.md
```

## Cai dat

```bash
pip install -r requirements.txt
```

## Cach chay

Neu ban gap loi import `No module named 'src'` tren mot so moi truong, hay chay theo module mode:

```bash
python -m src.main
python -m src.scripts.train_rl
python -m src.scripts.compare --matches 20
```

Lenh dang tai lieu ben duoi (`python src/...`) van duoc ho tro.

### 1. Chay GUI

```bash
python src/main.py
```

Hoac:

```bash
python -m src.main
```

### 2. Cac mode cho GUI

Human vs Human:

```bash
python src/main.py --mode hvh
```

Human vs AI Minimax:

```bash
python src/main.py --mode hvai --depth 3
```

AI vs AI:

```bash
python src/main.py --mode aivai --depth 3
```

AI vs AI voi RL model:

```bash
python src/main.py --mode aivai --depth 3 --rl-model models/rl_agent.pth
```

## Train RL local

```bash
python src/scripts/train_rl.py
```

Hoac:

```bash
python -m src.scripts.train_rl
```

Luu y: train local can PyTorch va GPU neu muon toc do thuc te.

## Compare agent

```bash
python src/scripts/compare.py --matches 20 --rl-model models/rl_agent.pth
```

Hoac:

```bash
python -m src.scripts.compare --matches 20 --rl-model models/rl_agent.pth
```

## Chay test

```bash
pytest tests/
```

## Train tren Google Colab

De train RL tren Google Colab, xem huong dan chi tiet tai:

- `README_COLAB.md`

No bao gom:

- cach tao notebook Colab
- upload `colab_train.py`
- chay training lan dau
- resume tu checkpoint voi `--resume`
- download model ve local

## Ghi chu

- Board la `9x9`, khong phai `15x15`.
- RL can train moi tu dau cho board nay.
- Test hien tai tap trung vao game logic, heuristic, minimax, logger, replay buffer, va integration co ban.
