# ============================================================
# Google Colab: Gomoku RL Agent Training
# ============================================================
# Runtime: GPU (T4/P100/L4)
# ============================================================
# Cell 1: Install deps
# !pip install numpy pandas torch --quiet
#
# Cell 2: Run training (first time)
# %run /content/colab_train.py --episodes 2000 --save-every 200
#
# Cell 3: Resume training (upload models/ from previous session first)
# %run /content/colab_train.py --episodes 2000 --save-every 200 --resume

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ============================================================
# Constants
# ============================================================
BOARD_SIZE: int = 9
EMPTY: int = 0
X: int = 1
O: int = 2

# ============================================================
# Board
# ============================================================
class Board:
    def __init__(self) -> None:
        self.grid: np.ndarray = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
        self.move_count: int = 0

    def reset(self) -> None:
        self.grid.fill(EMPTY)
        self.move_count = 0

    def is_empty(self, row: int, col: int) -> bool:
        return self.grid[row, col] == EMPTY

    def place(self, row: int, col: int, player: int) -> bool:
        if not self.is_empty(row, col):
            return False
        self.grid[row, col] = player
        self.move_count += 1
        return True

    def get_valid_moves(self) -> list[tuple[int, int]]:
        return list(zip(*np.where(self.grid == EMPTY)))

# ============================================================
# Rules
# ============================================================
def check_direction(grid: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> bool:
    for k in range(5):
        r, c = row + dr * k, col + dc * k
        if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE:
            return False
        if grid[r, c] != player:
            return False
    return True

def is_win(grid: np.ndarray, player: int, last_move: tuple[int, int] | None = None) -> bool:
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    if last_move is not None:
        r_last, c_last = last_move
        for dr, dc in directions:
            count = 1
            r, c = r_last + dr, c_last + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
                count += 1
                r += dr
                c += dc
            r, c = r_last - dr, c_last - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                return True
        return False
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] == player:
                for dr, dc in directions:
                    count = 1
                    for k in range(1, 5):
                        nr, nc = r + dr * k, c + dc * k
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and grid[nr, nc] == player:
                            count += 1
                        else:
                            break
                    if count >= 5:
                        return True
    return False

def is_draw(grid: np.ndarray) -> bool:
    return not np.any(grid == EMPTY)

# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.states: deque[np.ndarray] = deque(maxlen=capacity)
        self.policies: deque[np.ndarray] = deque(maxlen=capacity)
        self.rewards: deque[float] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, policy: np.ndarray, reward: float) -> None:
        self.states.append(state)
        self.policies.append(policy)
        self.rewards.append(reward)

    def extend(
        self,
        states: list[np.ndarray],
        policies: list[np.ndarray],
        rewards: list[float],
    ) -> None:
        for s, p, r in zip(states, policies, rewards):
            self.push(s, p, r)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.states) < batch_size:
            batch_size = len(self.states)
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        states = np.array([self.states[i] for i in indices])
        policies = np.array([self.policies[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        return states, policies, rewards

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez_compressed(
            path,
            states=np.array(self.states),
            policies=np.array(self.policies),
            rewards=np.array(self.rewards),
        )

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.states = deque(data["states"], maxlen=self.states.maxlen)
        self.policies = deque(data["policies"], maxlen=self.policies.maxlen)
        self.rewards = deque(data["rewards"], maxlen=self.rewards.maxlen)

# ============================================================
# Neural Network
# ============================================================
class GomokuNet(nn.Module):
    def __init__(self, board_size: int = BOARD_SIZE) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.policy_head = nn.Linear(64 * board_size * board_size, board_size * board_size)
        self.value_head = nn.Linear(64 * board_size * board_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x_flat = x.view(x.size(0), -1)
        policy = self.policy_head(x_flat)
        value = torch.tanh(self.value_head(x_flat))
        return policy, value

def encode_board(grid: np.ndarray, player: int) -> np.ndarray:
    state = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    state[0] = (grid == player).astype(np.float32)
    state[1] = (grid == (O if player == X else X)).astype(np.float32)
    state[2] = (grid == EMPTY).astype(np.float32)
    return state

# ============================================================
# RL Agent
# ============================================================
class RLAgent:
    def __init__(self, device: str = "cpu", buffer_capacity: int = 100_000) -> None:
        self.device = device
        self.network = GomokuNet().to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def get_move(self, grid: np.ndarray, player: int, deterministic: bool = False) -> tuple[int, int]:
        state = encode_board(grid, player)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, _ = self.network(state_tensor)
        policy = policy.squeeze(0).cpu().numpy()
        valid = np.where(grid.flatten() == EMPTY)[0]
        if len(valid) == 0:
            return BOARD_SIZE // 2, BOARD_SIZE // 2
        policy_flat = policy.flatten()
        policy_flat[~np.isin(np.arange(len(policy_flat)), valid)] = -float("inf")
        probs = np.exp(policy_flat - np.max(policy_flat))
        probs = probs / probs.sum()
        if deterministic:
            move_idx = int(np.argmax(probs))
        else:
            move_idx = int(np.random.choice(len(probs), p=probs))
        return move_idx // BOARD_SIZE, move_idx % BOARD_SIZE

    def record_experience(
        self,
        states: list[np.ndarray],
        policies: list[np.ndarray],
        rewards: list[float],
    ) -> None:
        self.buffer.extend(states, policies, rewards)

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.buffer) < batch_size:
            return 0.0
        states, policies, rewards = self.buffer.sample(batch_size)
        state_tensors = torch.tensor(states, dtype=torch.float32).to(self.device)
        policy_targets = torch.tensor(policies, dtype=torch.float32).to(self.device)
        value_targets = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        policy_pred, value_pred = self.network(state_tensors)
        policy_loss = -torch.sum(policy_targets * torch.log_softmax(policy_pred, dim=1), dim=1).mean()
        value_loss = nn.MSELoss()(value_pred, value_targets)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.network.eval()

    def save_buffer(self, path: str) -> None:
        self.buffer.save(path)

    def load_buffer(self, path: str) -> None:
        self.buffer.load(path)

# ============================================================
# Self-Play Training
# ============================================================
def play_self_play_game(agent: RLAgent, epsilon: float = 0.1) -> tuple[
    list[np.ndarray], list[np.ndarray], list[float], int | None, list[tuple[int, int, int]]
]:
    board = Board()
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    moves: list[tuple[int, int, int]] = []
    current_player = X

    while True:
        state = encode_board(board.grid, current_player)
        states.append(state)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            policy, _ = agent.network(state_tensor)
        policy = policy.squeeze(0).cpu().numpy()

        valid_moves = board.get_valid_moves()
        if not valid_moves:
            break

        policy_flat = policy.flatten()
        valid_indices = [r * BOARD_SIZE + c for r, c in valid_moves]
        policy_flat[~np.isin(np.arange(len(policy_flat)), valid_indices)] = -float("inf")

        if np.random.random() < epsilon:
            move_idx = np.random.choice(valid_indices)
        else:
            probs = np.exp(policy_flat - np.max(policy_flat))
            probs = probs / probs.sum()
            move_idx = np.random.choice(len(probs), p=probs)

        r, c = move_idx // BOARD_SIZE, move_idx % BOARD_SIZE
        board.place(r, c, current_player)
        moves.append((current_player, r, c))

        policy_target = np.zeros(BOARD_SIZE * BOARD_SIZE)
        policy_target[move_idx] = 1.0
        policies.append(policy_target)

        if is_win(board.grid, current_player, last_move=(r, c)):
            winner = current_player
            break
        elif is_draw(board.grid):
            winner = None
            break

        current_player = O if current_player == X else X

    rewards: list[float] = []
    for i in range(len(states)):
        turn = X if i % 2 == 0 else O
        if winner is None:
            rewards.append(0.0)
        elif winner == turn:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)

    return states, policies, rewards, winner, moves

def log_game_replay(log_path: str, match_id: int, moves: list[tuple[int, int, int]], winner: int | None) -> None:
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
    record = {
        "match_id": int(match_id),
        "moves": [{"player": int(p), "row": int(r), "col": int(c)} for p, r, c in moves],
        "winner": "X" if winner == X else ("O" if winner == O else "Draw"),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

def validate_agent(agent: RLAgent, num_matches: int = 10) -> float:
    win = 0
    for _ in range(num_matches):
        board = Board()
        current_player = X
        while True:
            if current_player == X:
                move = agent.get_move(board.grid, current_player, deterministic=True)
            else:
                valid = board.get_valid_moves()
                if not valid:
                    break
                move = valid[np.random.randint(len(valid))]
            r, c = move
            board.place(r, c, current_player)
            if is_win(board.grid, current_player, last_move=(r, c)):
                if current_player == X:
                    win += 1
                break
            elif is_draw(board.grid):
                break
            current_player = O if current_player == X else X
    return win / num_matches

# ============================================================
# Main
# ============================================================
MAX_EPISODES = 5000


def main() -> None:
    parser = argparse.ArgumentParser(description="Gomoku RL Agent Training (self-play)")
    parser.add_argument("--episodes", type=int, default=1000, help="Tong so van self-play (toi da 5000)")
    parser.add_argument("--batch-size", type=int, default=64, help="So sample moi lan train")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Kich thuoc replay buffer")
    parser.add_argument("--save-every", type=int, default=100, help="Luu checkpoint sau moi N van")
    parser.add_argument("--epsilon-start", type=float, default=0.5, help="Epsilon bat dau")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Epsilon ket thuc")
    parser.add_argument("--model-path", default="models/rl_agent.pth", help="Duong dan luu model")
    parser.add_argument("--resume", action="store_true", help="Tiep tuc tu checkpoint cuoi")
    args = parser.parse_args()

    if args.episodes > MAX_EPISODES:
        print(f"--episodes vuot gioi han, tu dong gioi han ve {MAX_EPISODES} (ban yeu cau {args.episodes}).")
        args.episodes = MAX_EPISODES

    model_path = args.model_path
    meta_path = os.path.join(os.path.dirname(model_path) or ".", "checkpoint.json")
    buffer_path = model_path.replace(".pth", "_buffer.npz")
    replay_log = "logs/replays.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_episode = 1
    win_counts: dict[str, int] = {"X": 0, "O": 0, "Draw": 0}

    if args.resume and os.path.exists(model_path) and os.path.exists(meta_path):
        agent = RLAgent(device=device, buffer_capacity=args.buffer_size)
        agent.load(model_path)
        agent.load_buffer(buffer_path)
        meta = json.load(open(meta_path))
        start_episode = meta["episodes_done"] + 1
        win_counts = meta.get("win_counts", {"X": 0, "O": 0, "Draw": 0})
        print(f"Resumed from episode {meta['episodes_done']}")
    else:
        agent = RLAgent(device=device, buffer_capacity=args.buffer_size)

    total_target = start_episode - 1 + args.episodes
    print(f"Device: {device}")
    print(f"Episodes: {start_episode}-{total_target} ({args.episodes} van moi)")
    print(f"Batch: {args.batch_size}, Buffer: {args.buffer_size}")
    print()

    losses: list[float] = []

    for episode in range(start_episode, total_target + 1):
        progress = (episode - start_episode + 1) / args.episodes
        epsilon = max(args.epsilon_end, args.epsilon_start * (1 - progress))
        states, policies, rewards, winner, moves = play_self_play_game(agent, epsilon=epsilon)
        agent.record_experience(states, policies, rewards)

        winner_name = "X" if winner == X else ("O" if winner == O else "Draw")
        win_counts[winner_name] = win_counts.get(winner_name, 0) + 1

        log_game_replay(replay_log, episode, moves, winner)

        loss = 0.0
        if len(agent.buffer) >= args.batch_size:
            loss = agent.train_step(batch_size=args.batch_size)
            losses.append(loss)

        if episode % args.save_every == 0 or episode == start_episode:
            avg_loss = sum(losses[-100:]) / min(len(losses), 100) if losses else 0
            print(f"Episode {episode:5d}/{total_target} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Buffer: {len(agent.buffer):6d} | "
                  f"X: {win_counts['X']:3d} | "
                  f"O: {win_counts['O']:3d} | "
                  f"Draw: {win_counts['Draw']:3d}")

        if episode % args.save_every == 0:
            agent.save(model_path)
            agent.save_buffer(buffer_path)
            with open(meta_path, "w") as f:
                json.dump({"episodes_done": episode, "win_counts": win_counts}, f)
            win_rate = validate_agent(agent)
            print(f"  -> Saved checkpoint | Validation vs random: {win_rate*100:.0f}%")

    agent.save(model_path)
    agent.save_buffer(buffer_path)
    with open(meta_path, "w") as f:
        json.dump({"episodes_done": total_target, "win_counts": win_counts}, f)
    print(f"\nTraining complete. Model saved to {model_path}")
    print(f"Final: X={win_counts['X']}, O={win_counts['O']}, Draw={win_counts['Draw']}")

if __name__ == "__main__":
    main()
