import argparse
import json
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Google Colab: AlphaZero-style Gomoku Training (9x9)
# ============================================================
# Cell 1:
# !pip install numpy pandas torch --quiet
#
# Cell 2 (fresh run):
# %run /content/colab_train.py --episodes 2000 --save-every 200
#
# Cell 3 (resume):
# %run /content/colab_train.py --episodes 2000 --save-every 200 --resume


# ============================================================
# Constants
# ============================================================
BOARD_SIZE: int = 9
EMPTY: int = 0
X: int = 1
O: int = 2
MAX_EPISODES: int = 5000


def other_player(player: int) -> int:
    return O if player == X else X


# ============================================================
# Board + Rules
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

    # Fallback to full board check (highly optimized using numpy)
    mask = (grid == player)
    for dr, dc in directions:
        win_mask = mask.copy()
        for k in range(1, 5):
            shifted = np.zeros_like(mask)
            r_start, r_end = max(0, dr * k), min(BOARD_SIZE, BOARD_SIZE + dr * k)
            c_start, c_end = max(0, dc * k), min(BOARD_SIZE, BOARD_SIZE + dc * k)
            sr_start, sr_end = max(0, -dr * k), min(BOARD_SIZE, BOARD_SIZE - dr * k)
            sc_start, sc_end = max(0, -dc * k), min(BOARD_SIZE, BOARD_SIZE - dc * k)
            shifted[r_start:r_end, c_start:c_end] = mask[sr_start:sr_end, sc_start:sc_end]
            win_mask &= shifted
        if np.any(win_mask):
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
        for state, policy, reward in zip(states, policies, rewards):
            self.push(state, policy, reward)

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
        # np.savez is much faster than np.savez_compressed for large buffers
        np.savez(
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
# Network
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return torch.relu(x)


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        num_res_blocks: int = 5,
        channels: int = 64,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels=channels) for _ in range(num_res_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


def encode_board(grid: np.ndarray, player: int) -> np.ndarray:
    state = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    state[0] = (grid == player).astype(np.float32)
    state[1] = (grid == other_player(player)).astype(np.float32)
    state[2] = (grid == EMPTY).astype(np.float32)
    return state


def augment_data(state: np.ndarray, policy: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    board_size = state.shape[1]
    policy_2d = policy.reshape(board_size, board_size)
    augmented: list[tuple[np.ndarray, np.ndarray]] = []

    for k in range(4):
        s_rot = np.rot90(state, k, axes=(1, 2)).copy()
        p_rot = np.rot90(policy_2d, k).copy().flatten()
        augmented.append((s_rot, p_rot))

        s_flip = np.flip(s_rot, axis=2).copy()
        p_flip = np.fliplr(np.rot90(policy_2d, k)).copy().flatten()
        augmented.append((s_flip, p_flip))

    return augmented


# ============================================================
# MCTS
# ============================================================
@dataclass
class MCTSNode:
    player: int
    parent: "MCTSNode | None" = None
    prior: float = 0.0
    move_from_parent: tuple[int, int] | None = None
    children: dict[tuple[int, int], "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_loss: int = 0
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        # Include virtual loss in Q calculation to discourage multiple simulations
        # from picking the same path before evaluation returns.
        effective_visits = self.visit_count + self.virtual_loss
        if effective_visits == 0:
            return 0.0
        return self.value_sum / effective_visits


class MCTS:
    def __init__(
        self,
        network: nn.Module,
        device: str,
        num_simulations: int = 80,
        c_puct: float = 1.4,
    ) -> None:
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.network.eval() # Ensure eval mode is set once

    def search(
        self,
        root_state: np.ndarray,
        player: int,
        temperature: float = 1.0,
        batch_size: int = 16, # Optimized batch size
    ) -> np.ndarray:
        root = MCTSNode(player=player)
        self._evaluate_and_expand_batch([root], [root_state])

        num_batches = max(1, self.num_simulations // batch_size)
        
        with torch.inference_mode():
            for _ in range(num_batches):
                leaves: list[MCTSNode] = []
                leaf_states: list[np.ndarray] = []
                paths: list[list[MCTSNode]] = []

                # Collect batch of leaves
                for _ in range(batch_size):
                    node = root
                    path = [node]
                    curr_state = root_state.copy()

                    while node.is_expanded and node.children:
                        move, node = self._select_child(node)
                        curr_state[move[0], move[1]] = other_player(node.player)
                        path.append(node)
                        # Add virtual loss while traversing
                        node.virtual_loss += 3 # Standard virtual loss value
                    
                    leaves.append(node)
                    leaf_states.append(curr_state)
                    paths.append(path)

                # Batch expansion and prediction
                values = self._evaluate_and_expand_batch(leaves, leaf_states)
                
                # Backpropagate and remove virtual loss
                for path, value in zip(paths, values):
                    for node in path[1:]: # Don't apply to root
                        node.virtual_loss -= 3
                    self._backpropagate(path, value)

        return self._build_policy(root, temperature)

    def _select_child(self, node: MCTSNode) -> tuple[tuple[int, int], MCTSNode]:
        moves = list(node.children.keys())
        children = list(node.children.values())
        
        # Effective visits include virtual loss to encourage exploration
        v_counts = np.array([c.visit_count + c.virtual_loss for c in children])
        priors = np.array([c.prior for c in children])
        q_values = np.array([-c.q_value for c in children])
        
        exploration = self.c_puct * priors * np.sqrt(node.visit_count + node.virtual_loss + 1) / (1 + v_counts)
        scores = q_values + exploration
        
        best_idx = np.argmax(scores)
        return moves[best_idx], children[best_idx]

    def _evaluate_and_expand_batch(self, nodes: list[MCTSNode], states: list[np.ndarray]) -> list[float]:
        results: list[float] = []
        to_predict_indices: list[int] = []
        to_predict_states: list[np.ndarray] = []
        to_predict_players: list[int] = []

        for i, (node, state) in enumerate(zip(nodes, states)):
            # Terminal check
            if node.move_from_parent is not None:
                prev_player = other_player(node.player)
                if is_win(state, prev_player, last_move=node.move_from_parent):
                    results.append(-1.0)
                    continue
            if is_draw(state):
                results.append(0.0)
                continue
            
            # If already expanded (can happen in batch), use cached value estimation or just predict again
            # In simple batched MCTS, we filter out duplicates or just re-predict.
            # Here we just mark for prediction for simplicity.
            results.append(0.0) # Placeholder
            to_predict_indices.append(i)
            to_predict_states.append(state)
            to_predict_players.append(node.player)

        if to_predict_states:
            policies, values = self._predict_batch(to_predict_states, to_predict_players)
            for idx, policy, value in zip(to_predict_indices, policies, values):
                node = nodes[idx]
                state = states[idx]
                
                # Expand
                if not node.is_expanded:
                    valid_mask = (state.flatten() == EMPTY).astype(np.float32)
                    masked_policy = policy * valid_mask
                    total = float(masked_policy.sum())
                    if total <= 0:
                        masked_policy = valid_mask / valid_mask.sum()
                    else:
                        masked_policy /= total

                    node.is_expanded = True
                    for move_idx in np.where(valid_mask > 0)[0]:
                        r, c = divmod(int(move_idx), BOARD_SIZE)
                        node.children[(r, c)] = MCTSNode(
                            player=other_player(node.player),
                            parent=node,
                            prior=float(masked_policy[move_idx]),
                            move_from_parent=(r, c),
                        )
                
                results[idx] = value

        return results

    def _predict_batch(self, grids: list[np.ndarray], players: list[int]) -> tuple[np.ndarray, np.ndarray]:
        # Encode all states in the batch
        batch_states = np.stack([encode_board(g, p) for g, p in zip(grids, players)])
        state_tensor = torch.as_tensor(batch_states, dtype=torch.float32, device=self.device)

        with torch.amp.autocast(device_type="cuda" if "cuda" in self.device else "cpu"):
            policy_logits, values = self.network(state_tensor)
        
        policies = torch.softmax(policy_logits.detach(), dim=1).cpu().numpy()
        value_scalars = values.detach().squeeze(1).cpu().numpy()
        return policies, value_scalars

    def _terminal_value(self, node: MCTSNode, state: np.ndarray) -> float | None:
        if node.move_from_parent is not None:
            prev_player = other_player(node.player)
            if is_win(state, prev_player, last_move=node.move_from_parent):
                return -1.0

        if is_draw(state):
            return 0.0

        return None

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _build_policy(self, root: MCTSNode, temperature: float) -> np.ndarray:
        pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)

        if not root.children:
            return pi

        for move, child in root.children.items():
            idx = move[0] * BOARD_SIZE + move[1]
            pi[idx] = float(child.visit_count)

        if temperature <= 0:
            out = np.zeros_like(pi)
            out[int(np.argmax(pi))] = 1.0
            return out

        pi = np.power(pi, 1.0 / temperature)
        return pi / pi.sum()


# ============================================================
# Agent
# ============================================================
class AlphaZeroAgent:
    def __init__(
        self,
        device: str = "cpu",
        buffer_capacity: int = 100_000,
        lr: float = 1e-3,
        lr_decay_steps: int = 2000,
        lr_decay_gamma: float = 0.5,
        num_simulations: int = 80,
        c_puct: float = 1.4,
        num_res_blocks: int = 5,
        channels: int = 64,
    ) -> None:
        self.device = device
        self.network = AlphaZeroNet(
            board_size=BOARD_SIZE,
            num_res_blocks=num_res_blocks,
            channels=channels,
        ).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_decay_steps,
            gamma=lr_decay_gamma,
        )
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.mcts = MCTS(
            network=self.network,
            device=device,
            num_simulations=num_simulations,
            c_puct=c_puct,
        )
        self.scaler = torch.amp.GradScaler("cuda") if "cuda" in device else None

    def set_num_simulations(self, num_simulations: int) -> None:
        self.mcts.num_simulations = num_simulations

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.buffer) < batch_size:
            return 0.0

        states, policies, rewards = self.buffer.sample(batch_size)
        state_tensors = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        policy_targets = torch.as_tensor(policies, dtype=torch.float32, device=self.device)
        value_targets = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.network.train() # Set to train mode
        
        device_type = "cuda" if "cuda" in self.device else "cpu"
        with torch.amp.autocast(device_type=device_type):
            policy_pred, value_pred = self.network(state_tensors)
            policy_loss = -torch.sum(
                policy_targets * torch.log_softmax(policy_pred, dim=1),
                dim=1,
            ).mean()
            value_loss = nn.MSELoss()(value_pred, value_targets)
            loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
            
        self.scheduler.step()
        self.network.eval() # Return to eval mode for MCTS
        return float(loss.item())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.network.eval()

    def save_buffer(self, path: str) -> None:
        self.buffer.save(path)

    def load_buffer(self, path: str) -> None:
        self.buffer.load(path)


# ============================================================
# Self-play + Logging
# ============================================================
def play_self_play_game(
    agent: AlphaZeroAgent,
    exploration_moves: int = 12,
    mcts_batch: int = 32,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], int | None, list[tuple[int, int, int]]]:
    board = Board()
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    players: list[int] = []
    moves: list[tuple[int, int, int]] = []
    current_player = X
    move_count = 0

    while True:
        state = encode_board(board.grid, current_player)
        temperature = 1.0 if move_count < exploration_moves else 0.0
        pi = agent.mcts.search(board.grid.copy(), current_player, temperature=temperature, batch_size=mcts_batch)

        valid_mask = (board.grid.flatten() == EMPTY).astype(np.float32)
        pi = pi * valid_mask
        total = float(pi.sum())
        if total <= 0:
            valid_indices = np.where(valid_mask > 0)[0]
            if len(valid_indices) == 0:
                winner = None
                break
            pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            pi[valid_indices] = 1.0 / len(valid_indices)
        else:
            pi = pi / total

        if temperature > 0:
            move_idx = int(np.random.choice(len(pi), p=pi))
        else:
            move_idx = int(np.argmax(pi))

        r, c = divmod(move_idx, BOARD_SIZE)
        board.place(r, c, current_player)

        states.append(state)
        policies.append(pi.astype(np.float32))
        players.append(current_player)
        moves.append((current_player, r, c))
        move_count += 1

        if is_win(board.grid, current_player, last_move=(r, c)):
            winner = current_player
            break
        if is_draw(board.grid):
            winner = None
            break

        current_player = other_player(current_player)

    rewards: list[float] = []
    for player in players:
        if winner is None:
            rewards.append(0.0)
        elif winner == player:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)

    aug_states: list[np.ndarray] = []
    aug_policies: list[np.ndarray] = []
    aug_rewards: list[float] = []
    for state, policy, reward in zip(states, policies, rewards):
        for state_aug, policy_aug in augment_data(state, policy):
            aug_states.append(state_aug)
            aug_policies.append(policy_aug)
            aug_rewards.append(reward)

    return aug_states, aug_policies, aug_rewards, winner, moves


def log_game_replay(log_path: str, match_id: int, moves: list[tuple[int, int, int]], winner: int | None) -> None:
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
    record = {
        "match_id": int(match_id),
        "moves": [{"player": int(p), "row": int(r), "col": int(c)} for p, r, c in moves],
        "winner": "X" if winner == X else ("O" if winner == O else "Draw"),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def validate_agent(agent: AlphaZeroAgent, num_matches: int = 10, sims: int = 50) -> float:
    original_sims = agent.mcts.num_simulations
    agent.set_num_simulations(sims)

    wins = 0
    try:
        for _ in range(num_matches):
            board = Board()
            current_player = X
            while True:
                if current_player == X:
                    pi = agent.mcts.search(board.grid.copy(), current_player, temperature=0.0)
                    move_idx = int(np.argmax(pi))
                    move = divmod(move_idx, BOARD_SIZE)
                else:
                    valid = board.get_valid_moves()
                    if not valid:
                        break
                    move = valid[np.random.randint(len(valid))]

                r, c = move
                board.place(r, c, current_player)

                if is_win(board.grid, current_player, last_move=(r, c)):
                    if current_player == X:
                        wins += 1
                    break
                if is_draw(board.grid):
                    break

                current_player = other_player(current_player)
    finally:
        agent.set_num_simulations(original_sims)

    return wins / num_matches


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaZero Gomoku agent (self-play)")
    parser.add_argument("--episodes", type=int, default=1000, help="Total self-play episodes (max 5000)")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N episodes")
    parser.add_argument("--model-path", default="models/rl_agent.pth", help="Model save path")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--mcts-sims", type=int, default=80, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.4, help="PUCT exploration coefficient")
    parser.add_argument("--exploration-moves", type=int, default=12, help="Opening moves sampled stochastically")
    parser.add_argument("--num-res-blocks", type=int, default=5, help="Number of residual blocks")
    parser.add_argument("--channels", type=int, default=64, help="ResNet channel width")
    parser.add_argument("--mcts-batch", type=int, default=32, help="MCTS batch size per simulation step")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    args = parser.parse_args()

    if args.episodes > MAX_EPISODES:
        print(f"--episodes capped at {MAX_EPISODES} (requested {args.episodes})")
        args.episodes = MAX_EPISODES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
        print("CPU mode: torch.num_threads set to 1")
    
    model_path = args.model_path
    meta_path = os.path.join(os.path.dirname(model_path) or ".", "checkpoint.json")
    buffer_path = model_path.replace(".pth", "_buffer.npz")
    replay_log = "logs/replays.jsonl"

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    agent = AlphaZeroAgent(
        device=device,
        buffer_capacity=args.buffer_size,
        num_simulations=args.mcts_sims,
        c_puct=args.c_puct,
        num_res_blocks=args.num_res_blocks,
        channels=args.channels,
    )

    if device == "cuda" and hasattr(torch, "compile"):
        try:
            print("Compiling model with torch.compile...")
            compiled = torch.compile(agent.network)
            agent.network = compiled
            agent.mcts.network = compiled
        except Exception as e:
            print(f"torch.compile failed: {e}")

    start_episode = 1
    win_counts: dict[str, int] = {"X": 0, "O": 0, "Draw": 0}

    if args.resume and os.path.exists(model_path) and os.path.exists(meta_path):
        agent.load(model_path)
        if os.path.exists(buffer_path):
            agent.load_buffer(buffer_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        start_episode = int(meta.get("episodes_done", 0)) + 1
        win_counts = meta.get("win_counts", win_counts)
        print(f"Resumed from episode {start_episode - 1}")

    total_target = start_episode - 1 + args.episodes
    print(f"Device: {device}")
    print(f"Episodes: {start_episode}-{total_target} ({args.episodes} new)")
    print(f"MCTS sims: {args.mcts_sims}, c_puct: {args.c_puct}")
    print(f"MCTS batch: {args.mcts_batch}, Train batch: {args.batch_size}, Grad accum: {args.grad_accum}")
    print(f"Buffer: {args.buffer_size}")
    print()

    losses: list[float] = []

    for episode in range(start_episode, total_target + 1):
        states, policies, rewards, winner, moves = play_self_play_game(
            agent,
            exploration_moves=args.exploration_moves,
            mcts_batch=args.mcts_batch,
        )
        agent.buffer.extend(states, policies, rewards)

        winner_name = "X" if winner == X else ("O" if winner == O else "Draw")
        win_counts[winner_name] = win_counts.get(winner_name, 0) + 1
        log_game_replay(replay_log, episode, moves, winner)

        loss = 0.0
        if len(agent.buffer) >= args.batch_size:
            sub_batch = max(1, args.batch_size // args.grad_accum)
            for _ in range(args.grad_accum):
                loss = agent.train_step(batch_size=sub_batch)
            losses.append(loss)

        if episode % args.save_every == 0 or episode == start_episode:
            avg_loss = sum(losses[-100:]) / min(len(losses), 100) if losses else 0.0
            print(
                f"Episode {episode:5d}/{total_target} | "
                f"Loss: {avg_loss:.4f} | "
                f"Buffer: {len(agent.buffer):6d} | "
                f"X: {win_counts['X']:4d} | "
                f"O: {win_counts['O']:4d} | "
                f"Draw: {win_counts['Draw']:4d}"
            )

        if episode % args.save_every == 0:
            agent.save(model_path)
            agent.save_buffer(buffer_path)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "episodes_done": episode,
                        "win_counts": win_counts,
                        "mcts_sims": args.mcts_sims,
                        "c_puct": args.c_puct,
                    },
                    f,
                )
            # Only validate every 500 episodes to save time
            if episode % max(500, args.save_every) == 0:
                win_rate = validate_agent(agent, sims=50)
                print(f"  -> Saved checkpoint | Validation vs random: {win_rate * 100:.0f}%")
            else:
                print(f"  -> Saved checkpoint")

    agent.save(model_path)
    agent.save_buffer(buffer_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "episodes_done": total_target,
                "win_counts": win_counts,
                "mcts_sims": args.mcts_sims,
                "c_puct": args.c_puct,
            },
            f,
        )

    print(f"\nTraining complete. Model saved to {model_path}")
    print(f"Final: X={win_counts['X']}, O={win_counts['O']}, Draw={win_counts['Draw']}")


if __name__ == "__main__":
    main()
