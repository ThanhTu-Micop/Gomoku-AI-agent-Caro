import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.ai.base import Agent
from src.ai.mcts import MCTS
from src.ai.threats import find_critical_threats, find_open_fours, find_fork_moves
from src.game.constants import BOARD_SIZE, EMPTY, X, O
from src.utils.replay_buffer import ReplayBuffer


class GomokuNetLegacy(nn.Module):
    """Original lightweight CNN used by the legacy RL agent."""

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


GomokuNet = GomokuNetLegacy


class ResidualBlock(nn.Module):
    """Residual block with two Conv-BN layers and skip connection."""

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
    """ResNet with policy/value heads for AlphaZero-style training."""

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
    state[1] = (grid == (O if player == X else X)).astype(np.float32)
    state[2] = (grid == EMPTY).astype(np.float32)
    return state


def augment_data(state: np.ndarray, policy: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate 8 symmetry augmentations from one (state, policy) pair."""
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


class RLAgent(Agent):
    """Legacy policy/value agent without MCTS (kept for comparison)."""

    def __init__(
        self,
        device: str = "cpu",
        buffer_capacity: int = 100_000,
        lr: float = 1e-3,
        lr_decay_steps: int = 500,
        lr_decay_gamma: float = 0.5,
    ) -> None:
        self.device = device
        self.network = GomokuNetLegacy().to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_decay_steps,
            gamma=lr_decay_gamma,
        )
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

    def train_step(self, batch_size: int = 64) -> dict[str, float]:
        if len(self.buffer) < batch_size:
            return {"total": 0.0, "policy": 0.0, "value": 0.0}

        states, policies, rewards = self.buffer.sample(batch_size)
        state_tensors = torch.tensor(states, dtype=torch.float32).to(self.device)
        policy_targets = torch.tensor(policies, dtype=torch.float32).to(self.device)
        value_targets = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.network.train()
        policy_pred, value_pred = self.network(state_tensors)

        policy_loss = -torch.sum(
            policy_targets * torch.log_softmax(policy_pred, dim=1),
            dim=1,
        ).mean()
        value_loss = nn.MSELoss()(value_pred, value_targets)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"total": loss.item(), "policy": policy_loss.item(), "value": value_loss.item()}

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("network", checkpoint)
        self.network.load_state_dict(state_dict)
        self.network.eval()

    def save_buffer(self, path: str) -> None:
        self.buffer.save(path)

    def load_buffer(self, path: str) -> None:
        self.buffer.load(path)


class AlphaZeroAgent(Agent):
    """AlphaZero-style agent using ResNet and MCTS for move selection."""

    def __init__(
        self,
        device: str = "cpu",
        buffer_capacity: int = 100_000,
        lr: float = 1e-3,
        lr_decay_steps: int = 2_000,
        lr_decay_gamma: float = 0.5,
        num_simulations: int = 200,
        c_puct: float = 1.5,
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
            device=self.device,
            num_simulations=num_simulations,
            c_puct=c_puct,
        )

    def set_num_simulations(self, num_simulations: int) -> None:
        self.mcts.num_simulations = num_simulations

    def get_move(self, grid: np.ndarray, player: int, deterministic: bool = True) -> tuple[int, int]:
        temperature = 0.0 if deterministic else 1.0
        pi = self.mcts.search(grid, player, temperature=temperature)
        valid_mask = (grid.flatten() == EMPTY).astype(np.float32)
        pi = pi * valid_mask
        total = float(pi.sum())

        if total <= 0:
            valid = np.where(valid_mask > 0)[0]
            if len(valid) == 0:
                return BOARD_SIZE // 2, BOARD_SIZE // 2
            move_idx = int(valid[0])
        else:
            pi = pi / total
            if deterministic:
                move_idx = int(np.argmax(pi))
            else:
                move_idx = int(np.random.choice(len(pi), p=pi))

        best_move = (move_idx // BOARD_SIZE, move_idx % BOARD_SIZE)

        # Safety net: if opponent has immediate winning threat and MCTS missed it, force block
        opponent = O if player == X else X
        critical = find_critical_threats(grid, opponent)
        if critical:
            if best_move not in critical:
                return critical[0]

        open_fours = find_open_fours(grid, opponent)
        if open_fours:
            if best_move not in open_fours:
                return open_fours[0]

        # Fork check: if opponent can create 2+ threats, must block
        opponent_forks = find_fork_moves(grid, opponent)
        if opponent_forks:
            if best_move not in opponent_forks:
                return opponent_forks[0]

        return best_move

    def record_experience(
        self,
        states: list[np.ndarray],
        policies: list[np.ndarray],
        rewards: list[float],
    ) -> None:
        self.buffer.extend(states, policies, rewards)

    def train_step(self, batch_size: int = 64) -> dict[str, float]:
        if len(self.buffer) < batch_size:
            return {"total": 0.0, "policy": 0.0, "value": 0.0}

        states, policies, rewards = self.buffer.sample(batch_size)
        state_tensors = torch.tensor(states, dtype=torch.float32).to(self.device)
        policy_targets = torch.tensor(policies, dtype=torch.float32).to(self.device)
        value_targets = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.network.train()
        policy_pred, value_pred = self.network(state_tensors)
        policy_loss = -torch.sum(
            policy_targets * torch.log_softmax(policy_pred, dim=1),
            dim=1,
        ).mean()
        value_loss = nn.MSELoss()(value_pred, value_targets)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.network.eval()
        return {"total": loss.item(), "policy": policy_loss.item(), "value": value_loss.item()}

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Support both formats: dict with 'network' key (new) or plain state_dict (legacy)
        if 'network' in checkpoint:
            state_dict = checkpoint['network']
        else:
            state_dict = checkpoint

        # Strip _orig_mod. prefix from torch.compile
        clean_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace('_orig_mod.', '')
            clean_state_dict[clean_key] = value

        self.network.load_state_dict(clean_state_dict)

        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.network.eval()

    def save_buffer(self, path: str) -> None:
        self.buffer.save(path)

    def load_buffer(self, path: str) -> None:
        self.buffer.load(path)
