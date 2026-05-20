import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.ai.base import Agent
from src.game.constants import BOARD_SIZE, EMPTY, X, O
from src.game.rules import is_win, is_draw
from src.utils.replay_buffer import ReplayBuffer


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


class RLAgent(Agent):
    def __init__(
        self,
        device: str = "cpu",
        buffer_capacity: int = 100_000,
        lr: float = 1e-3,
        lr_decay_steps: int = 500,
        lr_decay_gamma: float = 0.5,
    ) -> None:
        self.device = device
        self.network = GomokuNet().to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_decay_steps, gamma=lr_decay_gamma
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

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.buffer) < batch_size:
            return 0.0

        states, policies, rewards = self.buffer.sample(batch_size)

        state_tensors = torch.tensor(states, dtype=torch.float32).to(self.device)
        policy_targets = torch.tensor(policies, dtype=torch.float32).to(self.device)
        value_targets = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        policy_pred, value_pred = self.network(state_tensors)

        policy_loss = -torch.sum(
            policy_targets * torch.log_softmax(policy_pred, dim=1), dim=1
        ).mean()

        value_loss = nn.MSELoss()(value_pred, value_targets)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.network.eval()

    def save_buffer(self, path: str) -> None:
        self.buffer.save(path)

    def load_buffer(self, path: str) -> None:
        self.buffer.load(path)
