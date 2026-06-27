import numpy as np
import torch
import torch.nn as nn

from src.ai.mcts import MCTS
from src.game.constants import BOARD_SIZE, EMPTY, O, X


class CenterBiasedNet(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        logits = torch.zeros((batch, BOARD_SIZE * BOARD_SIZE), dtype=torch.float32, device=x.device)
        center_idx = (BOARD_SIZE // 2) * BOARD_SIZE + (BOARD_SIZE // 2)
        logits[:, center_idx] = 3.0
        value = torch.zeros((batch, 1), dtype=torch.float32, device=x.device)
        return logits, value


class UniformNet(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        logits = torch.zeros((batch, BOARD_SIZE * BOARD_SIZE), dtype=torch.float32, device=x.device)
        value = torch.zeros((batch, 1), dtype=torch.float32, device=x.device)
        return logits, value


def test_mcts_returns_valid_pi_and_masks_occupied() -> None:
    mcts = MCTS(network=CenterBiasedNet(), num_simulations=50, c_puct=1.4)
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)

    pi = mcts.search(grid, X, temperature=1.0)
    assert pi.shape == (BOARD_SIZE * BOARD_SIZE,)
    assert np.isclose(pi.sum(), 1.0)

    center_idx = (BOARD_SIZE // 2) * BOARD_SIZE + (BOARD_SIZE // 2)
    assert int(np.argmax(pi)) == center_idx

    grid[0, 0] = X
    pi2 = mcts.search(grid, O, temperature=1.0)
    assert np.isclose(pi2.sum(), 1.0)
    assert pi2[0] == 0.0


def test_mcts_finds_immediate_winning_move() -> None:
    mcts = MCTS(network=UniformNet(), num_simulations=80, c_puct=1.4)
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)

    grid[0, 0] = X
    grid[0, 1] = X
    grid[0, 2] = X
    grid[0, 3] = X
    grid[1, 0] = O
    grid[1, 1] = O

    pi = mcts.search(grid, X, temperature=0.0)
    move_idx = int(np.argmax(pi))
    assert move_idx == 4
