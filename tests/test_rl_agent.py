import numpy as np
import torch

from src.ai.rl_agent import AlphaZeroAgent, AlphaZeroNet
from src.game.constants import BOARD_SIZE, EMPTY, X


def test_alphazero_net_forward_shapes() -> None:
    net = AlphaZeroNet(board_size=BOARD_SIZE, num_res_blocks=2, channels=32)
    x = torch.zeros((1, 3, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    policy, value = net(x)

    assert policy.shape == (1, BOARD_SIZE * BOARD_SIZE)
    assert value.shape == (1, 1)


def test_alphazero_agent_returns_valid_move() -> None:
    agent = AlphaZeroAgent(device="cpu", num_simulations=10, num_res_blocks=2, channels=32)
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    move = agent.get_move(grid, X)

    assert 0 <= move[0] < BOARD_SIZE
    assert 0 <= move[1] < BOARD_SIZE
    assert grid[move] == EMPTY
