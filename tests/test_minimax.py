import numpy as np
from src.ai.minimax import MinimaxAgent
from src.game.constants import BOARD_SIZE, EMPTY, X, O


def test_minimax_returns_valid_move_on_empty() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    agent = MinimaxAgent(depth=1)
    move = agent.get_move(grid, X)
    assert 0 <= move[0] < BOARD_SIZE
    assert 0 <= move[1] < BOARD_SIZE
    assert grid[move] == EMPTY


def test_minimax_returns_valid_move_on_partial_board() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[4, 4] = X
    agent = MinimaxAgent(depth=1)
    move = agent.get_move(grid, O)
    assert 0 <= move[0] < BOARD_SIZE
    assert 0 <= move[1] < BOARD_SIZE
    assert grid[move] == EMPTY


def test_minimax_blocks_immediate_threat() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, :4] = X
    agent = MinimaxAgent(depth=3)
    move = agent.get_move(grid, O)
    assert move == (0, 4)


def test_minimax_wins_if_possible() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, :4] = O
    agent = MinimaxAgent(depth=3)
    move = agent.get_move(grid, O)
    assert move == (0, 4)
