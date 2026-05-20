import numpy as np
from src.ai.heuristic import evaluate
from src.game.constants import BOARD_SIZE, EMPTY, X, O


def test_evaluate_empty_board() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    score = evaluate(grid, X)
    assert score == 0.0


def test_evaluate_winning_position() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, :5] = X
    score = evaluate(grid, X)
    assert score > 0


def test_evaluate_blocking() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, :4] = X
    grid[0, 5] = O
    score_x = evaluate(grid, X)
    score_o = evaluate(grid, O)
    assert score_x > 0
    assert score_o < 0


def test_evaluate_open_four() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, 2:6] = X
    score = evaluate(grid, X)
    assert score > 0
