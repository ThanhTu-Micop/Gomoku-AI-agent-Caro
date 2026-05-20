import numpy as np
from src.game.rules import is_win, is_draw, is_game_over, valid_moves
from src.game.constants import BOARD_SIZE, EMPTY, X, O


def test_is_win_horizontal() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, :5] = X
    assert is_win(grid, X)
    assert not is_win(grid, O)


def test_is_win_vertical() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[:5, 0] = X
    assert is_win(grid, X)


def test_is_win_diagonal() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    for i in range(5):
        grid[i, i] = X
    assert is_win(grid, X)


def test_is_win_anti_diagonal() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    for i in range(5):
        grid[i, 4 - i] = X
    assert is_win(grid, X)


def test_is_win_not_five() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, :4] = X
    assert not is_win(grid, X)


def test_is_draw_full_board() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), X, dtype=int)
    assert is_draw(grid)


def test_is_draw_not_full() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    assert not is_draw(grid)


def test_is_game_over_win() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, :5] = X
    game_over, winner = is_game_over(grid)
    assert game_over
    assert winner == X


def test_is_game_over_draw() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    pattern = [
        [X, O, X, O, O],
        [O, X, O, X, O],
        [X, O, O, X, O],
        [O, X, X, O, X],
        [X, O, X, O, X],
    ]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            grid[r, c] = pattern[r % 5][c % 5]
    game_over, winner = is_game_over(grid)
    assert game_over
    assert winner is None


def test_is_game_over_not_over() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    game_over, winner = is_game_over(grid)
    assert not game_over
    assert winner is None


def test_valid_moves() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    moves = valid_moves(grid)
    assert len(moves) == BOARD_SIZE * BOARD_SIZE


def test_valid_moves_partial() -> None:
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, 0] = X
    grid[0, 1] = O
    moves = valid_moves(grid)
    assert len(moves) == BOARD_SIZE * BOARD_SIZE - 2
