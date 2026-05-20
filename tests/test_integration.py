import numpy as np
from src.game.board import Board
from src.game.rules import is_win, is_draw, is_game_over, valid_moves
from src.game.constants import BOARD_SIZE, EMPTY, X, O


def test_full_game_win_horizontal():
    b = Board()
    moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4)]
    for i, (r, c) in enumerate(moves):
        p = X if i % 2 == 0 else O
        b.place(r, c, p)
    game_over, winner = is_game_over(b.grid)
    assert game_over
    assert winner == X


def test_full_game_win_vertical():
    b = Board()
    moves = [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0)]
    for i, (r, c) in enumerate(moves):
        p = X if i % 2 == 0 else O
        b.place(r, c, p)
    game_over, winner = is_game_over(b.grid)
    assert game_over
    assert winner == X


def test_full_game_draw():
    b = Board()
    pattern = [
        [X, O, X, O, O],
        [O, X, O, X, O],
        [X, O, O, X, O],
        [O, X, X, O, X],
        [X, O, X, O, X],
    ]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            b.place(r, c, pattern[r % 5][c % 5])
    game_over, winner = is_game_over(b.grid)
    assert game_over
    assert winner is None


def test_cannot_overwrite():
    b = Board()
    assert b.place(0, 0, X)
    assert not b.place(0, 0, O)
    assert b.grid[0, 0] == X


def test_valid_moves_decreases():
    b = Board()
    initial = len(b.get_valid_moves())
    assert initial == BOARD_SIZE * BOARD_SIZE
    b.place(0, 0, X)
    assert len(b.get_valid_moves()) == initial - 1


def test_board_reset():
    b = Board()
    b.place(0, 0, X)
    b.place(1, 1, O)
    b.reset()
    assert b.grid[0, 0] == EMPTY
    assert b.grid[1, 1] == EMPTY
    assert b.move_count == 0


def test_board_copy():
    b = Board()
    b.place(0, 0, X)
    b2 = b.copy()
    assert b2.grid[0, 0] == X
    b.place(0, 1, O)
    assert b2.grid[0, 1] == EMPTY
