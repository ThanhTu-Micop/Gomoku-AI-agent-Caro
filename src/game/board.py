import numpy as np
from src.game.constants import BOARD_SIZE, EMPTY, X, O


class Board:
    """9x9 Gomoku board representation."""

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

    def undo(self, row: int, col: int) -> None:
        self.grid[row, col] = EMPTY
        self.move_count -= 1

    def get_valid_moves(self) -> list[tuple[int, int]]:
        return list(zip(*np.where(self.grid == EMPTY)))

    def is_full(self) -> bool:
        return self.move_count == BOARD_SIZE * BOARD_SIZE

    def copy(self) -> "Board":
        b = Board()
        b.grid = self.grid.copy()
        b.move_count = self.move_count
        return b

    def __repr__(self) -> str:
        symbols = {EMPTY: ".", X: "X", O: "O"}
        rows = []
        for r in range(BOARD_SIZE):
            row = " ".join(symbols[self.grid[r, c]] for c in range(BOARD_SIZE))
            rows.append(row)
        return "\n".join(rows)
