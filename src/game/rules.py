import numpy as np
from src.game.constants import BOARD_SIZE, EMPTY, X, O


def is_win(grid: np.ndarray, player: int, last_move: tuple[int, int] | None = None) -> bool:
    """Check if the given player has won the game.
    
    Args:
        grid: The board grid.
        player: The player to check (X or O).
        last_move: Optional last move (row, col) to optimize the check.
        
    Returns:
        True if the player has won, False otherwise.
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    if last_move is not None:
        r_last, c_last = last_move
        for dr, dc in directions:
            count = 1
            # Check in positive direction
            r, c = r_last + dr, c_last + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
                count += 1
                r += dr
                c += dc
            # Check in negative direction
            r, c = r_last - dr, c_last - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                return True
        return False
    
    # Fallback to full board check (optimized)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] != player:
                continue
            for dr, dc in directions:
                # Only check in one direction to avoid double counting
                count = 1
                for k in range(1, 5):
                    nr, nc = r + dr * k, c + dc * k
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and grid[nr, nc] == player:
                        count += 1
                    else:
                        break
                if count >= 5:
                    return True
    return False


def is_draw(grid: np.ndarray) -> bool:
    return not np.any(grid == EMPTY)


def is_game_over(grid: np.ndarray) -> tuple[bool, int | None]:
    if is_win(grid, X):
        return True, X
    if is_win(grid, O):
        return True, O
    if is_draw(grid):
        return True, None
    return False, None


def valid_moves(grid: np.ndarray) -> list[tuple[int, int]]:
    return list(zip(*np.where(grid == EMPTY)))
