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
    
    # Fallback to full board check (highly optimized using numpy)
    for dr, dc in directions:
        # We check for 5-in-a-row by shifting the grid and bitwise-anding
        # player_mask & shift(player_mask, 1) & shift(player_mask, 2) ...
        # This is much faster than nested loops in Python.
        mask = (grid == player)
        win_mask = mask.copy()
        for k in range(1, 5):
            # Shift mask in direction (dr, dc)
            shifted = np.zeros_like(mask)
            r_start, r_end = max(0, dr*k), min(BOARD_SIZE, BOARD_SIZE + dr*k)
            c_start, c_end = max(0, dc*k), min(BOARD_SIZE, BOARD_SIZE + dc*k)
            sr_start, sr_end = max(0, -dr*k), min(BOARD_SIZE, BOARD_SIZE - dr*k)
            sc_start, sc_end = max(0, -dc*k), min(BOARD_SIZE, BOARD_SIZE - dc*k)
            
            shifted[r_start:r_end, c_start:c_end] = mask[sr_start:sr_end, sc_start:sc_end]
            win_mask &= shifted
            
        if np.any(win_mask):
            return True
    return False


def is_draw(grid: np.ndarray) -> bool:
    return not np.any(grid == EMPTY)


def is_game_over(grid: np.ndarray, last_move: tuple[int, int] | None = None) -> tuple[bool, int | None]:
    if last_move is not None:
        r, c = last_move
        player = grid[r, c]
        if player != EMPTY and is_win(grid, player, last_move):
            return True, player
        if is_draw(grid):
            return True, None
        return False, None

    if is_win(grid, X):
        return True, X
    if is_win(grid, O):
        return True, O
    if is_draw(grid):
        return True, None
    return False, None


def valid_moves(grid: np.ndarray) -> list[tuple[int, int]]:
    return list(zip(*np.where(grid == EMPTY)))
