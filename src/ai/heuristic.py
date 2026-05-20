import numpy as np
from src.game.constants import BOARD_SIZE, EMPTY, X, O


PATTERN_SCORES: dict[tuple[int, int, int], float] = {
    (5, 2, 0): 100000.0,
    (5, 1, 1): 100000.0,
    (5, 0, 2): 100000.0,
    (4, 2, 0): 10000.0,
    (4, 1, 1): 1000.0,
    (4, 0, 2): 100.0,
    (3, 2, 0): 500.0,
    (3, 1, 1): 100.0,
    (3, 0, 2): 50.0,
    (2, 2, 0): 50.0,
    (2, 1, 1): 10.0,
    (2, 0, 2): 5.0,
    (1, 2, 0): 5.0,
    (1, 1, 1): 1.0,
    (1, 0, 2): 0.0,
}


def _get_all_lines(grid: np.ndarray) -> list[np.ndarray]:
    lines = []
    # Rows
    for r in range(BOARD_SIZE):
        lines.append(grid[r, :])
    # Columns
    for c in range(BOARD_SIZE):
        lines.append(grid[:, c])
    # Diagonals
    for d in range(-BOARD_SIZE + 5, BOARD_SIZE - 4):
        lines.append(np.diagonal(grid, offset=d))
        lines.append(np.diagonal(np.fliplr(grid), offset=d))
    return lines


def _evaluate_line(line: np.ndarray, player: int) -> float:
    score = 0.0
    opponent = O if player == X else X
    n = len(line)
    if n < 5:
        return 0.0

    # Sliding window of size 5 or more to find patterns
    # Simplified approach: for each segment of same-colored stones
    i = 0
    while i < n:
        if line[i] == player:
            start = i
            while i < n and line[i] == player:
                i += 1
            count = i - start
            
            open_ends = 0
            blocked = 0
            
            # Check left side
            if start == 0:
                blocked += 1
            elif line[start - 1] == EMPTY:
                open_ends += 1
            elif line[start - 1] == opponent:
                blocked += 1
            
            # Check right side
            if i == n:
                blocked += 1
            elif line[i] == EMPTY:
                open_ends += 1
            elif line[i] == opponent:
                blocked += 1
                
            key = (min(count, 5), open_ends, blocked)
            if key in PATTERN_SCORES:
                score += PATTERN_SCORES[key]
        else:
            i += 1
    return score


def evaluate(grid: np.ndarray, player: int) -> float:
    player_score = 0.0
    opponent_score = 0.0
    opponent = O if player == X else X
    
    lines = _get_all_lines(grid)
    for line in lines:
        player_score += _evaluate_line(line, player)
        opponent_score += _evaluate_line(line, opponent)
        
    return player_score - opponent_score
