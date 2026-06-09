import numpy as np
from src.game.constants import BOARD_SIZE, EMPTY, X, O

# Pattern scores (higher = more threatening)
SCORES = {
    'FIVE': 1_000_000,
    'FOUR_OPEN': 100_000,
    'FOUR_BLOCKED': 10_000,
    'THREE_OPEN': 5_000,
    'THREE_BLOCKED': 500,
    'TWO_OPEN': 100,
    'TWO_BLOCKED': 10,
}

# Pre-compute center for positional bonus
_CENTER = BOARD_SIZE // 2
_CENTER_WEIGHT = 3.0

# Pre-built pattern tables: (pattern_string, score_key)
# Ordered longest-first to avoid substring overlaps during counting.
_FOUR_OPEN = ["011110"]
_FOUR_BLOCKED = ["211110", "011112", "10111", "11011", "11101"]
_THREE_OPEN = ["010110", "011010", "01110"]
_THREE_BLOCKED = ["211100", "001112", "210110", "011012", "21110", "01112", "1101", "1011"]
_TWO_OPEN = ["01010", "0110"]


def _build_line_strings(grid: np.ndarray, player: int, opponent: int) -> list[str]:
    """Convert all board lines to pattern strings in a single pass.

    Uses direct numpy indexing — avoids .tolist() overhead per line.
    '1' = player, '2' = opponent, '0' = empty.
    """
    # Lookup table: cell value -> character
    lut = ['0'] * 3  # EMPTY=0
    lut[player] = '1'
    lut[opponent] = '2'

    lines: list[str] = []

    # Rows & columns
    for i in range(BOARD_SIZE):
        row_s = ''.join(lut[grid[i, c]] for c in range(BOARD_SIZE))
        col_s = ''.join(lut[grid[r, i]] for r in range(BOARD_SIZE))
        lines.append(row_s)
        lines.append(col_s)

    # Diagonals (only those long enough to contain 5-in-a-row)
    for d in range(-BOARD_SIZE + 5, BOARD_SIZE - 4):
        diag = np.diagonal(grid, offset=d)
        lines.append(''.join(lut[v] for v in diag))
        anti = np.diagonal(grid[:, ::-1], offset=d)
        lines.append(''.join(lut[v] for v in anti))

    return lines


def _count_pattern(s: str, pattern: str) -> int:
    """Count non-overlapping occurrences of pattern in s."""
    count = 0
    start = 0
    plen = len(pattern)
    while True:
        idx = s.find(pattern, start)
        if idx == -1:
            break
        count += 1
        start = idx + plen
    return count


def _evaluate_line(s: str) -> tuple[float, int, int]:
    """Score a single line string. Returns (score, open_threes, fours)."""
    if len(s) < 5:
        return 0.0, 0, 0

    # Five — instant win, early return
    if '11111' in s:
        return float(SCORES['FIVE']), 0, 0

    score = 0.0
    num_fours = 0
    num_open_threes = 0

    # Open four
    for p in _FOUR_OPEN:
        c = _count_pattern(s, p)
        if c:
            score += SCORES['FOUR_OPEN'] * c
            num_fours += c

    # Blocked / split four
    for p in _FOUR_BLOCKED:
        c = _count_pattern(s, p)
        if c:
            score += SCORES['FOUR_BLOCKED'] * c
            num_fours += c
    # Edge fours (line starts/ends)
    if s[:5] == '11110':
        score += SCORES['FOUR_BLOCKED']
        num_fours += 1
    if s[-5:] == '01111':
        score += SCORES['FOUR_BLOCKED']
        num_fours += 1

    # Open three
    for p in _THREE_OPEN:
        c = _count_pattern(s, p)
        if c:
            score += SCORES['THREE_OPEN'] * c
            num_open_threes += c

    # Blocked three
    for p in _THREE_BLOCKED:
        c = _count_pattern(s, p)
        if c:
            score += SCORES['THREE_BLOCKED'] * c

    # Open two
    for p in _TWO_OPEN:
        c = _count_pattern(s, p)
        if c:
            score += SCORES['TWO_OPEN'] * c

    return score, num_open_threes, num_fours


def evaluate(grid: np.ndarray, player: int) -> float:
    """Evaluate board position for *player*.

    Positive = player advantage, negative = opponent advantage.
    """
    opponent = O if player == X else X

    # Build line strings once per player (avoid double extraction)
    p_lines = _build_line_strings(grid, player, opponent)
    o_lines = _build_line_strings(grid, opponent, player)

    player_score = 0.0
    opponent_score = 0.0
    p_open_threes = 0
    p_fours = 0
    o_open_threes = 0
    o_fours = 0

    for ps, os_ in zip(p_lines, o_lines):
        s, ot, f = _evaluate_line(ps)
        player_score += s
        p_open_threes += ot
        p_fours += f

        s, ot, f = _evaluate_line(os_)
        opponent_score += s
        o_open_threes += ot
        o_fours += f

    # Double-threat bonus
    if p_open_threes >= 2 or (p_open_threes >= 1 and p_fours >= 1):
        player_score += SCORES['FOUR_OPEN'] * 0.8
    if o_open_threes >= 2 or (o_open_threes >= 1 and o_fours >= 1):
        opponent_score += SCORES['FOUR_OPEN'] * 0.9

    # Center-proximity positional bonus
    pos_bonus = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] == player:
                dist = abs(r - _CENTER) + abs(c - _CENTER)
                pos_bonus += _CENTER_WEIGHT * max(0, BOARD_SIZE - dist)
            elif grid[r, c] == opponent:
                dist = abs(r - _CENTER) + abs(c - _CENTER)
                pos_bonus -= _CENTER_WEIGHT * max(0, BOARD_SIZE - dist)

    return player_score - opponent_score + pos_bonus
