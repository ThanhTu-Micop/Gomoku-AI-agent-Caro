import numpy as np
from src.game.constants import BOARD_SIZE, EMPTY, X, O

# Pattern scores matching the priority table:
#   Five                    100000
#   Double Open Four         50000
#   Open Four                10000
#   Open Three + Open Four    8000
#   Double Open Three         5000
#   Closed Four               2000
#   Open Three                1000
#   Broken Three               300
#   Open Two                   100
#   Single Stone                10
SCORES = {
    'FIVE': 100_000,
    'DOUBLE_OPEN_FOUR': 50_000,
    'FOUR_OPEN': 10_000,
    'FOUR_BLOCKED': 2_000,
    'OPEN_THREE_PLUS_FOUR': 8_000,
    'DOUBLE_OPEN_THREE': 5_000,
    'THREE_OPEN': 1_000,
    'THREE_BLOCKED': 300,
    'TWO_OPEN': 100,
    'TWO_BLOCKED': 10,
    'FORK_BONUS': 200_000,
    'EDGE_PENALTY': -200,
    'HIDDEN_CONNECTION': 200,
    'SHAPE_SPLIT_FOUR_BONUS': 15_000,
    'CENTER_BONUS': 5.0,
    'CENTER_RING_BONUS': 2.0,
}

# Pre-compute center for positional bonus
_CENTER = BOARD_SIZE // 2

# Pre-built pattern tables: (pattern_string, score_key)
# Ordered longest-first to avoid substring overlaps during counting.
_FOUR_OPEN = ["011110"]
_FOUR_BLOCKED = ["211110", "011112", "10111", "11011", "11101"]
_THREE_OPEN = ["010110", "011010", "01110"]
_THREE_BLOCKED = ["211100", "001112", "210110", "011012", "21110", "01112"]
_TWO_OPEN = ["01010", "0110"]

# Broken three patterns (Group 5, patterns 23-28)
_BROKEN_THREE = ["1101", "1011", "10101", "11001", "10011"]

# Hidden connection patterns (Group 10, patterns 48-50)
_HIDDEN_CONN = ["1001", "10001", "10101", "01001", "10010"]

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _build_line_strings(grid: np.ndarray, player: int, opponent: int) -> list[str]:
    """Convert all board lines to pattern strings in a single pass."""
    lut = ['0'] * 3
    lut[player] = '1'
    lut[opponent] = '2'

    lines: list[str] = []

    for i in range(BOARD_SIZE):
        row_s = ''.join(lut[grid[i, c]] for c in range(BOARD_SIZE))
        col_s = ''.join(lut[grid[r, i]] for r in range(BOARD_SIZE))
        lines.append(row_s)
        lines.append(col_s)

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


def _evaluate_line(s: str) -> tuple[float, int, int, int]:
    """Score a single line string. Returns (score, open_threes, fours, hidden_conns)."""
    if len(s) < 5:
        return 0.0, 0, 0, 0

    # Five — instant win, early return
    if '11111' in s:
        return float(SCORES['FIVE']), 0, 0, 0

    score = 0.0
    num_fours = 0
    num_open_threes = 0
    num_hidden = 0

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
    if s[:5] == '11110':
        score += SCORES['FOUR_BLOCKED']
        num_fours += 1
    if s[-5:] == '01111':
        score += SCORES['FOUR_BLOCKED']
        num_fours += 1

    # Shape trap: split four (XX_XX) is MORE dangerous
    if '11011' in s:
        score += SCORES['SHAPE_SPLIT_FOUR_BONUS']

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

    # Broken three (patterns 23-28)
    for p in _BROKEN_THREE:
        c = _count_pattern(s, p)
        if c:
            score += SCORES['THREE_BLOCKED'] * c * 0.8

    # Open two
    for p in _TWO_OPEN:
        c = _count_pattern(s, p)
        if c:
            score += SCORES['TWO_OPEN'] * c

    # Hidden connections: X__X, X_X_X (long-range danger)
    for p in _HIDDEN_CONN:
        c = _count_pattern(s, p)
        if c:
            score += SCORES['HIDDEN_CONNECTION'] * c
            num_hidden += c

    return score, num_open_threes, num_fours, num_hidden


def _count_threat_directions(grid: np.ndarray, row: int, col: int, player: int) -> int:
    """Count how many directions a placed stone creates threats (3+ in a row)."""
    count = 0
    for dr, dc in DIRECTIONS:
        streak = 1
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
            streak += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
            streak += 1
            r -= dr
            c -= dc
        if streak >= 3:
            count += 1
    return count


def _count_open_threes_at_position(
    grid: np.ndarray, row: int, col: int, player: int
) -> int:
    """Count open three directions created by placing stone at (row, col)."""
    count = 0
    for dr, dc in DIRECTIONS:
        streak = 1
        # Positive direction
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
            streak += 1
            r += dr
            c += dc
        # Check if positive end is open
        pos_open = (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == EMPTY)
        # Negative direction
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
            streak += 1
            r -= dr
            c -= dc
        # Check if negative end is open
        neg_open = (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == EMPTY)
        # Open three: 3+ in a row with both ends open
        if streak >= 3 and pos_open and neg_open:
            count += 1
    return count


def _count_fours_at_position(
    grid: np.ndarray, row: int, col: int, player: int
) -> int:
    """Count four-in-a-row directions created by placing stone at (row, col)."""
    count = 0
    for dr, dc in DIRECTIONS:
        streak = 1
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
            streak += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
            streak += 1
            r -= dr
            c -= dc
        if streak >= 4:
            count += 1
    return count


def _edge_penalty(row: int, col: int) -> float:
    """Penalty for edge/corner positions — fewer directions to grow."""
    if row == 0 or row == BOARD_SIZE - 1 or col == 0 or col == BOARD_SIZE - 1:
        return SCORES['EDGE_PENALTY']
    if (row == 1 or row == BOARD_SIZE - 2) and (col == 1 or col == BOARD_SIZE - 2):
        return SCORES['EDGE_PENALTY'] * 0.5
    return 0.0


def _center_bonus(row: int, col: int) -> float:
    """Bonus for center positions — more directions to grow.

    Uses ring-based scoring: center 3x3 gets highest bonus,
    next ring gets medium, etc.
    """
    dist = max(abs(row - _CENTER), abs(col - _CENTER))
    if dist == 0:
        return SCORES['CENTER_BONUS'] * 3
    elif dist == 1:
        return SCORES['CENTER_BONUS'] * 2
    elif dist == 2:
        return SCORES['CENTER_RING_BONUS']
    return 0.0


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
    p_hidden = 0
    o_open_threes = 0
    o_fours = 0
    o_hidden = 0

    for ps, os_ in zip(p_lines, o_lines):
        s, ot, f, h = _evaluate_line(ps)
        player_score += s
        p_open_threes += ot
        p_fours += f
        p_hidden += h

        s, ot, f, h = _evaluate_line(os_)
        opponent_score += s
        o_open_threes += ot
        o_fours += f
        o_hidden += h

    # Double-threat bonus (patterns 34-38):
    # P34: Double open three → nearly unbeatable
    # P35: Open three + open four → almost certain win
    # P36: Two open fours → certain win
    if p_open_threes >= 2:
        player_score += SCORES['DOUBLE_OPEN_THREE']
    elif p_open_threes >= 1 and p_fours >= 1:
        player_score += SCORES['OPEN_THREE_PLUS_FOUR']
    elif p_fours >= 2:
        player_score += SCORES['DOUBLE_OPEN_FOUR']

    if o_open_threes >= 2:
        opponent_score += SCORES['DOUBLE_OPEN_THREE']
    elif o_open_threes >= 1 and o_fours >= 1:
        opponent_score += SCORES['OPEN_THREE_PLUS_FOUR']
    elif o_fours >= 2:
        opponent_score += SCORES['DOUBLE_OPEN_FOUR']

    # Fork bonus: check all empty positions for multi-direction threats
    p_fork_count = 0
    o_fork_count = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] != EMPTY:
                continue
            p_threats = _count_threat_directions(grid, r, c, player)
            if p_threats >= 2:
                p_fork_count += 1
            o_threats = _count_threat_directions(grid, r, c, opponent)
            if o_threats >= 2:
                o_fork_count += 1

    if p_fork_count >= 1:
        player_score += SCORES['FORK_BONUS'] * p_fork_count
    if o_fork_count >= 1:
        opponent_score += SCORES['FORK_BONUS'] * o_fork_count

    # Hidden connection bonus: X__X, X_X_X patterns
    if p_hidden >= 2:
        player_score += SCORES['HIDDEN_CONNECTION'] * p_hidden * 0.5
    if o_hidden >= 2:
        opponent_score += SCORES['HIDDEN_CONNECTION'] * o_hidden * 0.5

    # Center control + edge penalty
    pos_bonus = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] == player:
                pos_bonus += _center_bonus(r, c)
                pos_bonus += _edge_penalty(r, c)
            elif grid[r, c] == opponent:
                pos_bonus -= _center_bonus(r, c)
                pos_bonus -= _edge_penalty(r, c)

    return player_score - opponent_score + pos_bonus
