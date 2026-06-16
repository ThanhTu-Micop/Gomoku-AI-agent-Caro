"""Comprehensive Gomoku tactical pattern recognition (50+ patterns).

Pattern groups:
  Group 1 (1-5):   Five-in-a-row (handled by is_win, not here)
  Group 2 (6-10):  Open Four — _XXXX_, _XXXX, XXXX_, _XXX_X, _X_XXX
  Group 3 (11-15): Closed Four — OXXXX_, _XXXXO, OXXXXO, XXXX_X, X_XXXX
  Group 4 (16-22): Open Three — _XXX_, _XX_X_, _X_XX_, _X_X_X_, _XXX__, __XXX_
  Group 5 (23-28): Broken Three — XX_X, X_XX, X_X_X, XX__X, X__XX, X_X__X
  Group 6 (29-33): Open Two — _XX_, _X_X_, _X__X_, __XX_, _XX__
  Group 7 (34-38): Double Threat — forks, combined fours+threes
  Group 8 (39-43): Fork Shapes — cross, T, L, diagonal forks
  Group 9 (44-47): Defensive — blocking patterns
  Group 10 (48-50): Hidden Threats — X__X, X_X_X, X__X__X

Scoring priority (for MCTS boost and heuristic):
  Five                    100000
  Double Open Four         50000
  Open Four                10000
  Open Three + Open Four    8000
  Double Open Three         5000
  Closed Four               2000
  Open Three                1000
  Broken Three               300
  Open Two                   100
  Single Stone                10
"""

from __future__ import annotations

import numpy as np

from src.game.constants import BOARD_SIZE, EMPTY, X, O


def _other(player: int) -> int:
    return O if player == X else X


DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _extract_line(
    grid: np.ndarray, r: int, c: int, dr: int, dc: int, length: int
) -> list[tuple[int, int, int]]:
    """Extract a line of cells from (r,c) in direction (dr,dc).

    Returns list of (value, row, col) for each valid cell.
    """
    cells: list[tuple[int, int, int]] = []
    for k in range(length):
        nr, nc = r + dr * k, c + dc * k
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            cells.append((int(grid[nr, nc]), nr, nc))
        else:
            break
    return cells


def _line_to_str(cells: list[tuple[int, int, int]], player: int) -> str:
    """Convert line to pattern string: '1'=player, '2'=opponent, '0'=empty."""
    opp = _other(player)
    lut = {EMPTY: "0", player: "1", opp: "2"}
    return "".join(lut[v] for v, _, _ in cells)


def _empty_cells_in_line(cells: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """Return (row, col) for empty cells in a line."""
    return [(r, c) for v, r, c in cells if v == EMPTY]


# ═══════════════════════════════════════════════════════════
# GROUP 2: OPEN FOUR (patterns 6-10)
# ═══════════════════════════════════════════════════════════

def find_open_fours(grid: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Find open four patterns — immediate win if not blocked.

    Patterns detected (in 5-cell windows):
      P6:  _XXXX_  (both ends open → instant win)
      P7:  _XXXX   (left open, right edge/blocked)
      P8:  XXXX_   (left edge/blocked, right open)
      P9:  _XXX_X  (split four, gap at position 4)
      P10: X_XXX_  (split four, gap at position 1)
    """
    threats: dict[tuple[int, int], int] = {}

    for dr, dc in DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = _extract_line(grid, r, c, dr, dc, 7)
                if len(cells) < 5:
                    continue
                line = _line_to_str(cells, player)

                # Scan with sliding windows of length 5 and 6
                for win_len in (5, 6, 7):
                    for i in range(len(line) - win_len + 1):
                        window = line[i : i + win_len]

                        # P6: _XXXX_ (7-cell window, both ends open)
                        if win_len >= 7 and window[:6] == "011110":
                            for v, nr, nc in [cells[i], cells[i + 6]]:
                                if v == EMPTY:
                                    threats[(nr, nc)] = max(threats.get((nr, nc), 0), 10)

                        # P7: _XXXX (left open, need 5-cell window at boundary)
                        if window[:5] == "01111" and (len(window) < 6 or window[5] != "0"):
                            if window[0] == "0":
                                pos = (cells[i][1], cells[i][2])
                                if grid[pos[0], pos[1]] == EMPTY:
                                    threats[pos] = max(threats.get(pos, 0), 8)

                        # P8: XXXX_ (right open)
                        if window[-5:] == "11110" and (i == 0 or window[0] != "0"):
                            if window[-1] == "0":
                                pos = (cells[i + win_len - 1][1], cells[i + win_len - 1][2])
                                if grid[pos[0], pos[1]] == EMPTY:
                                    threats[pos] = max(threats.get(pos, 0), 8)

                        # P9: _XXX_X (split four with gap)
                        if window == "011101" or window == "111010":
                            for k in range(win_len):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 9)

                        # P10: X_XXX_ (split four with gap)
                        if window == "101110" or window == "011101":
                            for k in range(win_len):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 9)

    # Sort by priority (highest first)
    return [m for m, _ in sorted(threats.items(), key=lambda x: x[1], reverse=True)]


# ═══════════════════════════════════════════════════════════
# GROUP 3: CLOSED FOUR (patterns 11-15)
# ═══════════════════════════════════════════════════════════

def find_closed_fours(grid: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Find closed four patterns — only one blocking move.

    Patterns detected (in 5-6 cell windows):
      P11: OXXXX_  (opponent blocks left)
      P12: _XXXXO  (opponent blocks right)
      P13: OXXXXO  (both ends blocked — already dead, skip)
      P14: XXXX_X  (gap split, blocked one end)
      P15: X_XXXX  (gap split, blocked other end)

    Note: P13 is already dead (both ends blocked) — we skip it.
    We also detect _XXXX and XXXX_ as closed fours when one end is
    at board edge (can't be extended).
    """
    threats: dict[tuple[int, int], int] = {}
    opp = _other(player)

    for dr, dc in DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = _extract_line(grid, r, c, dr, dc, 7)
                if len(cells) < 6:
                    continue
                line = _line_to_str(cells, player)

                for i in range(len(line) - 5):
                    window = line[i : i + 6]

                    # P11: OXXXX_ (opponent on left, open on right)
                    if window[1:5] == "1111" and window[0] == "2" and window[5] == "0":
                        pos = (cells[i + 5][1], cells[i + 5][2])
                        if grid[pos[0], pos[1]] == EMPTY:
                            threats[pos] = max(threats.get(pos, 0), 7)

                    # P12: _XXXXO (open on left, opponent on right)
                    if window[1:5] == "1111" and window[0] == "0" and window[5] == "2":
                        pos = (cells[i][1], cells[i][2])
                        if grid[pos[0], pos[1]] == EMPTY:
                            threats[pos] = max(threats.get(pos, 0), 7)

                    # P14: XXXX_X (4 consecutive + gap, both in 6-window)
                    if window[:4] == "1111" and window[4] == "0" and window[5] == "1":
                        pos = (cells[i + 4][1], cells[i + 4][2])
                        if grid[pos[0], pos[1]] == EMPTY:
                            threats[pos] = max(threats.get(pos, 0), 7)

                    # P15: X_XXXX (gap + 4 consecutive)
                    if window[0] == "1" and window[1] == "0" and window[2:6] == "1111":
                        pos = (cells[i + 1][1], cells[i + 1][2])
                        if grid[pos[0], pos[1]] == EMPTY:
                            threats[pos] = max(threats.get(pos, 0), 7)

    return [m for m, _ in sorted(threats.items(), key=lambda x: x[1], reverse=True)]


# ═══════════════════════════════════════════════════════════
# GROUP 4: OPEN THREE (patterns 16-22)
# ═══════════════════════════════════════════════════════════

def find_open_threes(grid: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Find open three patterns — can become open four next move.

    Patterns detected:
      P16: _XXX_   (consecutive three, both ends open)
      P17: _XX_X_  (split three with gap at 3)
      P18: _X_XX_  (split three with gap at 2)
      P19: _XX_X_  (same as P17, different scan position)
      P20: _X_X_X_ (double split three)
      P21: _XXX__  (three with extra space on right)
      P22: __XXX_  (three with extra space on left)
    """
    threats: dict[tuple[int, int], int] = {}

    for dr, dc in DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = _extract_line(grid, r, c, dr, dc, 7)
                if len(cells) < 7:
                    continue
                line = _line_to_str(cells, player)

                # Scan with 5-cell and 6-cell windows
                for win_len in (5, 6, 7):
                    for i in range(len(line) - win_len + 1):
                        window = line[i : i + win_len]

                        # P16: _XXX_ (5-cell window)
                        if window == "01110":
                            for k in range(5):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 5)

                        # P17/P19: _XX_X_ (6-cell window)
                        if window == "011010" or window == "010110":
                            for k in range(win_len):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 5)

                        # P18: _X_XX_ (6-cell window)
                        if window == "010110":
                            for k in range(6):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 5)

                        # P20: _X_X_X_ (7-cell window)
                        if window == "0101010":
                            for k in range(7):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 4)

                        # P21: _XXX__ (7-cell window)
                        if win_len >= 7 and window[:5] == "01110" and window[5:] == "00":
                            for k in range(5):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 4)

                        # P22: __XXX_ (7-cell window)
                        if win_len >= 7 and window[:2] == "00" and window[2:5] == "111" and window[5] == "0":
                            for k in range(6):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 4)

    return [m for m, _ in sorted(threats.items(), key=lambda x: x[1], reverse=True)]


# ═══════════════════════════════════════════════════════════
# GROUP 5: BROKEN THREE (patterns 23-28)
# ═══════════════════════════════════════════════════════════

def find_broken_threes(grid: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Find broken three patterns — precursors to open threes.

    Patterns detected (in 4-6 cell windows):
      P23: XX_X   (gap at position 3)
      P24: X_XX   (gap at position 1)
      P25: X_X_X  (two gaps)
      P26: XX__X  (double gap)
      P27: X__XX  (double gap)
      P28: X_X__X (triple with gaps)
    """
    threats: dict[tuple[int, int], int] = {}

    for dr, dc in DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = _extract_line(grid, r, c, dr, dc, 6)
                if len(cells) < 4:
                    continue
                line = _line_to_str(cells, player)

                for win_len in (4, 5, 6):
                    for i in range(len(line) - win_len + 1):
                        window = line[i : i + win_len]

                        # P23: XX_X
                        if window == "1101" or window == "1011":
                            for k in range(win_len):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 3)

                        # P24: X_XX (same as P23 reversed in some windows)
                        if window == "1011":
                            for k in range(win_len):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 3)

                        # P25: X_X_X
                        if window == "10101":
                            for k in range(5):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 3)

                        # P26: XX__X
                        if window == "11001":
                            for k in range(5):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 3)

                        # P27: X__XX
                        if window == "10011":
                            for k in range(5):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 3)

                        # P28: X_X__X
                        if window == "101001" or window == "100101":
                            for k in range(6):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 2)

    return [m for m, _ in sorted(threats.items(), key=lambda x: x[1], reverse=True)]


# ═══════════════════════════════════════════════════════════
# GROUP 6: OPEN TWO (patterns 29-33)
# ═══════════════════════════════════════════════════════════

def find_open_twos(grid: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Find open two patterns — foundation for future threats.

    Patterns detected (in 4-5 cell windows):
      P29: _XX_   (consecutive two, both ends open)
      P30: _X_X_  (split two)
      P31: _X__X_ (wide split two)
      P32: __XX_  (two with extra space left)
      P33: _XX__  (two with extra space right)
    """
    threats: dict[tuple[int, int], int] = {}

    for dr, dc in DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = _extract_line(grid, r, c, dr, dc, 6)
                if len(cells) < 4:
                    continue
                line = _line_to_str(cells, player)

                for win_len in (4, 5, 6):
                    for i in range(len(line) - win_len + 1):
                        window = line[i : i + win_len]

                        # P29: _XX_
                        if window == "0110":
                            for k in range(4):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 2)

                        # P30: _X_X_
                        if window == "01010":
                            for k in range(5):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 2)

                        # P31: _X__X_
                        if window == "010010":
                            for k in range(6):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 1)

                        # P32: __XX_
                        if window[:3] == "001" and window[3:] == "10":
                            for k in range(5):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 1)

                        # P33: _XX__
                        if window[:2] == "01" and window[2:4] == "10" and window[4:] == "00":
                            for k in range(5):
                                if window[k] == "0":
                                    pos = (cells[i + k][1], cells[i + k][2])
                                    if grid[pos[0], pos[1]] == EMPTY:
                                        threats[pos] = max(threats.get(pos, 0), 1)

    return [m for m, _ in sorted(threats.items(), key=lambda x: x[1], reverse=True)]


# ═══════════════════════════════════════════════════════════
# GROUP 10: HIDDEN THREATS (patterns 48-50)
# ═══════════════════════════════════════════════════════════

def find_hidden_connections(grid: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Find hidden connection patterns — long-range danger.

    Patterns detected:
      P48: X__X    (2-gap connection, score 2)
      P49: X_X_X   (double-split, score 3)
      P50: X__X__X (triple-gap, score 1)
    """
    threats: dict[tuple[int, int], int] = {}

    for dr, dc in DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = _extract_line(grid, r, c, dr, dc, 7)
                if len(cells) < 5:
                    continue
                line = _line_to_str(cells, player)

                # P48: X__X (5-cell window)
                for i in range(len(line) - 4):
                    window = line[i : i + 5]
                    if window in ("10010", "01001", "10001"):
                        for k in range(5):
                            if window[k] == "0":
                                pos = (cells[i + k][1], cells[i + k][2])
                                if grid[pos[0], pos[1]] == EMPTY:
                                    threats[pos] = max(threats.get(pos, 0), 2)

                # P49: X_X_X (6-cell window)
                for i in range(len(line) - 5):
                    window = line[i : i + 6]
                    if window in ("101010", "010101"):
                        for k in range(6):
                            if window[k] == "0":
                                pos = (cells[i + k][1], cells[i + k][2])
                                if grid[pos[0], pos[1]] == EMPTY:
                                    threats[pos] = max(threats.get(pos, 0), 3)

                # P50: X__X__X (7-cell window)
                for i in range(len(line) - 6):
                    window = line[i : i + 7]
                    if window in ("1001001", "0010010", "10010010"):
                        for k in range(7):
                            if window[k] == "0":
                                pos = (cells[i + k][1], cells[i + k][2])
                                if grid[pos[0], pos[1]] == EMPTY:
                                    threats[pos] = max(threats.get(pos, 0), 1)

    return [m for m, _ in sorted(threats.items(), key=lambda x: x[1], reverse=True)]


# ═══════════════════════════════════════════════════════════
# GROUP 7: DOUBLE THREAT / FORK (patterns 34-38)
# ═══════════════════════════════════════════════════════════

def count_threats_at_position(
    grid: np.ndarray, row: int, col: int, player: int
) -> int:
    """Count how many threat directions a move creates for player.

    A threat = at least 3 in a row with room to grow to 5.
    Returns count of directions with threats.
    """
    if grid[row, col] != EMPTY:
        return 0

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


def find_fork_moves(grid: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Find moves that create 2+ threat directions (fork/double threat).

    Covers:
      P34: Double open three (2+ open threes created)
      P35: Open three + open four
      P36: Two open fours
      P37: Closed four + open three
      P38: Fork in 3+ directions
    """
    fork_moves: list[tuple[int, int]] = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] != EMPTY:
                continue
            threats = count_threats_at_position(grid, r, c, player)
            if threats >= 2:
                fork_moves.append((r, c))

    return fork_moves


# ═══════════════════════════════════════════════════════════
# GROUP 8: FORK SHAPES (patterns 39-43)
# ═══════════════════════════════════════════════════════════

def find_fork_shapes(grid: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Find fork shape patterns — multi-directional attacks.

    Patterns detected:
      P39: Cross Fork   — stone creates threats in 3+ directions (cross shape)
      P40: T Fork       — stone creates threats in 2 perpendicular directions (T shape)
      P41: L Fork       — stone creates threats in 2 adjacent directions (L shape)
      P42: Diagonal Fork — stone creates diagonal + straight threats
      P43: Double Diagonal Fork — threats on both diagonals
    """
    fork_moves: dict[tuple[int, int], int] = {}

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if grid[r, c] != EMPTY:
                continue

            # Count threats per direction
            dir_threats: list[tuple[int, int, int]] = []  # (dr, dc, streak)
            for dr, dc in DIRECTIONS:
                streak = 1
                r2, c2 = r + dr, c + dc
                while 0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE and grid[r2, c2] == player:
                    streak += 1
                    r2 += dr
                    c2 += dc
                r2, c2 = r - dr, c - dc
                while 0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE and grid[r2, c2] == player:
                    streak += 1
                    r2 -= dr
                    c2 -= dc
                if streak >= 3:
                    dir_threats.append((dr, dc, streak))

            if len(dir_threats) >= 3:
                # P39: Cross Fork (3+ directions)
                fork_moves[(r, c)] = 4
            elif len(dir_threats) >= 2:
                dr1, dc1, s1 = dir_threats[0]
                dr2, dc2, s2 = dir_threats[1]

                # P42/P43: Diagonal Fork (at least one diagonal direction)
                is_diag1 = (dr1, dc1) in ((1, 1), (1, -1))
                is_diag2 = (dr2, dc2) in ((1, 1), (1, -1))
                if is_diag1 and is_diag2:
                    fork_moves[(r, c)] = max(fork_moves.get((r, c), 0), 4)
                elif is_diag1 or is_diag2:
                    # P42: Diagonal + straight
                    fork_moves[(r, c)] = max(fork_moves.get((r, c), 0), 3)
                else:
                    # P40: T Fork (perpendicular straight directions)
                    if (dr1, dc1) != (-dr2, -dc2):  # not opposite
                        fork_moves[(r, c)] = max(fork_moves.get((r, c), 0), 3)
                    # P41: L Fork (adjacent)
                    fork_moves[(r, c)] = max(fork_moves.get((r, c), 0), 2)

    return [m for m, _ in sorted(fork_moves.items(), key=lambda x: x[1], reverse=True)]


# ═══════════════════════════════════════════════════════════
# GROUP 9: DEFENSIVE (patterns 44-47)
# ═══════════════════════════════════════════════════════════

def find_critical_threats(
    grid: np.ndarray, opponent: int
) -> list[tuple[int, int]]:
    """Find moves that opponent MUST play to win immediately.

    Detects: XX_XX, X_XXX, XXX_X, XXXX_ (split/hidden 4).
    These are the highest-priority defensive moves.
    """
    threats: set[tuple[int, int]] = set()

    for dr, dc in DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = _extract_line(grid, r, c, dr, dc, 5)
                if len(cells) < 5:
                    continue
                vals = [v for v, _, _ in cells]
                if vals.count(opponent) == 4 and vals.count(EMPTY) == 1:
                    for v, nr, nc in cells:
                        if v == EMPTY:
                            threats.add((nr, nc))

    return list(threats)


def find_tempo_moves(
    grid: np.ndarray, player: int
) -> list[tuple[int, int]]:
    """Find moves that simultaneously block opponent AND create player threat.

    These are the most efficient moves — gaining tempo by doing two things
    at once instead of just blocking (tempo loss) or just attacking.
    """
    opponent = _other(player)
    tempo_moves: list[tuple[int, int]] = []

    # Get opponent's threats and our offensive opportunities
    opp_critical = set(find_critical_threats(grid, opponent))
    opp_open_fours = set(find_open_fours(grid, opponent))
    opp_forks = set(find_fork_moves(grid, opponent))
    blocking_moves = opp_critical | opp_open_fours | opp_forks

    my_forks = set(find_fork_moves(grid, player))
    my_connections = set(find_hidden_connections(grid, player))

    # Tempo move = blocks opponent AND creates our own threat
    for move in blocking_moves:
        if move in my_forks or move in my_connections:
            tempo_moves.append(move)

    return tempo_moves


# ═══════════════════════════════════════════════════════════
# SCORING: Unified move scoring based on priority table
# ═══════════════════════════════════════════════════════════

# Priority scores matching the user's specification
SCORES = {
    "FIVE": 100_000,
    "DOUBLE_OPEN_FOUR": 50_000,
    "OPEN_FOUR": 10_000,
    "OPEN_THREE_PLUS_FOUR": 8_000,
    "DOUBLE_OPEN_THREE": 5_000,
    "CLOSED_FOUR": 2_000,
    "OPEN_THREE": 1_000,
    "BROKEN_THREE": 300,
    "OPEN_TWO": 100,
    "SINGLE_STONE": 10,
}


def score_position_for_player(
    grid: np.ndarray, row: int, col: int, player: int
) -> int:
    """Score a move for player based on the priority table.

    Returns the score based on the BEST pattern this move creates.
    Higher score = more threatening.
    """
    if grid[row, col] != EMPTY:
        return 0

    # Temporarily place the stone
    grid[row, col] = player
    opp = _other(player)

    # Check what patterns this move creates
    score = 0

    # P1-5: Five in a row
    from src.game.rules import is_win
    if is_win(grid, player, last_move=(row, col)):
        grid[row, col] = EMPTY
        return SCORES["FIVE"]

    # Count open threes and fours created by this move
    open_threes_created = 0
    fours_created = 0

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
            fours_created += 1
        elif streak >= 3:
            # Check if it's open (both ends available)
            r1, c1 = row + dr * streak, col + dc * streak
            r2, c2 = row - dr, col - dc
            open_ends = 0
            if 0 <= r1 < BOARD_SIZE and 0 <= c1 < BOARD_SIZE and grid[r1, c1] == EMPTY:
                open_ends += 1
            if 0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE and grid[r2, c2] == EMPTY:
                open_ends += 1
            if open_ends >= 2:
                open_threes_created += 1

    # Fork moves (2+ threat directions)
    threat_dirs = count_threats_at_position(grid, row, col, player)

    # Score based on what was created
    if fours_created >= 2:
        score = SCORES["DOUBLE_OPEN_FOUR"]
    elif fours_created >= 1 and open_threes_created >= 1:
        score = SCORES["OPEN_THREE_PLUS_FOUR"]
    elif open_threes_created >= 2:
        score = SCORES["DOUBLE_OPEN_THREE"]
    elif fours_created >= 1:
        score = SCORES["OPEN_FOUR"]
    elif open_threes_created >= 1:
        score = SCORES["OPEN_THREE"]
    elif threat_dirs >= 2:
        score = SCORES["DOUBLE_OPEN_THREE"]
    else:
        # Check broken threes and open twos
        broken_threes = len(find_broken_threes(grid, player))
        open_twos = len(find_open_twos(grid, player))
        if broken_threes > 0:
            score = SCORES["BROKEN_THREE"]
        elif open_twos > 0:
            score = SCORES["OPEN_TWO"]
        else:
            score = SCORES["SINGLE_STONE"]

    grid[row, col] = EMPTY
    return score


# ═══════════════════════════════════════════════════════════
# UTILITY: Combined threat finder
# ═══════════════════════════════════════════════════════════

def edge_distance_bonus(row: int, col: int) -> float:
    """Bonus for positions closer to center, penalty for edges."""
    center = BOARD_SIZE // 2
    dist = max(abs(row - center), abs(col - center))
    return max(0.0, 4.0 - dist) * 0.5


def find_all_threats(
    grid: np.ndarray, player: int
) -> list[tuple[int, int]]:
    """Find all threats for player, prioritized by severity.

    Priority:
    1. Critical (4-in-a-row with gap) -> immediate loss
    2. Open four (_XXXX_) -> almost certain loss
    3. Closed four -> dangerous
    4. Fork moves (2+ threat directions) -> very dangerous
    5. Tempo moves (block + create own threat) -> most efficient
    6. Open three -> dangerous
    7. Broken three -> moderately dangerous
    8. Hidden connections -> long-range danger
    9. Open two -> foundation
    """
    opponent = _other(player)
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []

    # Layer 1: must-block (critical four-in-a-row with gap)
    for move in find_critical_threats(grid, opponent):
        if move not in seen and grid[move[0], move[1]] == EMPTY:
            seen.add(move)
            result.append(move)

    # Layer 2: open four
    for move in find_open_fours(grid, opponent):
        if move not in seen and grid[move[0], move[1]] == EMPTY:
            seen.add(move)
            result.append(move)

    # Layer 3: closed four
    for move in find_closed_fours(grid, opponent):
        if move not in seen and grid[move[0], move[1]] == EMPTY:
            seen.add(move)
            result.append(move)

    # Layer 4: fork (opponent creates 2+ threats)
    for move in find_fork_moves(grid, opponent):
        if move not in seen and grid[move[0], move[1]] == EMPTY:
            seen.add(move)
            result.append(move)

    # Layer 5: tempo moves (block + create own threat)
    for move in find_tempo_moves(grid, player):
        if move not in seen and grid[move[0], move[1]] == EMPTY:
            seen.add(move)
            result.append(move)

    # Layer 6: open three
    for move in find_open_threes(grid, opponent):
        if move not in seen and grid[move[0], move[1]] == EMPTY:
            seen.add(move)
            result.append(move)

    # Layer 7: broken three
    for move in find_broken_threes(grid, opponent):
        if move not in seen and grid[move[0], move[1]] == EMPTY:
            seen.add(move)
            result.append(move)

    # Layer 8: hidden connections
    for move in find_hidden_connections(grid, opponent):
        if move not in seen and grid[move[0], move[1]] == EMPTY:
            seen.add(move)
            result.append(move)

    # Layer 9: open two
    for move in find_open_twos(grid, opponent):
        if move not in seen and grid[move[0], move[1]] == EMPTY:
            seen.add(move)
            result.append(move)

    return result


def find_offensive_forks(
    grid: np.ndarray, player: int
) -> list[tuple[int, int]]:
    """Find moves that create 2+ threats for the player (offensive forks)."""
    return find_fork_moves(grid, player)


# ═══════════════════════════════════════════════════════════
# MCTS BOOST: Compute policy boost multipliers for all moves
# ═══════════════════════════════════════════════════════════

def compute_threat_boosts(
    grid: np.ndarray, player: int
) -> np.ndarray:
    """Compute per-move policy boost multipliers based on threat analysis.

    Returns array of shape (BOARD_SIZE * BOARD_SIZE,) with multipliers.
    Higher multiplier = more urgent move.
    """
    opponent = _other(player)
    boosts = np.ones(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)

    # Defensive: must-block threats from opponent
    for move in find_critical_threats(grid, opponent):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 10.0)

    for move in find_open_fours(grid, opponent):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 8.0)

    for move in find_closed_fours(grid, opponent):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 6.0)

    for move in find_fork_moves(grid, opponent):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 5.0)

    # Offensive: our own attacking moves
    for move in find_fork_moves(grid, player):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 4.5)

    for move in find_offensive_forks(grid, player):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 4.0)

    for move in find_tempo_moves(grid, player):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 3.5)

    for move in find_open_threes(grid, opponent):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 3.0)

    for move in find_broken_threes(grid, opponent):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 2.0)

    for move in find_hidden_connections(grid, opponent):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 1.5)

    for move in find_open_twos(grid, opponent):
        idx = move[0] * BOARD_SIZE + move[1]
        boosts[idx] = max(boosts[idx], 1.2)

    return boosts
