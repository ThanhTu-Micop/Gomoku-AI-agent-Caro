import numpy as np
from src.game.constants import BOARD_SIZE, EMPTY, X, O

# Trọng số cho các mẫu cờ
# P: Player, E: Empty, O: Opponent
# 5: Thắng tuyệt đối
# 4_OPEN: 4 quân hở 2 đầu (nước đi sau chắc chắn thắng)
# 4_BLOCKED/SPLIT: 4 quân bị chặn 1 đầu hoặc có lỗ hồng (cần nước đi ngay lập tức để thắng)
# 3_OPEN: 3 quân hở 2 đầu (tạo threat cực mạnh)
# 3_SPLIT: 3 quân có lỗ hổng hở 2 đầu

SCORES = {
    'FIVE': 1000000,
    'FOUR_OPEN': 100000,
    'FOUR_BLOCKED': 10000,
    'THREE_OPEN': 5000,
    'THREE_BLOCKED': 500,
    'TWO_OPEN': 100,
    'TWO_BLOCKED': 10,
}

def _get_all_lines(grid: np.ndarray) -> list[list[int]]:
    lines = []
    # Rows
    for r in range(BOARD_SIZE):
        lines.append(grid[r, :].tolist())
    # Columns
    for c in range(BOARD_SIZE):
        lines.append(grid[:, c].tolist())
    # Diagonals
    for d in range(-BOARD_SIZE + 5, BOARD_SIZE - 4):
        lines.append(np.diagonal(grid, offset=d).tolist())
        lines.append(np.diagonal(np.fliplr(grid), offset=d).tolist())
    return lines

def _evaluate_line(line: list[int], player: int) -> tuple[float, int, int]:
    """Trả về (score, num_open_threes, num_fours)"""
    score = 0.0
    num_open_threes = 0
    num_fours = 0
    opponent = O if player == X else X
    n = len(line)
    if n < 5:
        return 0, 0, 0

    # Chuyển line thành chuỗi để dễ dùng regex hoặc pattern matching
    # 0: Empty, 1: Player, 2: Opponent (chuẩn hóa về 0, 1, 2)
    s = ""
    for cell in line:
        if cell == EMPTY: s += "0"
        elif cell == player: s += "1"
        else: s += "2"

    # Nhận diện các mẫu cờ
    # 5 quân
    if "11111" in s:
        score += SCORES['FIVE']
        return score, 0, 0 # Thắng ngay

    # 4 quân
    # 4 hở 2 đầu: 011110
    if "011110" in s:
        score += SCORES['FOUR_OPEN']
        num_fours += 1
    
    # 4 bị chặn 1 đầu: 211110, 011112, hoặc nằm ở mép
    # 4 nhảy: 10111, 11011, 11101
    fours_blocked = ["211110", "011112", "10111", "11011", "11101"]
    if s.startswith("11110"): score += SCORES['FOUR_BLOCKED']; num_fours += 1
    if s.endswith("01111"): score += SCORES['FOUR_BLOCKED']; num_fours += 1
    for p in fours_blocked:
        if p in s:
            score += SCORES['FOUR_BLOCKED']
            num_fours += 1

    # 3 quân
    # 3 hở 2 đầu: 01110, 010110, 011010
    open_threes = ["01110", "010110", "011010"]
    for p in open_threes:
        if p in s:
            score += SCORES['THREE_OPEN']
            num_open_threes += 1
    
    # 3 bị chặn: 211100, 001112, 1011, 1101, ...
    # (Chỉ liệt kê một số mẫu tiêu biểu để tránh quá nặng)
    blocked_threes = ["211100", "001112", "210110", "011012"]
    for p in blocked_threes:
        if p in s:
            score += SCORES['THREE_BLOCKED']
            
    # 2 quân hở
    if "0110" in s or "01010" in s:
        score += SCORES['TWO_OPEN']

    return score, num_open_threes, num_fours

def evaluate(grid: np.ndarray, player: int) -> float:
    player_score = 0.0
    opponent_score = 0.0
    opponent = O if player == X else X
    
    p_open_threes = 0
    p_fours = 0
    o_open_threes = 0
    o_fours = 0
    
    lines = _get_all_lines(grid)
    for line in lines:
        ps, pot, pf = _evaluate_line(line, player)
        os, oot, of = _evaluate_line(line, opponent)
        
        player_score += ps
        opponent_score += os
        p_open_threes += pot
        p_fours += pf
        o_open_threes += oot
        o_fours += of
        
    # Thưởng cho Double Threat (2 đường 3 mở hoặc 1 hở 3 + 1 bốn)
    if p_open_threes >= 2 or (p_open_threes >= 1 and p_fours >= 1):
        player_score += SCORES['FOUR_OPEN'] * 0.8
    
    if o_open_threes >= 2 or (o_open_threes >= 1 and o_fours >= 1):
        opponent_score += SCORES['FOUR_OPEN'] * 0.9 # Ưu tiên chặn đối phương (phòng thủ mạnh hơn)
        
    return player_score - opponent_score
