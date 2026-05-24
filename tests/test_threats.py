import numpy as np
from src.ai.heuristic import evaluate
from src.game.constants import BOARD_SIZE, EMPTY, X, O

def test_split_four_threat() -> None:
    # Test mẫu X.XXX (split four)
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, 0] = X
    grid[0, 2] = X
    grid[0, 3] = X
    grid[0, 4] = X
    
    score = evaluate(grid, X)
    assert score >= 10000, f"Split four score should be high, got {score}"

def test_split_four_center_threat() -> None:
    # Test mẫu XX.XX (split four)
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    grid[0, 0] = X
    grid[0, 1] = X
    grid[0, 3] = X
    grid[0, 4] = X
    
    score = evaluate(grid, X)
    assert score >= 10000, f"Split four center score should be high, got {score}"

def test_double_three_threat() -> None:
    # Test 2 đường 3 hở
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    # Đường 1: hàng 0
    grid[0, 2:5] = X # .XXX.
    # Đường 2: cột 2
    grid[2, 2] = X
    grid[3, 2] = X
    grid[4, 2] = X # Cột 2 có XXX
    
    # Lưu ý: Cần đảm bảo các đầu là EMPTY (đã mặc định)
    score = evaluate(grid, X)
    # FOUR_OPEN score is 100000. Double three should be around that.
    assert score >= 80000, f"Double three score should be very high, got {score}"

if __name__ == "__main__":
    test_split_four_threat()
    test_split_four_center_threat()
    test_double_three_threat()
    print("All split threat tests passed!")
