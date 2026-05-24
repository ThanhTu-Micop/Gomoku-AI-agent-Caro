import numpy as np
from src.ai.minimax import MinimaxAgent
from src.game.constants import BOARD_SIZE, EMPTY, X, O

def test_candidate_expansion() -> None:
    agent = MinimaxAgent()
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=int)
    
    # Đặt 1 quân tại giữa (4,4)
    grid[4, 4] = X
    
    candidates = agent._get_candidate_moves(grid)
    
    # Kiểm tra ô (2,2) - cách (4,4) 2 đơn vị
    assert (2, 2) in candidates, "Should include (2,2) in candidates"
    assert (6, 6) in candidates, "Should include (6,6) in candidates"
    assert (4, 2) in candidates, "Should include (4,2) in candidates"
    
    # Kiểm tra ô (1,1) - cách (4,4) 3 đơn vị (không nên có)
    assert (1, 1) not in candidates, "Should not include (1,1) (too far)"
    
    print("Candidate expansion test passed!")

if __name__ == "__main__":
    test_candidate_expansion()
