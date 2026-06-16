"""Quick benchmark: test multiple sims/depth combos for RL:Minimax ratio."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.board import Board
from src.game.rules import is_win, is_draw
from src.game.constants import X, O
from src.ai.minimax import MinimaxAgent
from src.ai.rl_agent import AlphaZeroAgent
import time


def play_one_match(ai1, ai2, ai1_name, ai2_name):
    board = Board()
    current_player = X
    move_count = 0

    while True:
        agent = ai1 if current_player == X else ai2
        if isinstance(agent, AlphaZeroAgent):
            move = agent.get_move(board.grid.copy(), current_player, deterministic=True)
        else:
            move = agent.get_move(board.grid.copy(), current_player)
        if move is None:
            break
        r, c = move
        board.place(r, c, current_player)
        move_count += 1

        if is_win(board.grid, current_player, last_move=(r, c)):
            return current_player, move_count
        elif is_draw(board.grid):
            return None, move_count

        current_player = O if current_player == X else X

    return None, move_count


def benchmark(sims, depth, num_matches=20):
    rl_wins = 0
    mm_wins = 0
    draws = 0
    total_moves = 0

    rl = AlphaZeroAgent(
        num_simulations=sims, c_puct=1.4,
        num_res_blocks=5, channels=64,
    )
    rl.load("models/rl_agent.pth")
    mm = MinimaxAgent(depth=depth)

    for i in range(num_matches):
        if i % 2 == 0:
            winner, moves = play_one_match(rl, mm, f"RL({sims})", f"MM(d={depth})")
        else:
            winner, moves = play_one_match(mm, rl, f"MM(d={depth})", f"RL({sims})")

        total_moves += moves
        if winner == X:
            # X wins; in even matches rl=X, odd matches mm=X
            if i % 2 == 0:
                rl_wins += 1
            else:
                mm_wins += 1
        elif winner == O:
            if i % 2 == 0:
                mm_wins += 1
            else:
                rl_wins += 1
        else:
            draws += 1

    avg_moves = total_moves / num_matches
    rl_pct = rl_wins / num_matches * 100
    mm_pct = mm_wins / num_matches * 100
    return rl_wins, mm_wins, draws, avg_moves, rl_pct, mm_pct


if __name__ == "__main__":
    configs = [
        (20, 2),
        (40, 2),
        (60, 2),
        (80, 2),
        (20, 3),
        (40, 3),
        (60, 3),
        (80, 3),
        (100, 3),
        (80, 4),
    ]

    print(f"{'Sims':>5} {'Depth':>5} | {'RL':>3} {'MM':>3} {'Draw':>4} | {'RL%':>5} {'MM%':>5} | {'AvgMoves':>8}")
    print("-" * 65)

    for sims, depth in configs:
        rl_w, mm_w, dr, avg_m, rl_p, mm_p = benchmark(sims, depth, num_matches=20)
        print(f"{sims:>5} {depth:>5} | {rl_w:>3} {mm_w:>3} {dr:>4} | {rl_p:>5.1f}% {mm_p:>5.1f}% | {avg_m:>8.1f}")
