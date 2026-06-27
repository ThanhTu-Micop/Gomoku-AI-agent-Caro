import argparse
import os
import sys
import time
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.game.board import Board
from src.game.rules import is_win, is_draw
from src.game.constants import X, O
from src.ai.minimax import MinimaxAgent
from src.ai.rl_agent import AlphaZeroAgent, RLAgent
from src.utils.logger import log_match, log_game_replay


def get_game_stage(move_count: int) -> str:
    if move_count < 10:
        return "early"
    elif move_count < 30:
        return "mid"
    else:
        return "end"


def play_match(
    ai1,
    ai2,
    ai1_name: str,
    ai2_name: str,
    match_id: int,
    log_replay: bool = False,
) -> None:
    board = Board()
    current_player = X
    move_count = 0
    times_by_stage = {"early": [], "mid": [], "end": []}
    moves_history = []

    winner = None

    while True:
        stage = get_game_stage(move_count)
        start_time = time.time()

        agent = ai1 if current_player == X else ai2
        if isinstance(agent, (AlphaZeroAgent, RLAgent)):
            move = agent.get_move(board.grid.copy(), current_player, deterministic=True)
        else:
            move = agent.get_move(board.grid.copy(), current_player)
        if move is None:
            break

        r, c = move
        board.place(r, c, current_player)
        moves_history.append((current_player, r, c))
        move_count += 1

        elapsed = time.time() - start_time
        times_by_stage[stage].append(elapsed)

        if is_win(board.grid, current_player, last_move=(r, c)):
            winner = current_player
            break
        elif is_draw(board.grid):
            winner = None
            break

        current_player = O if current_player == X else X

    log_match(
        match_id=match_id,
        ai1_name=ai1_name,
        ai2_name=ai2_name,
        winner=winner,
        move_count=move_count,
        times_by_stage=times_by_stage,
    )

    if log_replay:
        log_game_replay(match_id=match_id, moves=moves_history, winner=winner)

    winner_name = "Draw" if winner is None else (ai1_name if winner == X else ai2_name)
    print(f"Match {match_id}: {winner_name} wins in {move_count} moves")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Minimax vs RL agent")
    parser.add_argument("--matches", type=int, default=10, help="Number of matches to play")
    parser.add_argument("--start-id", type=int, default=0, help="Match ID offset")
    parser.add_argument("--depth", type=int, default=3, help="Minimax search depth")
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["rl", "alphazero"],
        default="alphazero",
        help="Opponent agent type (rl=fast policy net, alphazero=MCTS)",
    )
    parser.add_argument("--rl-model", type=str, default=None, help="RL model checkpoint path")
    parser.add_argument("--mcts-sims", type=int, default=80, help="MCTS simulations for AlphaZero")
    parser.add_argument("--c-puct", type=float, default=1.4, help="PUCT exploration coefficient")
    parser.add_argument("--num-res-blocks", type=int, default=5, help="ResNet residual blocks")
    parser.add_argument("--channels", type=int, default=64, help="ResNet channel width")
    parser.add_argument("--log-replay", action="store_true", help="Log game replays to JSONL")
    args = parser.parse_args()

    ai1 = MinimaxAgent(depth=args.depth)
    ai1_name = f"Minimax(d={args.depth})"

    if args.agent_type == "rl":
        ai2 = RLAgent()
        ai2_name = "RL Agent"
        if args.rl_model:
            ai2.load(args.rl_model)
            print(f"Loaded RL model from {args.rl_model}")
        else:
            print("Warning: --rl-model not provided. Using untrained RL Agent.")
    else:
        ai2 = AlphaZeroAgent(
            num_simulations=args.mcts_sims,
            c_puct=args.c_puct,
            num_res_blocks=args.num_res_blocks,
            channels=args.channels,
        )
        ai2_name = f"AlphaZero(sims={args.mcts_sims})"
        if args.rl_model:
            ai2.load(args.rl_model)
            print(f"Loaded RL model from {args.rl_model}")
        else:
            print("Warning: --rl-model not provided. Using untrained AlphaZero Agent.")

    print(f"Starting {args.matches} matches: {ai1_name} vs {ai2_name}")

    for i in range(args.matches):
        if i % 2 == 0:
            play_match(ai1, ai2, ai1_name, ai2_name, args.start_id + i + 1, log_replay=args.log_replay)
        else:
            play_match(ai2, ai1, ai2_name, ai1_name, args.start_id + i + 1, log_replay=args.log_replay)

    print("Comparison complete. Results saved to logs/matches.csv")


if __name__ == "__main__":
    main()
