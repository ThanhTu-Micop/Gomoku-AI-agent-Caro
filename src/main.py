import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.gui import main
from src.ui.gui import HumanAgent
from src.ai.minimax import MinimaxAgent
from src.ai.rl_agent import RLAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gomoku AI Agent")
    parser.add_argument(
        "--mode",
        choices=["hvh", "hvai", "aivai"],
        default="hvh",
        help="Game mode: hvh (human vs human), hvai (human vs AI), aivai (AI vs AI)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Search depth for Minimax agent",
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default=None,
        help="Path to RL model checkpoint",
    )
    return parser.parse_args()


def main_entry() -> None:
    args = parse_args()

    if args.mode == "hvh":
        p1 = HumanAgent()
        p2 = HumanAgent()
    elif args.mode == "hvai":
        p1 = HumanAgent()
        p2 = MinimaxAgent(depth=args.depth)
    else:
        p1 = MinimaxAgent(depth=args.depth)
        if args.rl_model:
            p2 = RLAgent()
            p2.load(args.rl_model)
        else:
            p2 = MinimaxAgent(depth=args.depth)

    main(player1=p1, player2=p2)


if __name__ == "__main__":
    main_entry()
