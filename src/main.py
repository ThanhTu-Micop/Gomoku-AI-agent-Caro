import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.gui import main
from src.ui.gui import HumanAgent
from src.ai.minimax import MinimaxAgent
from src.ai.rl_agent import AlphaZeroAgent, RLAgent


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
        "--ai",
        choices=["minimax", "rl", "alphazero"],
        default="minimax",
        help="AI type used in hvai mode and as ai2 in aivai",
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default=None,
        help="Path to RL model checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to AI model checkpoint (alias of --rl-model)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=200,
        help="MCTS simulations for AlphaZero",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration coefficient for AlphaZero",
    )
    return parser.parse_args()


def build_ai(args: argparse.Namespace):
    model_path = args.model or args.rl_model
    if args.ai == "minimax":
        return MinimaxAgent(depth=args.depth)

    if args.ai == "rl":
        agent = RLAgent()
        if model_path:
            agent.load(model_path)
        return agent

    agent = AlphaZeroAgent(num_simulations=args.mcts_sims, c_puct=args.c_puct)
    if model_path:
        agent.load(model_path)
    return agent


def main_entry() -> None:
    args = parse_args()

    if args.mode == "hvh":
        p1 = HumanAgent()
        p2 = HumanAgent()
    elif args.mode == "hvai":
        p1 = HumanAgent()
        p2 = build_ai(args)
    else:
        p1 = MinimaxAgent(depth=args.depth)
        p2 = build_ai(args)

    main(player1=p1, player2=p2)


if __name__ == "__main__":
    main_entry()
