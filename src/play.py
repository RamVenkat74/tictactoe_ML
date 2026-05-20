"""
Play against the trained Q-Learning agent in the terminal.
============================================================
Run:
    python src/play.py                    # loads q_table.pkl
    python src/play.py --model my.pkl     # loads custom file
    python src/play.py --train-first      # trains then plays

You are X (always go first).  Agent is O.
"""

import argparse
import os
import sys

from environment import TicTacToe, X, O
from q_agent import QLearningAgent


BANNER = r"""
  ___  _   ___            _____            ___        _
 |   \(_) / __|     ___  |_   _|__ ___   |   \  ___ | |__  ___
 | |) || || (__    |___|   | | / _// -_)  | |) |/ _ \| '_ \(_-<
 |___/ |_| \___|          |_| \__\\___|  |___/ \___/|_.__//__/

         Q-Learning Agent  ·  Reinforcement Learning Demo
"""


def print_board(env: TicTacToe):
    print()
    lines = env.render().split("\n")
    # show cell indices alongside the board for easy input
    idx = 0
    for line in lines:
        if "-" in line:
            print(" " * 14 + line)
        else:
            cells = line.split(" | ")
            nums = [str(idx + c) if env.board[idx + c] == 0 else " "
                    for c in range(3)]
            print(f"  [{' | '.join(nums)}]    {line}")
            idx += 3
    print()


def get_player_move(available: list) -> int:
    while True:
        try:
            move = int(input(f"  Your move (0-8, available: {available}): "))
            if move in available:
                return move
            print("  ✗ Cell taken or out of range. Try again.")
        except ValueError:
            print("  ✗ Please enter a number.")
        except KeyboardInterrupt:
            print("\n  Goodbye!")
            sys.exit(0)


def play_game(agent: QLearningAgent) -> str:
    """Play one game. Returns 'X', 'O', or 'draw'."""
    env = TicTacToe()
    symbols = {1: "X", 2: "O", None: "—"}

    print_board(env)

    while not env.done:
        # --- Human turn (X) ---
        available = env.available_moves()
        human_move = get_player_move(available)
        env.step(human_move, player=X)
        print_board(env)
        if env.done:
            break

        # --- Agent turn (O) ---
        state = env.state_key()
        o_moves = env.available_moves()
        if not o_moves:
            break
        action = agent.choose_action(state, o_moves, greedy=True)
        env.step(action, player=O)
        print(f"  Agent (O) plays cell {action}")
        print_board(env)

    # Result
    if env.winner == X:
        return "X"
    elif env.winner == O:
        return "O"
    return "draw"


def main():
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe vs Q-agent")
    parser.add_argument("--model",       default="q_table.pkl")
    parser.add_argument("--train-first", action="store_true",
                        help="Train a fresh agent before playing")
    args = parser.parse_args()

    print(BANNER)

    agent = QLearningAgent()

    if args.train_first or not os.path.exists(args.model):
        print("  No saved model found — training first (20 000 episodes)...\n")
        from train import train
        agent = train(episodes=20_000, save_path=args.model, print_every=5000)
    else:
        agent.load(args.model)

    wins = losses = draws = 0

    while True:
        print("─" * 50)
        result = play_game(agent)

        if result == "X":
            wins += 1
            print("  🎉  You win!  The agent still has more to learn.\n")
        elif result == "O":
            losses += 1
            print("  🤖  Agent wins!  Try a different strategy.\n")
        else:
            draws += 1
            print("  🤝  It's a draw!\n")

        print(f"  Score  →  You: {wins}   Agent: {losses}   Draws: {draws}")

        again = input("\n  Play again? [Y/n]: ").strip().lower()
        if again in ("n", "no", "q", "quit"):
            print("\n  Thanks for playing!\n")
            break


if __name__ == "__main__":
    main()
