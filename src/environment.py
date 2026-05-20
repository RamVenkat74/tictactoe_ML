from __future__ import annotations
from typing import Optional


EMPTY = 0
X = 1
O = 2

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

REWARDS = {
    "agent_win":  1.0,
    "agent_loss": -1.0,
    "draw":        0.5,
    "step":        0.0,
}


class TicTacToe:
    """
    Tic-Tac-Toe game state.

    Usage:
        env = TicTacToe()
        env.reset()
        state = env.state_key()
        available = env.available_moves()
        reward, done = env.step(action, player=O)
    """

    def __init__(self):
        self.board: list[int] = [EMPTY] * 9
        self.done: bool = False
        self.winner: Optional[int] = None
        self.winning_line: Optional[tuple] = None

    def reset(self) -> "TicTacToe":
        self.board = [EMPTY] * 9
        self.done = False
        self.winner = None
        self.winning_line = None
        return self

    def state_key(self) -> str:
        """Compact string representation used as Q-table key."""
        return "".join(map(str, self.board))

    def available_moves(self) -> list[int]:
        return [i for i, v in enumerate(self.board) if v == EMPTY]

    def is_full(self) -> bool:
        return all(v != EMPTY for v in self.board)

    def check_winner(self) -> Optional[int]:
        """Return winning player (1 or 2), or None if no winner yet."""
        for a, b, c in WIN_LINES:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                self.winning_line = (a, b, c)
                return self.board[a]
        return None

    def step(self, action: int, player: int) -> tuple[float, bool]:

        if self.done:
            raise RuntimeError("Game already over. Call reset().")
        if self.board[action] != EMPTY:
            raise ValueError(f"Cell {action} is already taken.")

        self.board[action] = player
        winner = self.check_winner()

        if winner == O:
            self.done, self.winner = True, O
            return REWARDS["agent_win"], True

        if winner == X:
            self.done, self.winner = True, X
            return REWARDS["agent_loss"], True

        if self.is_full():
            self.done, self.winner = True, None
            return REWARDS["draw"], True

        return REWARDS["step"], False

    SYMBOLS = {EMPTY: ".", X: "X", O: "O"}

    def render(self) -> str:
        rows = []
        for r in range(3):
            cells = [self.SYMBOLS[self.board[r * 3 + c]] for c in range(3)]
            rows.append(" | ".join(cells))
        return ("\n" + "-" * 9 + "\n").join(rows)

    def __repr__(self) -> str:
        return f"TicTacToe(done={self.done}, winner={self.SYMBOLS[self.winner] if self.winner is not None else '?'})"
