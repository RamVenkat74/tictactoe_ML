"""Graphical Tic-Tac-Toe — click to play against the Q-Learning agent.

Run from the project root:
    python src/gui.py
    python src/gui.py --model src/q_table.pkl
"""

import argparse
import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(__file__))

from environment import TicTacToe, X, O
from q_agent import QLearningAgent


# ── Palette (Catppuccin Mocha) ────────────────────────────────────────────
BG_COLOR    = "#1e1e2e"
CELL_NORMAL = "#313244"
CELL_HOVER  = "#45475a"
CELL_WIN    = "#a6e3a1"
X_COLOR     = "#f38ba8"
O_COLOR     = "#89b4fa"
TEXT_COLOR  = "#cdd6f4"
DIM_COLOR   = "#6c7086"
BTN_COLOR   = "#89b4fa"
BTN_FG      = "#1e1e2e"

CELL_SIZE  = 130
GAP        = 10
BOARD_PAD  = 16

FONT_MARK  = ("Segoe UI", 60, "bold")
FONT_TITLE = ("Segoe UI", 17, "bold")
FONT_SCORE = ("Segoe UI", 12)
FONT_STATUS= ("Segoe UI", 13)
FONT_BTN   = ("Segoe UI", 11, "bold")


class TicTacToeGUI:
    def __init__(self, root: tk.Tk, agent: QLearningAgent):
        self.root   = root
        self.agent  = agent
        self.env    = TicTacToe()
        self.wins   = 0
        self.losses = 0
        self.draws  = 0
        self._hover: int | None = None
        self._locked = False   # prevents clicks while agent is thinking

        root.title("Tic-Tac-Toe  ·  Q-Learning Agent")
        root.configure(bg=BG_COLOR)
        root.resizable(False, False)

        self._build()
        self._redraw()

    # ── Layout ────────────────────────────────────────────────────────────
    def _build(self):
        board_px = 3 * CELL_SIZE + 4 * GAP + 2 * BOARD_PAD

        tk.Label(self.root, text="Tic-Tac-Toe  ×  Q-Agent",
                 bg=BG_COLOR, fg=TEXT_COLOR, font=FONT_TITLE
                 ).pack(pady=(20, 2))

        self._score_var = tk.StringVar()
        self._score_var.set("You: 0     Agent: 0     Draws: 0")
        tk.Label(self.root, textvariable=self._score_var,
                 bg=BG_COLOR, fg=DIM_COLOR, font=FONT_SCORE
                 ).pack(pady=(0, 10))

        # Legend row
        legend = tk.Frame(self.root, bg=BG_COLOR)
        legend.pack(pady=(0, 6))
        tk.Label(legend, text="X  You", bg=BG_COLOR, fg=X_COLOR,
                 font=("Segoe UI", 11, "bold")).pack(side="left", padx=12)
        tk.Label(legend, text="O  Agent", bg=BG_COLOR, fg=O_COLOR,
                 font=("Segoe UI", 11, "bold")).pack(side="left", padx=12)

        self._canvas = tk.Canvas(self.root, width=board_px, height=board_px,
                                 bg=BG_COLOR, highlightthickness=0)
        self._canvas.pack(padx=24)
        self._canvas.bind("<Motion>",   self._on_hover)
        self._canvas.bind("<Leave>",    self._on_leave)
        self._canvas.bind("<Button-1>", self._on_click)

        self._status_var = tk.StringVar(value="Your turn  (X)")
        tk.Label(self.root, textvariable=self._status_var,
                 bg=BG_COLOR, fg=TEXT_COLOR, font=FONT_STATUS
                 ).pack(pady=(10, 4))

        btn_frame = tk.Frame(self.root, bg=BG_COLOR)
        btn_frame.pack(pady=(4, 20))

        tk.Button(btn_frame, text="New Game",
                  command=self._new_game,
                  bg=BTN_COLOR, fg=BTN_FG, font=FONT_BTN,
                  relief="flat", padx=22, pady=7, cursor="hand2",
                  activebackground="#74c7ec", activeforeground=BTN_FG
                  ).pack(side="left", padx=6)

    # ── Drawing ───────────────────────────────────────────────────────────
    def _cell_bbox(self, idx: int):
        row, col = divmod(idx, 3)
        x1 = BOARD_PAD + GAP + col * (CELL_SIZE + GAP)
        y1 = BOARD_PAD + GAP + row * (CELL_SIZE + GAP)
        return x1, y1, x1 + CELL_SIZE, y1 + CELL_SIZE

    def _cell_at(self, x: int, y: int):
        for i in range(9):
            x1, y1, x2, y2 = self._cell_bbox(i)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    def _redraw(self):
        self._canvas.delete("all")
        win_cells = set(self.env.winning_line) if self.env.winning_line else set()
        available = set(self.env.available_moves()) if not self.env.done else set()

        for i in range(9):
            x1, y1, x2, y2 = self._cell_bbox(i)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if i in win_cells:
                fill = CELL_WIN
            elif i == self._hover and i in available and not self._locked:
                fill = CELL_HOVER
            else:
                fill = CELL_NORMAL

            # rounded rectangle via polygon
            r = 12
            self._canvas.create_polygon(
                x1+r, y1,  x2-r, y1,
                x2,   y1+r, x2,  y2-r,
                x2-r, y2,  x1+r, y2,
                x1,   y2-r, x1,  y1+r,
                fill=fill, smooth=True, outline="")

            mark = self.env.board[i]
            if mark == X:
                color = BG_COLOR if i in win_cells else X_COLOR
                self._canvas.create_text(cx, cy, text="X",
                                         font=FONT_MARK, fill=color)
            elif mark == O:
                color = BG_COLOR if i in win_cells else O_COLOR
                self._canvas.create_text(cx, cy, text="O",
                                         font=FONT_MARK, fill=color)

    # ── Mouse events ──────────────────────────────────────────────────────
    def _on_hover(self, event):
        cell = self._cell_at(event.x, event.y)
        if cell != self._hover:
            self._hover = cell
            self._redraw()

    def _on_leave(self, _event):
        self._hover = None
        self._redraw()

    def _on_click(self, event):
        if self._locked or self.env.done:
            return
        cell = self._cell_at(event.x, event.y)
        if cell is None or self.env.board[cell] != 0:
            return
        self._locked = True
        self.env.step(cell, player=X)
        self._redraw()
        if self.env.done:
            self._finish()
            return
        self._status_var.set("Agent is thinking…")
        self.root.after(350, self._agent_turn)

    # ── Game flow ─────────────────────────────────────────────────────────
    def _agent_turn(self):
        moves = self.env.available_moves()
        if moves:
            action = self.agent.choose_action(self.env.state_key(),
                                              moves, greedy=True)
            self.env.step(action, player=O)
        self._redraw()
        if self.env.done:
            self._finish()
        else:
            self._status_var.set("Your turn  (X)")
            self._locked = False

    def _finish(self):
        self._locked = True
        self._redraw()
        if self.env.winner == X:
            self.wins += 1
            self._status_var.set("You win!  Press New Game to play again.")
        elif self.env.winner == O:
            self.losses += 1
            self._status_var.set("Agent wins!  Press New Game to try again.")
        else:
            self.draws += 1
            self._status_var.set("Draw!  Press New Game to play again.")
        self._score_var.set(
            f"You: {self.wins}     Agent: {self.losses}     Draws: {self.draws}")

    def _new_game(self):
        self.env     = TicTacToe()
        self._hover  = None
        self._locked = False
        self._status_var.set("Your turn  (X)")
        self._redraw()


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GUI Tic-Tac-Toe vs Q-agent")
    parser.add_argument("--model", default=os.path.join(
        os.path.dirname(__file__), "q_table.pkl"))
    args = parser.parse_args()

    agent = QLearningAgent()
    if not os.path.exists(args.model):
        print(f"No model at '{args.model}' — training first (20 000 episodes)…")
        from train import train
        agent = train(episodes=20_000, save_path=args.model, print_every=5000)
    else:
        agent.load(args.model)

    root = tk.Tk()
    TicTacToeGUI(root, agent)
    root.mainloop()


if __name__ == "__main__":
    main()
