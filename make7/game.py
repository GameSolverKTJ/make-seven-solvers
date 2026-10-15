"""
Make 7 game engine — 7x7 board, gravity drop, unlimited tile supply.

Board convention
----------------
row 0 is the BOTTOM row, row 6 is the TOP row.
col 0 is 'A', col 6 is 'G'.

Cell encoding in `board` (int8):
    0   empty
   +v   a tile of value v (1/2/3) owned by GREEN  (player +1)
   -v   a tile of value v (1/2/3) owned by YELLOW (player -1)

Rules implemented
-----------------
* Players alternate; GREEN moves first.
* A move = (tile value, column). The tile falls to the lowest empty cell.
* Tile supply is UNLIMITED for both players (1s, 2s and 3s alike).
  This is a deliberate variant: the physical game caps each player at
  11x'1', 11x'2', 4x'3'.  Removing the cap makes the state Markovian in
  the board alone (no hand counts), which simplifies the network input.
* A '3' may only come to rest on a RED square.  The red squares were
  recovered empirically from 3,733 solved positions in the DF-PN exact
  database; the layout is left-right symmetric.
* A player wins immediately when, after their move, some contiguous run
  of THEIR OWN tiles (horizontal, vertical or either diagonal) sums to
  exactly 7.
* If the board fills with no such run, the game is a draw.
"""

from __future__ import annotations

import numpy as np

BOARD_H = 7
BOARD_W = 7
NUM_TILE_TYPES = 3
ACTION_SIZE = NUM_TILE_TYPES * BOARD_W  # 21 = {1,2,3} x {A..G}
TARGET_SUM = 7
MAX_PLIES = BOARD_H * BOARD_W  # 49

# (row, col) cells on which a '3' is allowed to land.
RED_SQUARES = frozenset(
    {(2, 0), (2, 6), (3, 3), (4, 1), (4, 5), (5, 2), (5, 4)}
)

RED_MASK = np.zeros((BOARD_H, BOARD_W), dtype=np.float32)
for _r, _c in RED_SQUARES:
    RED_MASK[_r, _c] = 1.0

# horizontal, vertical, diagonal-up-right, diagonal-down-right
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

COLUMN_NAMES = "ABCDEFG"

# Per-player tile supply. A limit tuple is (n1, n2, n3) where each entry is
# either an integer cap or None for "unlimited"; passing None instead of a
# tuple makes every tile unlimited, which is the default used for training.
#
# Legacy limited variants, kept only for explicit historical comparisons.
# The public release uses limits=None: all tile values are unlimited.
# A historical database's observed move counts do not establish its ruleset.
THREE_ONLY = (None, None, 4)
STANDARD_LIMITS = (11, 11, 4)


def action_to_move(action: int) -> tuple[int, int]:
    """action -> (tile_value, column_index)"""
    return action // BOARD_W + 1, action % BOARD_W


def move_to_action(tile: int, col: int) -> int:
    """(tile_value, column_index) -> action"""
    return (tile - 1) * BOARD_W + col


def action_to_str(action: int) -> str:
    tile, col = action_to_move(action)
    return f"{tile}{COLUMN_NAMES[col]}"


def str_to_action(token: str) -> int:
    tile = int(token[0])
    col = COLUMN_NAMES.index(token[1].upper())
    return move_to_action(tile, col)


def mirror_action(action: int) -> int:
    """Left-right mirror. The red-square layout is mirror invariant, so this
    is an exact symmetry of the game and is safe for data augmentation."""
    tile, col = action_to_move(action)
    return move_to_action(tile, BOARD_W - 1 - col)


class Make7State:
    """An immutable-by-convention game state. `step` returns a new object."""

    __slots__ = ("board", "heights", "player", "winner", "ply", "last_move",
                 "limits", "used")

    def __init__(self, board=None, heights=None, player=1, winner=None, ply=0,
                 last_move=None, limits=None, used=None):
        if board is None:
            board = np.zeros((BOARD_H, BOARD_W), dtype=np.int8)
        if heights is None:
            heights = np.zeros(BOARD_W, dtype=np.int8)
        self.board = board
        self.heights = heights
        self.player = player          # +1 GREEN to move, -1 YELLOW to move
        self.winner = winner          # None = ongoing, +1/-1 = winner, 0 = draw
        self.ply = ply
        self.last_move = last_move    # (row, col) of the most recent placement
        # `limits` is None for unlimited supply, or a (n1, n2, n3) tuple.
        # `used` counts placements as (g1, g2, g3, y1, y2, y3) and is only
        # maintained when limits are in force.
        self.limits = limits
        self.used = used if used is not None else (0, 0, 0, 0, 0, 0)

    def remaining(self, player: int) -> tuple[int, int, int]:
        """Tiles of each value still available to `player`."""
        big = 1 << 30
        if self.limits is None:
            return (big, big, big)
        off = 0 if player == 1 else 3
        return tuple(big if self.limits[i] is None
                     else self.limits[i] - self.used[off + i]
                     for i in range(3))

    # ------------------------------------------------------------------ #
    # basic queries
    # ------------------------------------------------------------------ #

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None

    def legal_actions(self) -> np.ndarray:
        """Boolean mask of shape (21,)."""
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        if self.winner is not None:
            return mask
        left = self.remaining(self.player)
        for col in range(BOARD_W):
            row = int(self.heights[col])
            if row >= BOARD_H:
                continue
            if left[0] > 0:
                mask[move_to_action(1, col)] = True
            if left[1] > 0:
                mask[move_to_action(2, col)] = True
            if left[2] > 0 and (row, col) in RED_SQUARES:
                mask[move_to_action(3, col)] = True
        return mask

    def legal_action_list(self) -> list[int]:
        return np.flatnonzero(self.legal_actions()).tolist()

    # ------------------------------------------------------------------ #
    # transition
    # ------------------------------------------------------------------ #

    def step(self, action: int) -> "Make7State":
        if self.winner is not None:
            raise ValueError("game is already over")
        tile, col = action_to_move(action)
        row = int(self.heights[col])
        if row >= BOARD_H:
            raise ValueError(f"column {COLUMN_NAMES[col]} is full")
        if tile == 3 and (row, col) not in RED_SQUARES:
            raise ValueError(f"a '3' cannot rest on {COLUMN_NAMES[col]}{row}")
        if self.remaining(self.player)[tile - 1] <= 0:
            raise ValueError(f"no '{tile}' tiles left for this player")

        board = self.board.copy()
        heights = self.heights.copy()
        board[row, col] = np.int8(tile * self.player)
        heights[col] = row + 1
        ply = self.ply + 1

        used = self.used
        if self.limits is not None:
            off = 0 if self.player == 1 else 3
            used = list(used)
            used[off + tile - 1] += 1
            used = tuple(used)

        if _is_winning_placement(board, row, col, self.player):
            winner = self.player
        elif ply >= MAX_PLIES:
            winner = 0
        else:
            winner = None

        nxt = Make7State(board, heights, -self.player, winner, ply, (row, col),
                         self.limits, used)
        # Under supply limits a player can, very rarely, be left holding only
        # 3s with no red square open. Treat that as a drawn position rather
        # than inventing a pass rule; `compare_rulesets.py` reports how often
        # it happens so the choice can be checked against the data.
        if self.limits is not None and winner is None and not nxt.legal_actions().any():
            nxt.winner = 0
        return nxt

    # ------------------------------------------------------------------ #
    # network input
    # ------------------------------------------------------------------ #

    def encode(self) -> np.ndarray:
        """(10, 7, 7) float32 tensor, always from the mover's point of view.

        planes 0..2  mover's 1s / 2s / 3s
        planes 3..5  opponent's 1s / 2s / 3s
        plane  6     red-square mask (constant)
        plane  7     all ones (bias / board-edge reference)
        plane  8     mover's remaining 3s, scaled to [0, 1]
        plane  9     opponent's remaining 3s, scaled to [0, 1]

        The last two planes matter only when a supply limit is in force; with
        unlimited tiles they are constant 1.0 and carry nothing. They are
        present unconditionally so that a single architecture covers both
        rulesets, which is what makes a limited-supply model comparable
        against the exact database without changing the network.

        Writing the board from the mover's point of view keeps the two
        players symmetric, so the network only has to learn one side. The
        cost is that the side to move is no longer an explicit input — it can
        still be inferred from the parity of the tile count, which is worth
        remembering when a model starts scoring positions by whose turn it is
        rather than by what is on the board.
        """
        planes = np.zeros((10, BOARD_H, BOARD_W), dtype=np.float32)
        # Signed board values already separate the two players, so one
        # multiply puts the mover's tiles in 1..3 and the opponent's in -1..-3.
        rel = self.board if self.player == 1 else -self.board
        for i, v in enumerate((1, 2, 3)):
            planes[i] = (rel == v)
            planes[3 + i] = (rel == -v)
        planes[6] = RED_MASK
        planes[7] = 1.0

        cap = None if self.limits is None else self.limits[2]
        if cap:
            mine, theirs = self.remaining(self.player), self.remaining(-self.player)
            planes[8] = min(1.0, mine[2] / cap)
            planes[9] = min(1.0, theirs[2] / cap)
        else:
            planes[8] = 1.0
            planes[9] = 1.0
        return planes

    def mirror(self) -> "Make7State":
        return Make7State(
            np.ascontiguousarray(self.board[:, ::-1]),
            np.ascontiguousarray(self.heights[::-1]),
            self.player,
            self.winner,
            self.ply,
            None if self.last_move is None
            else (self.last_move[0], BOARD_W - 1 - self.last_move[1]),
            self.limits,
            self.used,
        )

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #

    def key(self) -> bytes:
        return self.board.tobytes() + bytes([self.player & 0xFF])

    def result_for(self, player: int) -> float:
        """+1 win / -1 loss / 0 draw, from `player`'s point of view."""
        if self.winner is None:
            raise ValueError("game is not finished")
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def __str__(self) -> str:
        out = []
        for row in range(BOARD_H - 1, -1, -1):
            cells = []
            for col in range(BOARD_W):
                v = int(self.board[row, col])
                if v == 0:
                    cells.append(" # " if (row, col) in RED_SQUARES else " . ")
                elif v > 0:
                    cells.append(f" {v} ")        # GREEN upper
                else:
                    cells.append(f"({-v})")       # YELLOW in parens
            out.append(f"{row} " + "".join(cells))
        out.append("  " + "".join(f" {c} " for c in COLUMN_NAMES))
        who = "GREEN" if self.player == 1 else "YELLOW"
        if self.winner is None:
            out.append(f"ply {self.ply} — {who} to move")
        elif self.winner == 0:
            out.append(f"ply {self.ply} — draw")
        else:
            out.append(f"ply {self.ply} — "
                       f"{'GREEN' if self.winner == 1 else 'YELLOW'} wins")
        return "\n".join(out)


def _is_winning_placement(board: np.ndarray, row: int, col: int,
                          player: int) -> bool:
    """Does the tile just placed at (row, col) complete a run summing to 7?

    Only runs that contain the new tile can be new, so we walk the maximal
    same-owner run through (row, col) in each direction and test every
    sub-window that covers it.  Tile values are all >= 1, so a running sum is
    strictly increasing and we can stop as soon as it passes 7.
    """
    positive = player > 0
    centre = abs(int(board[row, col]))

    for dr, dc in _DIRECTIONS:
        before = []
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_H and 0 <= c < BOARD_W:
            v = int(board[r, c])
            if v == 0 or (v > 0) != positive:
                break
            before.append(abs(v))
            r, c = r - dr, c - dc

        after = []
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_H and 0 <= c < BOARD_W:
            v = int(board[r, c])
            if v == 0 or (v > 0) != positive:
                break
            after.append(abs(v))
            r, c = r + dr, c + dc

        before.reverse()
        pivot = len(before)
        values = before + [centre] + after

        n = len(values)
        for i in range(pivot + 1):
            total = 0
            for j in range(i, n):
                total += values[j]
                if total > TARGET_SUM:
                    break
                if total == TARGET_SUM and j >= pivot:
                    return True
    return False


def new_game(limits=None) -> Make7State:
    """Fresh position. `limits` is None for unlimited supply (the default used
    for training) or a (n1, n2, n3) tuple such as `STANDARD_LIMITS`."""
    return Make7State(limits=limits)


def play_sequence(tokens: str | list[str], limits=None) -> Make7State:
    """Replay a game written in the DF-PN campaign notation, e.g. '2D1D2B'."""
    if isinstance(tokens, str):
        tokens = [tokens[i:i + 2] for i in range(0, len(tokens), 2)]
    state = new_game(limits)
    for tok in tokens:
        state = state.step(str_to_action(tok))
    return state
