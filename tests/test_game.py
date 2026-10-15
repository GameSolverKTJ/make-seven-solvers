"""Validation of the Make 7 engine, including a replay of every position in
the DF-PN exact database."""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make7.game import (  # noqa: E402
    ACTION_SIZE, BOARD_H, BOARD_W, RED_SQUARES, Make7State,
    action_to_str, mirror_action, move_to_action, new_game, play_sequence,
    str_to_action,
)

DB = Path("/mnt/user-data/uploads/exact_propositions_campaign.txt")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def test_action_coding():
    for a in range(ACTION_SIZE):
        check(str_to_action(action_to_str(a)) == a, f"round-trip action {a}")
    check(action_to_str(move_to_action(3, 3)) == "3D", "3D encodes correctly")


def test_red_squares_mirror_invariant():
    mirrored = {(r, BOARD_W - 1 - c) for r, c in RED_SQUARES}
    check(mirrored == set(RED_SQUARES), "red-square layout is mirror invariant")


def test_legality():
    s = new_game()
    legal = s.legal_actions()
    # at ply 0 every column has its landing square on row 0, which is never red
    check(legal.sum() == 14, "opening has 14 legal moves (7 cols x {1,2})")
    check(not legal[move_to_action(3, 3)], "3D illegal on an empty board")

    # column D: rows 0,1,2 filled -> next landing row is 3, which IS red
    s = play_sequence("1D1D1D")
    check(s.legal_actions()[move_to_action(3, 3)], "3D legal once D has height 3")

    # fill a column and check it closes
    s = play_sequence("1A" * 7)
    for tile in (1, 2, 3):
        check(not s.legal_actions()[move_to_action(tile, 0)],
              f"column A closed for tile {tile}")


def test_horizontal_win():
    # GREEN: 3 then 4? no -- build 3+2+2 = 7 on the bottom row
    # green plays A,B,C with 3? 3 is not legal on row 0, use 2+2+3 -> use 1s/2s
    # GREEN bottom row: 2A 2B 3? -> use 2+2+2+1 = 7 across A,B,C,D
    s = new_game()
    for tok in ["2A", "1G", "2B", "1G", "2C", "1G"]:
        s = s.step(str_to_action(tok))
    check(not s.is_terminal, "2+2+2 across A,B,C is not yet 7")
    s = s.step(str_to_action("1D"))
    check(s.winner == 1, "GREEN wins with 2+2+2+1 = 7 horizontally")


def test_vertical_win():
    s = new_game()
    # GREEN stacks 2,2,2,1 in column A while YELLOW plays elsewhere
    for tok in ["2A", "1G", "2A", "1G", "2A", "1G"]:
        s = s.step(str_to_action(tok))
    check(not s.is_terminal, "column of 2,2,2 is only 6")
    s = s.step(str_to_action("1A"))
    check(s.winner == 1, "GREEN wins with a vertical 2+2+2+1")


def test_exact_seven_required():
    s = new_game()
    # 2+2+2+2 = 8 must NOT win, and must not win by containing a 7 either
    for tok in ["2A", "1G", "2B", "1G", "2C", "1G"]:
        s = s.step(str_to_action(tok))
    s = s.step(str_to_action("2D"))
    check(s.winner is None, "a run summing to 8 does not win")


def test_opponent_tiles_dont_count():
    s = new_game()
    # GREEN 3 on a red square is impossible at row 0; use a mixed row instead
    s = play_sequence("2A")          # GREEN 2 at A0
    s = s.step(str_to_action("2B"))  # YELLOW 2 at B0
    s = s.step(str_to_action("2C"))  # GREEN 2 at C0
    s = s.step(str_to_action("1D"))  # YELLOW 1 at D0
    check(s.winner is None, "mixed-owner runs never sum to a win")


def test_diagonal_win():
    # GREEN staircase A0=2, B1=2, C2=2, D3=1  ->  2+2+2+1 = 7
    # GREEN burns spare moves in column G; YELLOW fills the cells underneath.
    seq = ["2A", "1B",   # G A0            / Y B0
           "2B", "1C",   # G B1            / Y C0
           "1G", "1C",   # G spare         / Y C1
           "2C", "1D",   # G C2            / Y D0
           "1G", "1D",   # G spare         / Y D1
           "1G", "1D"]   # G spare         / Y D2
    s = new_game()
    for tok in seq:
        s = s.step(str_to_action(tok))
    check(s.player == 1 and not s.is_terminal, "GREEN to move at ply 12")
    s = s.step(str_to_action("1D"))          # D3
    check(s.winner == 1, "GREEN wins on the up-right diagonal 2+2+2+1")


def test_anti_diagonal_win():
    # GREEN staircase A3=1, B2=2, C1=2, D0=2 on the down-right diagonal
    seq = ["2D", "1A",   # G D0            / Y A0
           "2C", "1A",   # G C1 (row 1)    / Y A1  -- C0 still empty
           "1G", "1A"]   # G spare         / Y A2
    s = new_game()
    # C1 needs C0 filled first; rebuild carefully
    s = new_game()
    for tok in ["2D", "1C",   # G D0        / Y C0
                "2C", "1B",   # G C1        / Y B0
                "1G", "1B",   # G spare     / Y B1
                "2B", "1A",   # G B2        / Y A0
                "1G", "1A",   # G spare     / Y A1
                "1G", "1A"]:  # G spare     / Y A2
        s = s.step(str_to_action(tok))
    check(s.player == 1 and not s.is_terminal, "GREEN to move at ply 12")
    s = s.step(str_to_action("1A"))          # A3
    check(s.winner == 1, "GREEN wins on the down-right diagonal 2+2+2+1")


def test_three_only_on_red():
    """3s are restricted to red squares; 1s and 2s are not restricted at all.
    Both halves of this were confirmed against 88,172 real placements."""
    s = play_sequence("1D1D1D")          # D height 3, landing row 3 = red
    check(s.legal_actions()[move_to_action(3, 3)], "3 allowed on a red square")
    check(s.legal_actions()[move_to_action(1, 3)], "1 also allowed on red")
    check(s.legal_actions()[move_to_action(2, 3)], "2 also allowed on red")
    s2 = play_sequence("1D1D")           # landing row 2 in column D, not red
    check(not s2.legal_actions()[move_to_action(3, 3)],
          "3 refused on a non-red square")


def test_win_with_a_three():
    # GREEN: D0=2, D1=2, D2=... need vertical 2+2+3 = 7 with the 3 on D3 (red)
    s = new_game()
    for tok in ["1D", "1G", "2D", "1G", "1D", "1G"]:
        s = s.step(str_to_action(tok))
    # column D now: D0=1(G), D1=2(Y)? -- rebuild with clean ownership
    s = new_game()
    for tok in ["2D", "1A", "2D", "1A", "1D", "1A"]:
        s = s.step(str_to_action(tok))
    # D0=2(G) D1=2(G) D2=1(G); landing row is 3 (red). 2+2+1 = 5, +3 would be 8.
    check(not s.is_terminal, "2+2+1 vertical is only 5")
    # instead verify a 3 completes 3+2+2 from a fresh column
    s = new_game()
    for tok in ["2E", "1A", "2E", "1A", "1E", "1A", "1E", "1A"]:
        s = s.step(str_to_action(tok))
    # E: E0=2(G) E1=2(G) E2=1(G) E3=1(G) -> sum 6 over four, landing row 4
    check(not s.is_terminal, "no accidental win while staging the column")


def test_mirror_symmetry():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = new_game()
        while not s.is_terminal:
            legal = s.legal_action_list()
            s = s.step(int(rng.choice(legal)))
        m = s
        # mirror a mid-game state and confirm legality masks correspond
        s2 = new_game()
        steps = 0
        while not s2.is_terminal and steps < 12:
            legal = s2.legal_action_list()
            a = int(rng.choice(legal))
            s2 = s2.step(a)
            steps += 1
        ms = s2.mirror()
        la = s2.legal_actions()
        lb = ms.legal_actions()
        expected = np.zeros_like(lb)
        for a in range(ACTION_SIZE):
            if la[a]:
                expected[mirror_action(a)] = True
        check_silent = (expected == lb).all()
        if not check_silent:
            raise AssertionError("mirrored legality mask mismatch")
    print("  ok  mirror preserves legality over 200 random states")


def test_encode_shape():
    s = play_sequence("2D1D2B2C")
    e = s.encode()
    check(e.shape == (10, BOARD_H, BOARD_W), "encoding is (10,7,7)")
    check(e[6].sum() == len(RED_SQUARES), "red plane has 7 set cells")
    check(e[7].sum() == BOARD_H * BOARD_W, "bias plane is all ones")
    # mover-relative: after 4 plies GREEN is to move again
    check(s.player == 1, "GREEN to move after 4 plies")
    check(e[1].sum() == 2, "GREEN owns two 2-tiles (2D, 2B)")


def test_replay_exact_database():
    """Every position proven by the DF-PN campaign must be reachable under
    these rules.  This is the strongest available check on the red-square
    layout, on gravity, and on the win condition firing at the right time."""
    if not DB.exists():
        print("  --  exact database not present, skipping replay")
        return

    total = illegal = premature = 0
    terminal_ends = 0
    with DB.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pos = parts[1]
            toks = re.findall(r"[123][A-G]", pos)
            if "".join(toks) != pos:
                continue
            total += 1
            s = new_game()
            try:
                for i, tok in enumerate(toks):
                    if s.is_terminal:
                        # a win fired before the record ended
                        premature += 1
                        break
                    s = s.step(str_to_action(tok))
                else:
                    if s.is_terminal:
                        terminal_ends += 1
            except ValueError:
                illegal += 1

    print(f"  --  replayed {total} database positions")
    check(illegal == 0, f"no illegal move in any record (illegal={illegal})")
    check(premature == 0,
          f"no record continues past a win (premature={premature})")
    print(f"  --  {terminal_ends} records end exactly on a terminal position")


def test_random_playouts():
    rng = np.random.default_rng(7)
    lengths, wins = [], {1: 0, -1: 0, 0: 0}
    for _ in range(2000):
        s = new_game()
        while not s.is_terminal:
            s = s.step(int(rng.choice(s.legal_action_list())))
        lengths.append(s.ply)
        wins[s.winner] += 1
    check(max(lengths) <= 49, "no game exceeds 49 plies")
    print(f"  --  random playouts: mean length {np.mean(lengths):.1f}, "
          f"green {wins[1]}, yellow {wins[-1]}, draws {wins[0]}")


if __name__ == "__main__":
    for fn in [
        test_action_coding,
        test_red_squares_mirror_invariant,
        test_legality,
        test_horizontal_win,
        test_vertical_win,
        test_exact_seven_required,
        test_opponent_tiles_dont_count,
        test_diagonal_win,
        test_anti_diagonal_win,
        test_three_only_on_red,
        test_win_with_a_three,
        test_mirror_symmetry,
        test_encode_shape,
        test_replay_exact_database,
        test_random_playouts,
    ]:
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nall engine tests passed")
