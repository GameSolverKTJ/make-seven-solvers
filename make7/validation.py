"""Validation of a network against positions the DF-PN campaign proved.

Three things about this database force the design below.

Labels are lopsided and partly one-sided
----------------------------------------
Of 3,733 records, 1,251 are proven GREEN wins and only 387 proven YELLOW
wins, so plain accuracy rewards a model that answers "GREEN" every time with
76% before it has learned anything.  Every headline number here is therefore
balanced across the two classes.  A further 2,095 records only constrain the
result from one side — "GREEN DISPROVED" means GREEN cannot force a win, so
GREEN's value is at most 0 — and those are reported separately rather than
folded into an accuracy figure they do not fit.

Records are not independent
---------------------------
Proof-number search drills into whatever line is most proof-critical, so the
records cluster hard: at ply 25 all 2,782 of them descend from 46 distinct
19-ply positions, and one parent alone accounts for 787.  Averaging over
records would let a single region of the tree decide the score.  Every metric
here is computed per group and then averaged over groups, so each region
counts once.

Train and validation must be split at a shallow prefix
------------------------------------------------------
Splitting deep would put near-identical positions on both sides.  The split
is made on a short prefix — 12 plies by default — so held-out positions come
from genuinely different openings rather than from sibling continuations of
a line the model has already been fitted to.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .game import Make7State, new_game, str_to_action


@dataclass
class ExactRecord:
    tokens: list[str]
    position: str           # compact move string, e.g. 2D1C...
    proposition: str        # exact DB verdict, e.g. GREEN_PROVED
    group: str              # shallow prefix that defines the split unit
    label: int | None       # +1 GREEN win, -1 YELLOW win, 0 draw, None = one-sided
    bound: str | None       # "le0" (GREEN <= 0) or "ge0" (GREEN >= 0)
    state: Make7State


def _classify(parts: list[str]):
    """Map a database verdict to (label, bound, proposition name)."""
    verdict = parts[2]
    tail = parts[3] if len(parts) > 3 else ""

    if verdict.startswith("GREEN_WIN"):
        return 1, None, "GREEN_WIN"
    if verdict.startswith("YELLOW_WIN"):
        return -1, None, "YELLOW_WIN"
    if verdict.startswith("DRAW"):
        return 0, None, "DRAW"
    if verdict == "GREEN" and tail == "PROVED":
        return 1, None, "GREEN_PROVED"
    if verdict == "YELLOW" and tail == "PROVED":
        return -1, None, "YELLOW_PROVED"
    if verdict == "GREEN" and tail == "DISPROVED":
        return None, "le0", "GREEN_DISPROVED"  # GREEN cannot force a win
    if verdict == "YELLOW" and tail == "DISPROVED":
        return None, "ge0", "YELLOW_DISPROVED"  # YELLOW cannot force a win
    return None, None, "UNKNOWN"


def load_exact_records(path, limits=None, group_plies: int = 12
                       ) -> list[ExactRecord]:
    """Parse the database and replay every position under `limits`.

    Pass the same limits the model is trained with. Records that cannot be
    reached under those rules, or that are already finished, are dropped.
    """
    out: list[ExactRecord] = []
    with Path(path).open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            toks = re.findall(r"[123][A-G]", parts[1])
            if "".join(toks) != parts[1] or len(toks) <= group_plies:
                continue
            label, bound, proposition = _classify(parts)
            if label is None and bound is None:
                continue

            state = new_game(limits)
            ok = True
            for tok in toks:
                if state.is_terminal:
                    ok = False
                    break
                try:
                    state = state.step(str_to_action(tok))
                except ValueError:
                    ok = False
                    break
            if not ok or state.is_terminal:
                continue

            out.append(ExactRecord(
                toks, parts[1], proposition, "".join(toks[:group_plies]),
                label, bound, state))
    return out


def split_by_group(records: list[ExactRecord], validation_fraction: float = 0.4,
                   seed: int = 0) -> tuple[list[ExactRecord], list[ExactRecord]]:
    """Hold out whole groups, not individual records.

    Groups are shuffled and assigned to validation until the target share of
    *decisive* records is reached, since those carry the headline metric.
    """
    by_group: dict[str, list[ExactRecord]] = defaultdict(list)
    for r in records:
        by_group[r.group].append(r)

    keys = sorted(by_group)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    decisive_total = sum(1 for r in records if r.label is not None)
    target = decisive_total * validation_fraction

    val_keys, taken = [], 0
    for key in keys:
        if taken >= target:
            break
        val_keys.append(key)
        taken += sum(1 for r in by_group[key] if r.label is not None)

    val_set = set(val_keys)
    train = [r for r in records if r.group not in val_set]
    val = [r for r in records if r.group in val_set]
    return train, val


def green_values(records: list[ExactRecord], evaluator, batch_size: int = 512
                 ) -> np.ndarray:
    """Network value for each record, expressed from GREEN's point of view."""
    out = np.zeros(len(records), dtype=np.float32)
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        batch = np.stack([r.state.encode() for r in chunk])
        _, values = evaluator(batch)
        signs = np.array([r.state.player for r in chunk], dtype=np.float32)
        out[i:i + len(chunk)] = np.asarray(values, dtype=np.float32) * signs
    return out


def _class_weights(records, labels):
    """Weight records so that every group contributes equally within a class.

    Without this the largest proof group — 787 records descending from one
    ply-19 position — would dominate whichever class it belongs to.
    """
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for r, y in zip(records, labels):
        counts[(r.group, y)] += 1
    return np.array([1.0 / counts[(r.group, y)]
                     for r, y in zip(records, labels)], dtype=np.float64)


def _weighted_auc(scores, labels, weights) -> float:
    """Group-balanced AUC: the chance a GREEN win outranks a YELLOW win.

    This is the threshold-free counterpart to the accuracy figures. Those
    take the sign of the value, so a position the model scores at -0.01
    counts as a full error while one at -0.99 counts the same; AUC only asks
    whether wins are ranked above losses, which makes it markedly steadier
    from generation to generation. 0.5 is chance and it is unaffected by how
    lopsided the label counts are.
    """
    pos = scores[labels == 1]
    neg = scores[labels == -1]
    wp = weights[labels == 1]
    wn = weights[labels == -1]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).astype(np.float64)
    ties = (pos[:, None] == neg[None, :]).astype(np.float64)
    pair_w = wp[:, None] * wn[None, :]
    total = pair_w.sum()
    if total <= 0:
        return float("nan")
    return float(((greater + 0.5 * ties) * pair_w).sum() / total)


def _weighted_confusion(scores, labels, weights, threshold: float = 0.0):
    pred = np.where(scores > threshold, 1, -1)
    tp = weights[(labels == 1) & (pred == 1)].sum()
    fn = weights[(labels == 1) & (pred == -1)].sum()
    tn = weights[(labels == -1) & (pred == -1)].sum()
    fp = weights[(labels == -1) & (pred == 1)].sum()
    return tp, fp, tn, fn


def _macro_f1(tp, fp, tn, fn) -> float:
    """Mean of the two per-class F1 scores.

    Plain F1 has to pick one class as positive and ignores true negatives, so
    it would give different answers depending on whether GREEN or YELLOW is
    called positive. Averaging both classes restores the symmetry the problem
    actually has.
    """
    def f1(t, f_pos, f_neg):
        denom = 2 * t + f_pos + f_neg
        return 2 * t / denom if denom > 0 else float("nan")
    a = f1(tp, fp, fn)
    b = f1(tn, fn, fp)
    vals = [v for v in (a, b) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _mcc(tp, fp, tn, fn) -> float:
    """Matthews correlation: +1 perfect, 0 chance, -1 inverted.

    Reported alongside the others because it is the one figure here that goes
    negative when a model is systematically backwards, which is exactly the
    failure this database caught once already.
    """
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num / den) if den > 0 else float("nan")



def stratified_split_by_group(records: list[ExactRecord],
                              calibration_fraction: float = 0.5,
                              seed: int = 0
                              ) -> tuple[list[ExactRecord], list[ExactRecord]]:
    """Split whole proof groups into calibration/train and untouched test.

    The decisive labels are highly clustered by shallow prefix.  A plain
    shuffled group split can therefore put almost every YELLOW-win group on
    one side.  We stratify groups by whether they contain GREEN wins,
    YELLOW wins, both, or neither, while never splitting a group.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    by_group: dict[str, list[ExactRecord]] = defaultdict(list)
    for r in records:
        by_group[r.group].append(r)

    strata: dict[tuple[bool, bool], list[str]] = defaultdict(list)
    for key, rs in by_group.items():
        has_green = any(r.label == 1 for r in rs)
        has_yellow = any(r.label == -1 for r in rs)
        strata[(has_green, has_yellow)].append(key)

    rng = np.random.default_rng(seed)
    calibration_keys: set[str] = set()
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        if len(keys) == 1:
            # A singleton stratum cannot be represented on both sides. Keep it
            # untouched in test unless it contains no decisive information.
            take = 1 if stratum == (False, False) and calibration_fraction >= 0.5 else 0
        else:
            take = int(round(len(keys) * calibration_fraction))
            take = max(1, min(len(keys) - 1, take))
        calibration_keys.update(keys[:take])

    calibration = [r for r in records if r.group in calibration_keys]
    test = [r for r in records if r.group not in calibration_keys]
    return calibration, test


def threshold_metrics(records: list[ExactRecord], scores: np.ndarray,
                      threshold: float = 0.0) -> dict:
    """Group-balanced decisive metrics at an arbitrary GREEN-value threshold."""
    mask = np.array([r.label in (1, -1) for r in records], dtype=bool)
    rs = [r for r, m in zip(records, mask) if m]
    if not rs:
        return {"balanced_accuracy": float("nan"), "macro_f1": float("nan"),
                "mcc": float("nan"), "green_win_accuracy": float("nan"),
                "yellow_win_accuracy": float("nan")}
    y = np.array([r.label for r in rs], dtype=np.int8)
    x = np.asarray(scores)[mask]
    w = _class_weights(rs, y)
    tp, fp, tn, fn = _weighted_confusion(x, y, w, threshold=threshold)
    green = tp / (tp + fn) if tp + fn > 0 else float("nan")
    yellow = tn / (tn + fp) if tn + fp > 0 else float("nan")
    vals = [v for v in (green, yellow) if np.isfinite(v)]
    balanced = float(np.mean(vals)) if vals else float("nan")
    return {
        "balanced_accuracy": balanced,
        "macro_f1": _macro_f1(tp, fp, tn, fn),
        "mcc": _mcc(tp, fp, tn, fn),
        "green_win_accuracy": float(green),
        "yellow_win_accuracy": float(yellow),
    }


def optimal_threshold(records: list[ExactRecord], scores: np.ndarray) -> tuple[float, dict]:
    """Choose the GREEN/YELLOW threshold maximizing group-balanced accuracy.

    This must be called only on the calibration side of a group-held-out split.
    AUC is unaffected; the threshold is a diagnostic of value calibration.
    """
    mask = np.array([r.label in (1, -1) for r in records], dtype=bool)
    x = np.asarray(scores, dtype=np.float64)[mask]
    if x.size == 0:
        return float("nan"), threshold_metrics(records, scores, 0.0)
    unique = np.unique(x)
    if unique.size == 1:
        candidates = np.array([unique[0]], dtype=np.float64)
    else:
        mids = (unique[:-1] + unique[1:]) / 2.0
        eps = max(1e-6, float(unique[-1] - unique[0]) * 1e-6)
        candidates = np.concatenate(([unique[0] - eps], mids, [unique[-1] + eps]))
    best_t, best_m = 0.0, None
    best_score = -np.inf
    # Tie-break toward zero: it is the least intervention when several
    # thresholds give the same balanced accuracy.
    for t in candidates:
        m = threshold_metrics(records, scores, float(t))
        score = m["balanced_accuracy"]
        if (score > best_score + 1e-12 or
                (abs(score - best_score) <= 1e-12 and abs(t) < abs(best_t))):
            best_score, best_t, best_m = score, float(t), m
    return best_t, best_m


def calibration_report(calibration_records: list[ExactRecord], calibration_scores: np.ndarray,
                       test_records: list[ExactRecord], test_scores: np.ndarray) -> dict:
    """Fit threshold on calibration groups and report it on untouched groups."""
    threshold, cal = optimal_threshold(calibration_records, calibration_scores)
    at_zero = threshold_metrics(test_records, test_scores, 0.0)
    calibrated = threshold_metrics(test_records, test_scores, threshold)
    return {
        "optimal_threshold": threshold,
        "calibration_balanced_accuracy": cal["balanced_accuracy"],
        "test_balanced_accuracy_at_zero": at_zero["balanced_accuracy"],
        "test_mcc_at_zero": at_zero["mcc"],
        "calibrated_balanced_accuracy": calibrated["balanced_accuracy"],
        "calibrated_mcc": calibrated["mcc"],
        "calibrated_green_win_accuracy": calibrated["green_win_accuracy"],
        "calibrated_yellow_win_accuracy": calibrated["yellow_win_accuracy"],
    }


def sample_exact_decisive_batch(records: list[ExactRecord], batch_size: int,
                                rng: np.random.Generator):
    """Class- and group-balanced exact supervision for the value head.

    Only fully decisive GREEN/YELLOW results are used.  A group is sampled
    uniformly within each class and then one record uniformly within that
    group, so giant proof clusters cannot dominate. Targets are converted
    from GREEN's viewpoint to the network's player-to-move viewpoint.
    """
    by_class_group: dict[int, dict[str, list[ExactRecord]]] = {
        1: defaultdict(list), -1: defaultdict(list)}
    for r in records:
        if r.label in (1, -1):
            by_class_group[r.label][r.group].append(r)
    if not by_class_group[1] or not by_class_group[-1] or batch_size <= 0:
        return None

    states, targets = [], []
    labels = np.array([1, -1], dtype=np.int8)
    for i in range(batch_size):
        label = int(labels[i % 2])
        groups = sorted(by_class_group[label])
        group = groups[int(rng.integers(len(groups)))]
        choices = by_class_group[label][group]
        r = choices[int(rng.integers(len(choices)))]
        states.append(r.state.encode())
        targets.append(float(label * r.state.player))
    order = rng.permutation(batch_size)
    return (np.stack(states)[order].astype(np.float32),
            np.asarray(targets, dtype=np.float32)[order])


def evaluate(records: list[ExactRecord], evaluator, draw_band: float = 0.2,
             return_values: bool = False):
    """Group-averaged, class-balanced agreement with the exact results.

    Four figures are reported because they fail in different ways:

      balanced_accuracy  mean of the per-class recalls; 0.5 for any model
                         that answers the same way regardless of position
      macro_f1           adds precision, averaged over both classes
      auc                threshold-free ranking; the steadiest of the four
      mcc                goes negative when the model is systematically
                         backwards

    All of them weight groups equally rather than records, so no single proof
    line can carry the score.
    """
    if not records:
        empty = {"balanced_accuracy": float("nan"), "groups": 0}
        return (empty, np.zeros(0, dtype=np.float32)) if return_values else empty

    preds = green_values(records, evaluator)

    # correct[class][group] -> (hits, total)
    per_class: dict[int, dict[str, list[int]]] = {1: defaultdict(lambda: [0, 0]),
                                                  -1: defaultdict(lambda: [0, 0]),
                                                  0: defaultdict(lambda: [0, 0])}
    bounds: dict[str, dict[str, list[int]]] = {"le0": defaultdict(lambda: [0, 0]),
                                               "ge0": defaultdict(lambda: [0, 0])}

    for r, p in zip(records, preds):
        if r.label is not None:
            cell = per_class[r.label][r.group]
            cell[1] += 1
            if r.label == 0:
                cell[0] += int(abs(p) < draw_band)
            else:
                cell[0] += int(np.sign(p) == r.label)
        elif r.bound is not None:
            cell = bounds[r.bound][r.group]
            cell[1] += 1
            cell[0] += int(p <= draw_band if r.bound == "le0"
                           else p >= -draw_band)

    def group_mean(table):
        rates = [h / t for h, t in table.values() if t > 0]
        return float(np.mean(rates)) if rates else float("nan")

    green_acc = group_mean(per_class[1])
    yellow_acc = group_mean(per_class[-1])
    draw_acc = group_mean(per_class[0])

    present = [a for a in (green_acc, yellow_acc) if not np.isnan(a)]
    balanced = float(np.mean(present)) if present else float("nan")

    groups = {r.group for r in records}
    decisive_mask = np.array([r.label in (1, -1) for r in records])
    decisive_records = [r for r, m in zip(records, decisive_mask) if m]
    decisive_labels = np.array([r.label for r in decisive_records])
    decisive_scores = preds[decisive_mask]

    if len(decisive_records):
        w = _class_weights(decisive_records, decisive_labels)
        auc = _weighted_auc(decisive_scores, decisive_labels, w)
        tp, fp, tn, fn = _weighted_confusion(decisive_scores, decisive_labels, w)
        macro_f1 = _macro_f1(tp, fp, tn, fn)
        mcc = _mcc(tp, fp, tn, fn)
    else:
        auc = macro_f1 = mcc = float("nan")

    decisive = int(decisive_mask.sum())

    result = {
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "auc": auc,
        "mcc": mcc,
        "green_win_accuracy": green_acc,
        "yellow_win_accuracy": yellow_acc,
        "draw_accuracy": draw_acc,
        "le0_consistency": group_mean(bounds["le0"]),
        "ge0_consistency": group_mean(bounds["ge0"]),
        "records": len(records),
        "decisive": decisive,
        "groups": len(groups),
        "mean_green_value": float(preds.mean()),
    }
    return (result, preds) if return_values else result


def _predicted_outcome(value: float, draw_band: float) -> str:
    if value > draw_band:
        return "GREEN"
    if value < -draw_band:
        return "YELLOW"
    return "DRAW_OR_UNCERTAIN"


def proposition_margin(record: ExactRecord, green_value: float,
                       draw_band: float = 0.2) -> float:
    """Signed margin for an exact proposition; positive means consistent.

    This is deliberately a *margin*, not a probability.  For decisive
    propositions we use the same sign criterion as the validation metrics.
    For draws and one-sided bounds we retain the configured draw band.
    """
    v = float(green_value)
    if record.label == 1:
        return v
    if record.label == -1:
        return -v
    if record.label == 0:
        return draw_band - abs(v)
    if record.bound == "le0":
        return draw_band - v
    if record.bound == "ge0":
        return v + draw_band
    return float("nan")


def proposition_consistent(record: ExactRecord, green_value: float,
                           draw_band: float = 0.2) -> bool:
    v = float(green_value)
    if record.label == 1:
        return v > 0.0
    if record.label == -1:
        return v < 0.0
    if record.label == 0:
        return abs(v) < draw_band
    if record.bound == "le0":
        return v <= draw_band
    if record.bound == "ge0":
        return v >= -draw_band
    return False


def write_proposition_predictions(path, generation: int,
                                  records: list[ExactRecord],
                                  green_values_array: np.ndarray,
                                  draw_band: float = 0.2) -> dict:
    """Append per-proposition network estimates to a CSV time series.

    The file is intended as a diagnostic companion to ``validation.csv``.
    It records what the network currently thinks about every exact DF-PN
    proposition instead of reducing the database to a handful of aggregate
    metrics.  Values are always from GREEN's point of view.
    """
    path = Path(path)
    if len(records) != len(green_values_array):
        raise ValueError("records and green_values_array must have same length")

    fresh = not path.exists()
    supported = 0
    decisive_supported = 0
    decisive_total = 0
    bound_supported = 0
    bound_total = 0

    with path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if fresh:
            writer.writerow([
                "generation", "record_index", "position", "ply", "group",
                "side_to_move", "exact_proposition", "exact_green_value",
                "bound", "net_green_value", "predicted_outcome",
                "proposition_margin", "proposition_consistent",
            ])

        for idx, (record, value) in enumerate(zip(records, green_values_array)):
            value = float(value)
            ok = proposition_consistent(record, value, draw_band)
            margin = proposition_margin(record, value, draw_band)
            supported += int(ok)

            if record.label is not None:
                decisive_total += 1
                decisive_supported += int(ok)
            else:
                bound_total += 1
                bound_supported += int(ok)

            writer.writerow([
                generation,
                idx,
                record.position,
                record.state.ply,
                record.group,
                "GREEN" if record.state.player == 1 else "YELLOW",
                record.proposition,
                "" if record.label is None else record.label,
                "" if record.bound is None else record.bound,
                f"{value:.6f}",
                _predicted_outcome(value, draw_band),
                f"{margin:.6f}",
                int(ok),
            ])

    total = len(records)
    return {
        "proposition_supported": supported,
        "proposition_total": total,
        "proposition_support_rate": supported / total if total else float("nan"),
        "decisive_supported": decisive_supported,
        "decisive_total": decisive_total,
        "decisive_support_rate": (decisive_supported / decisive_total
                                    if decisive_total else float("nan")),
        "bound_supported": bound_supported,
        "bound_total": bound_total,
        "bound_support_rate": (bound_supported / bound_total
                                if bound_total else float("nan")),
    }


def describe(records: list[ExactRecord], name: str) -> str:
    groups = len({r.group for r in records})
    counts = defaultdict(int)
    for r in records:
        counts[r.label if r.label is not None else r.bound] += 1
    parts = [f"{name}: {len(records)} records, {groups} groups"]
    parts.append(f"GREEN win {counts[1]}, YELLOW win {counts[-1]}, "
                 f"draw {counts[0]}, GREEN<=0 {counts['le0']}, "
                 f"GREEN>=0 {counts['ge0']}")
    return "  |  ".join(parts)
