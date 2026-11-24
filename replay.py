"""Reconstruct the three reported SIB-200 families using saved scores and weights."""
# Distribution edit: corrected the publication-status wording in the module description.
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = "eng_Latn"
METRICS = ("mean_score_gap", "fpr_gap", "recall_gap")
EXPECTED = {
    "historical": "E2_Historical_Fixed_Simultaneous_Bands.csv",
    "tuned": "E2_Joint_Simultaneous_Bands.csv",
    "fixed_C1": "E2_Fixed_C_Sensitivity_Bands.csv",
}


def cutoff(scores, labels, counts):
    # Absent scores must also be absent from the candidate midpoint grid.
    x, y = np.repeat(scores, counts), np.repeat(labels, counts)
    neg, pos = np.sort(x[~y]), np.sort(x[y])
    if not len(neg) or not len(pos):
        return np.nan
    distinct = np.unique(x)
    mid = distinct[:-1] + np.diff(distinct) / 2
    candidates = np.r_[np.nextafter(distinct[0], -np.inf),
                       np.where(mid <= distinct[:-1], distinct[1:], mid),
                       np.nextafter(distinct[-1], np.inf)]
    fp = len(neg) - neg.searchsorted(candidates)
    tp = len(pos) - pos.searchsorted(candidates)
    feasible = 20 * fp <= len(neg)
    return np.median(candidates[feasible & (tp == tp[feasible].max())])


def weights_for(role, roster, keys, streams):
    groups = keys[role]
    ids = pd.Index(groups).get_indexer(roster.dependence_group)
    if min(ids) < 0 or set(roster.dependence_group) != set(groups):
        raise ValueError(f"Group mismatch for {role}")
    w = np.stack(streams[role + "_weights"]).astype(int)
    assert np.all(w >= 0) and np.all(w.sum(axis=1) == len(groups))
    # First row is the observed estimate with each document appearing once.
    return np.vstack([np.ones(len(ids), dtype=int), w[:, ids]])


def paired(scores, labels, weights, cuts):
    # The first score column is always English. Sign is target minus English.
    delta = scores[:, 1:] - scores[:, [0]]
    calls = (scores[None, :, :] >= cuts[:, None, None]).astype(np.int8)
    changes = calls[:, :, 1:] - calls[:, :, [0]]
    mean = np.einsum("di,ij->dj", weights, delta) / weights.sum(axis=1)[:, None]
    gaps = [mean]
    for mask in (~labels, labels):
        w = weights[:, mask]
        num = np.einsum("di,dij->dj", w, changes[:, mask, :])
        denom = w.sum(axis=1)[:, None]
        valid = (denom > 0) & np.isfinite(cuts[:, None])
        gaps.append(np.divide(num, denom, where=valid, out=np.full(num.shape, np.nan)))
    return np.stack(gaps, axis=-1)


def verify_family(name, allocation, arrays, columns, expected, rows):
    values = np.concatenate(arrays, axis=1)
    observed, simulated = values[0], values[1:]
    if name == "historical":
        simulated = simulated[:2000]
    complete = np.isfinite(simulated).all(axis=1)
    draws = simulated[complete][:2000]
    assert draws.shape == (2000, 84)
    dev = np.abs(draws - observed)
    maxima = np.max(dev, axis=1)
    radius = np.quantile(maxima, 1 - .05 / 3, method="linear")
    base = expected[name].query("allocation == @allocation").set_index(["topic", "language", "endpoint"])
    assert len(base) == len(columns) == 84 and set(columns) == set(base.index)
    base = base.loc[columns]
    lo, hi = ("lo", "hi") if name == "historical" else ("simultaneous_lo", "simultaneous_hi")
    for col, value in (("estimate", observed), ("critical_radius", radius),
                       (lo, observed - radius), (hi, observed + radius)):
        np.testing.assert_allclose(base[col], value, atol=1e-12, rtol=0)
    for key, point in zip(columns, observed):
        rows.append((name, allocation, *key, point, radius, point-radius, point+radius))
    if name != "historical":
        assert base.attempts.eq(len(simulated)).all()
        assert base.incomplete_draws.eq((~complete).sum()).all()
    print(f"{allocation} {name}: {len(draws)} complete, radius {radius:.9f}")
    if name == "tuned":
        by_type = dev.reshape(2000, -1, 3).max(axis=1)
        print("  recall-only diagnostic radius:", f"{np.quantile(by_type[:, 2], 1-.05/3):.9f}")
        print("  recall strictly largest:", np.sum(by_type[:, 2] > by_type[:, :2].max(axis=1)))
    if allocation == "seed_20260803":
        bracket = np.sort(maxima)[[1954, 1977]]
        for topic, target in [("health", "kaz_Cyrl"), ("travel", "amh_Ethi"), ("travel", "kaz_Cyrl")]:
            i = columns.index((topic, target, "recall_gap"))
            print(f"  {topic}/{target} upper pp {100*(observed[i]+radius):+.2f}; "
                  f"MC bracket {100*(observed[i]+bracket[0]):+.2f}, {100*(observed[i]+bracket[1]):+.2f}")
    return {key[:2] for key, upper in zip(columns, observed + radius)
            if key[2] == "recall_gap" and upper < 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Write reconstructed rows to a new CSV.")
    args = parser.parse_args()
    rows = []
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {path}")
    items = pd.read_csv(ROOT / "items.csv").set_index("item_id")
    assert items.index.is_unique
    scores = pd.read_parquet(ROOT / "scores.parquet")
    weights = pd.read_parquet(ROOT / "E2_Draw_Group_Weights.parquet")
    keys = json.loads((ROOT / "E2_Group_Keys.json").read_text())
    expected = {name: pd.read_csv(ROOT / "expected" / file) for name, file in EXPECTED.items()}
    supported = {name: [] for name in EXPECTED}
    languages = [SOURCE] + sorted(set(scores.language) - {SOURCE})
    assert len(items) == 799 and len(languages) == 8 and len(scores) == 76800
    for allocation, groups in keys.items():
        assert items.groupby("dependence_group")[allocation].nunique().eq(1).all()
        cal = items[items[allocation] == "calibration"]
        test = items[items[allocation] == "test"]
        streams = weights.query("allocation == @allocation").sort_values("draw_index")
        np.testing.assert_array_equal(streams.draw_index, np.arange(len(streams)))
        wc = weights_for("calibration", cal, groups, streams)
        wt = weights_for("test", test, groups, streams)
        blocks = {name: [] for name in EXPECTED}
        columns = []
        for topic in sorted(scores.topic.unique()):
            columns.extend((topic, lang, end) for lang in languages[1:] for end in METRICS)
            for variant in ("tuned", "fixed_C1"):
                case = scores.query("allocation == @allocation and topic == @topic and variant == @variant")
                matrix = case.pivot(index="item_id", columns="language", values="score")
                assert set(matrix.index) == set(cal.index) | set(test.index)
                assert len(matrix.columns) == 8 and np.isfinite(matrix.to_numpy()).all()
                source = matrix.loc[cal.index, SOURCE].to_numpy()
                x = matrix.loc[test.index, languages].to_numpy()
                y = test.topic.eq(topic).to_numpy()
                yc = cal.topic.eq(topic).to_numpy()
                cuts = np.array([cutoff(source, yc, w) for w in wc])
                attempted = np.r_[True, streams["attempted_" + variant].to_numpy(bool)]
                gaps = paired(x, y, wt[attempted], cuts[attempted])
                blocks[variant].append(gaps.reshape(len(gaps), -1))
                if variant == "tuned":
                    hist = paired(x, y, wt[:2001], np.full(2001, cuts[0]))
                    blocks["historical"].append(hist.reshape(2001, -1))
        for name in EXPECTED:
            supported[name].append(verify_family(name, allocation, blocks[name], columns, expected, rows))
    for name, sets in supported.items():
        print(name, "lower recall in every allocation:", sorted(set.intersection(*sets)))
    if args.export:
        with args.export.open("x", newline="", encoding="utf-8") as dst:
            writer = csv.writer(dst)
            writer.writerow(("analysis","allocation","topic","language","endpoint","estimate",
                             "critical_radius","simultaneous_lo","simultaneous_hi"))
            writer.writerows(rows)
    print("Verified all 756 endpoint rows from keyed inputs.")


if __name__ == "__main__":
    main()

