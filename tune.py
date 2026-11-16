"""Walk-forward tuning on REAL recorded trades (paper and/or live).

Replays the entry filters (min edge, min signal strength, time window) over
resolved ENTRY trades stored in the bot's SQLite DB. Parameters are chosen on
the older 70% of trades (train) and judged on the newest 30% (test), so a
setting only looks good if it also holds up on data it was not picked from.

Limitations: filters can only DROP recorded trades; position sizing and risk
state (cooldowns, daily loss) are not re-simulated. Needs a meaningful sample.

Usage:
    python tune.py --mode paper
    python tune.py --mode all --db ./btcbot.db --min-total 100
"""

from __future__ import annotations

import argparse
import itertools
import math

import calibration
from calibration import Entry, load_entries

EDGES = [0.05, 0.07, 0.10, 0.12, 0.15]
STRENGTHS = [0.30, 0.40, 0.50, 0.60]
# (min seconds remaining, max seconds remaining) at entry
WINDOWS = [(0, 300), (30, 240), (60, 240), (60, 210), (90, 210)]
TRAIN_FRACTION = 0.70


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def stats(es: list[Entry]) -> dict:
    n = len(es)
    wins = sum(1 for e in es if e.outcome_correct == 1)
    pnl = sum(e.net_pnl for e in es)
    risked = sum(e.amount_usd + e.hedge_cost for e in es)
    return {
        "n": n,
        "wins": wins,
        "wr": wins / n if n else 0.0,
        "wr_low": wilson_lower(wins, n),
        "pnl": pnl,
        "roi": pnl / risked if risked > 0 else 0.0,
    }


def passes(e: Entry, edge: float, strength: float, window: tuple[int, int]) -> bool:
    t = e.time_remaining_at_entry
    return (
        e.signal_edge >= edge
        and e.signal_strength >= strength
        and window[0] <= t <= window[1]
    )


def fmt(s: dict) -> str:
    return (
        f"n={s['n']:>3} WR={s['wr']:>6.1%} (95% low {s['wr_low']:>5.1%}) "
        f"P&L={s['pnl']:>+8.2f} ROI={s['roi']:>+6.1%}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward filter tuning")
    ap.add_argument("--mode", choices=["live", "paper", "all"], default="paper")
    ap.add_argument("--db", default=None, help="Path to btcbot.db")
    ap.add_argument("--min-total", type=int, default=100)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if args.db:
        calibration.DB_PATH = args.db

    entries = sorted(load_entries(args.mode), key=lambda e: e.created_at)
    n = len(entries)
    print(f"DB: {calibration.DB_PATH}  mode={args.mode}  resolved entries={n}")
    if n < args.min_total:
        print(
            f"Not enough data: need at least {args.min_total} resolved trades "
            "to tune without overfitting. Keep the bot running in paper mode."
        )
        return

    split = int(n * TRAIN_FRACTION)
    train, test = entries[:split], entries[split:]
    min_n = max(15, int(0.10 * len(train)))

    base_train, base_test = stats(train), stats(test)
    print(f"\nBaseline (current filters, no extra restriction)")
    print(f"  train: {fmt(base_train)}")
    print(f"  test : {fmt(base_test)}")

    results = []
    for edge, strength, window in itertools.product(EDGES, STRENGTHS, WINDOWS):
        tr = [e for e in train if passes(e, edge, strength, window)]
        if len(tr) < min_n:
            continue
        te = [e for e in test if passes(e, edge, strength, window)]
        results.append(((edge, strength, window), stats(tr), stats(te)))

    if not results:
        print("\nNo parameter combination has enough train trades.")
        return

    results.sort(key=lambda r: r[1]["roi"], reverse=True)
    print(f"\nTop {args.top} by TRAIN ROI (min {min_n} trades; {len(results)} combos tried)")
    for (edge, strength, window), tr, te in results[: args.top]:
        print(f"\n  min_edge={edge:.2f} min_strength={strength:.2f} window={window}")
        print(f"    train: {fmt(tr)}")
        print(f"    test : {fmt(te)}")

    (edge, strength, window), tr, te = results[0]
    robust = te["n"] >= 10 and te["roi"] > base_test["roi"] and te["roi"] > 0
    print("\nVerdict")
    if robust:
        print(
            "  Best train setting also beats the baseline on the held-out test slice. "
            "Still treat it as a hypothesis: confirm on fresh paper data before going live."
        )
        print(f"  BOT_MIN_EDGE={edge}")
        print(f"  BOT_MIN_SIGNAL_STRENGTH={strength}")
        print(f"  BOT_WARMUP_SEC={300 - window[1]}")
        print(f"  BOT_COOLDOWN_SEC={window[0]}")
    else:
        print(
            "  No robust improvement: the best train setting does not clearly beat the "
            "baseline on unseen data (or the test sample is too small). Do not change "
            "parameters yet; collect more trades."
        )
    print(
        f"  Note: {len(results)} combinations were compared, so some apparent winners "
        "are expected by chance alone."
    )


if __name__ == "__main__":
    main()
