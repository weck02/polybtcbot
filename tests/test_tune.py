import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tune  # noqa: E402


def make_entry(**kw):
    base = dict(
        outcome_correct=1,
        net_pnl=1.0,
        amount_usd=2.0,
        hedge_cost=0.0,
        signal_edge=0.10,
        signal_strength=0.50,
        time_remaining_at_entry=120,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_wilson_lower_zero_n():
    assert tune.wilson_lower(0, 0) == 0.0


def test_wilson_lower_below_point_estimate():
    low = tune.wilson_lower(60, 100)
    assert 0.0 < low < 0.60


def test_wilson_lower_grows_with_sample_size():
    assert tune.wilson_lower(600, 1000) > tune.wilson_lower(60, 100)


def test_stats_empty():
    s = tune.stats([])
    assert s["n"] == 0 and s["wr"] == 0.0 and s["roi"] == 0.0


def test_stats_basic():
    es = [
        make_entry(),
        make_entry(outcome_correct=0, net_pnl=-2.0),
    ]
    s = tune.stats(es)
    assert s["n"] == 2
    assert s["wins"] == 1
    assert s["wr"] == 0.5
    assert s["pnl"] == -1.0
    assert abs(s["roi"] - (-1.0 / 4.0)) < 1e-9


def test_passes_accepts_matching_entry():
    assert tune.passes(make_entry(), 0.10, 0.50, (60, 240))


def test_passes_rejects_low_edge():
    assert not tune.passes(make_entry(signal_edge=0.05), 0.10, 0.30, (0, 300))


def test_passes_rejects_low_strength():
    assert not tune.passes(make_entry(signal_strength=0.2), 0.05, 0.30, (0, 300))


def test_passes_rejects_outside_window():
    assert not tune.passes(make_entry(time_remaining_at_entry=30), 0.05, 0.30, (60, 240))
    assert not tune.passes(make_entry(time_remaining_at_entry=250), 0.05, 0.30, (60, 240))
