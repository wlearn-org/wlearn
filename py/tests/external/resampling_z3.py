#!/usr/bin/env python3
"""Optional Z3 proofs for resampling interval arithmetic.

This follows Polygrad's pattern for solver checks: keep Z3 out of the normal
unit-test path, but make it easy to run as an external proof smoke when changing
index/window formulas.
"""

from __future__ import annotations

import sys

try:
    import z3
except Exception as exc:  # pragma: no cover - exercised in missing-dep envs
    raise SystemExit(
        "z3-solver is required for this external proof check. "
        "Install a safe version such as: pip install 'z3-solver<4.15.4'"
    ) from exc


def prove(name: str, constraints, obligations) -> None:
    for label, predicate in obligations:
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*constraints)
        solver.add(z3.Not(predicate))
        check = solver.check()
        if check == z3.sat:
            print(f"z3 proof failed: {name}: {label}", file=sys.stderr)
            print(solver.model(), file=sys.stderr)
            raise SystemExit(1)
        if check == z3.unknown:
            print(f"z3 proof unknown: {name}: {label}: {solver.reason_unknown()}", file=sys.stderr)
            raise SystemExit(2)


def prove_complete_row_window() -> None:
    n, lookback, assess_start, assess_stop, step, skip, anchor = z3.Ints(
        "n lookback assess_start assess_stop step skip anchor"
    )
    train_start = anchor - lookback + 1
    test_start = anchor + assess_start
    test_end = anchor + assess_stop
    train_len = anchor - train_start + 1
    test_len = test_end - test_start + 1

    constraints = [
        n >= 2,
        lookback >= 1,
        assess_start >= 1,
        assess_stop >= assess_start,
        step >= 1,
        skip >= 0,
        anchor >= 0,
        anchor < n,
        anchor >= lookback - 1,
        test_start < n,
        test_end < n,
    ]
    prove(
        "complete sliding_window",
        constraints,
        [
            ("train starts in bounds", train_start >= 0),
            ("train is non-empty", train_start <= anchor),
            ("test starts after train", anchor < test_start),
            ("test is non-empty", test_start <= test_end),
            ("test ends in bounds", test_end < n),
            ("train length equals lookback", train_len == lookback),
            ("test length equals assessment width", test_len == assess_stop - assess_start + 1),
        ],
    )


def prove_incomplete_row_window() -> None:
    n, lookback, assess_start, assess_stop, step, skip, anchor = z3.Ints(
        "n_i lookback_i assess_start_i assess_stop_i step_i skip_i anchor_i"
    )
    raw_train_start = anchor - lookback + 1
    train_start = z3.If(raw_train_start < 0, 0, raw_train_start)
    full_test_end = anchor + assess_stop
    test_start = anchor + assess_start
    test_end = z3.If(full_test_end > n - 1, n - 1, full_test_end)
    train_len = anchor - train_start + 1
    test_len = test_end - test_start + 1
    max_test_len = assess_stop - assess_start + 1

    constraints = [
        n >= 2,
        lookback >= 1,
        assess_start >= 1,
        assess_stop >= assess_start,
        step >= 1,
        skip >= 0,
        anchor >= 0,
        anchor < n,
        test_start < n,
    ]
    prove(
        "incomplete sliding_window",
        constraints,
        [
            ("train starts in bounds", train_start >= 0),
            ("train is non-empty", train_start <= anchor),
            ("train ends in bounds", anchor < n),
            ("test starts after train", anchor < test_start),
            ("test is non-empty", test_start <= test_end),
            ("test ends in bounds", test_end < n),
            ("train length is capped by lookback", train_len <= lookback),
            ("test length is capped by assessment width", test_len <= max_test_len),
        ],
    )


def prove_value_window_intervals() -> None:
    first, last, anchor, lookback, assess_start, assess_stop = z3.Reals(
        "first last anchor lookback assess_start assess_stop"
    )
    train_min = anchor - lookback
    test_min = anchor + assess_start
    test_max = anchor + assess_stop
    train_floor = z3.If(first > train_min, first, train_min)
    test_ceiling = z3.If(last < test_max, last, test_max)

    constraints = [
        first <= anchor,
        anchor <= last,
        lookback > 0,
        assess_start > 0,
        assess_stop >= assess_start,
        test_min <= last,
    ]
    prove(
        "sliding_index/sliding_period value windows",
        constraints,
        [
            ("train interval starts before anchor", train_floor <= anchor),
            ("test interval starts after train interval", anchor < test_min),
            ("test interval is non-empty when emitted", test_min <= test_ceiling),
            ("test interval ends in index domain", test_ceiling <= last),
        ],
    )


def main() -> int:
    prove_complete_row_window()
    prove_incomplete_row_window()
    prove_value_window_intervals()
    print("z3 resampling proofs: complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
