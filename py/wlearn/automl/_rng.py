"""Deterministic LCG PRNG matching JS makeLCG from @wlearn/core."""


def make_lcg(seed=42):
    """Create a seeded LCG PRNG returning floats in [0, 1).

    Identical to JS: s = (s * 1664525 + 1013904223) & 0x7fffffff
    """
    s = [seed & 0x7fffffff]

    def rng():
        s[0] = (s[0] * 1664525 + 1013904223) & 0x7fffffff
        return s[0] / 0x7fffffff

    return rng


def shuffle(arr, rng):
    """Fisher-Yates shuffle matching JS shuffle from @wlearn/core.

    Mutates arr in-place, returns arr.
    Works with lists and numpy arrays.
    """
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = int(rng() * (i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr
