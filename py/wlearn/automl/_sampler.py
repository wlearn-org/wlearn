"""Search space sampling matching JS automl/sampler.js."""

import math

from ._rng import make_lcg


def sample_param(param, rng):
    """Sample a single value from a SearchParam definition."""
    ptype = param['type']
    if ptype == 'categorical':
        return param['values'][int(rng() * len(param['values']))]
    elif ptype == 'uniform':
        return param['low'] + rng() * (param['high'] - param['low'])
    elif ptype == 'log_uniform':
        return math.exp(
            math.log(param['low']) + rng() * (math.log(param['high']) - math.log(param['low']))
        )
    elif ptype == 'int_uniform':
        return param['low'] + int(rng() * (param['high'] - param['low'] + 1))
    elif ptype == 'int_log_uniform':
        return round(math.exp(
            math.log(param['low']) + rng() * (math.log(param['high']) - math.log(param['low']))
        ))
    else:
        raise ValueError(f'Unknown SearchParam type: "{ptype}"')


def sample_config(space, rng):
    """Sample a complete config from a SearchSpace, respecting conditions."""
    config = {}
    keys = list(space.keys())

    # First pass: non-conditional params
    for key in keys:
        if 'condition' not in space[key] or space[key]['condition'] is None:
            config[key] = sample_param(space[key], rng)

    # Second pass: conditional params
    for key in keys:
        cond = space[key].get('condition')
        if not cond:
            continue
        satisfied = True
        for ck, cv in cond.items():
            if config.get(ck) != cv:
                satisfied = False
                break
        if satisfied:
            config[key] = sample_param(space[key], rng)

    return config


def random_configs(space, n, seed=42):
    """Generate n random configs from a SearchSpace."""
    rng = make_lcg(seed)
    configs = []
    for _ in range(n):
        configs.append(sample_config(space, rng))
    return configs


def grid_configs(space, steps=5):
    """Enumerate grid points from a SearchSpace."""
    keys = list(space.keys())
    if not keys:
        return [{}]

    non_cond = [k for k in keys if not space[k].get('condition')]
    cond_keys = [k for k in keys if space[k].get('condition')]

    value_arrays = [_discretize(space[k], steps) for k in non_cond]

    # Cartesian product of non-conditional params
    combos = [{}]
    for i, key in enumerate(non_cond):
        vals = value_arrays[i]
        new_combos = []
        for combo in combos:
            for v in vals:
                new_combos.append({**combo, key: v})
        combos = new_combos

    # Add conditional params where conditions are met
    for combo in combos:
        for key in cond_keys:
            cond = space[key]['condition']
            satisfied = True
            for ck, cv in cond.items():
                if combo.get(ck) != cv:
                    satisfied = False
                    break
            if satisfied:
                vals = _discretize(space[key], steps)
                combo[key] = vals[len(vals) // 2]

    return combos


def _discretize(param, steps):
    """Discretize a param into grid values."""
    ptype = param['type']
    if ptype == 'categorical':
        return list(param['values'])
    elif ptype == 'uniform':
        return [
            param['low'] + (param['high'] - param['low']) * i / max(1, steps - 1)
            for i in range(steps)
        ]
    elif ptype == 'log_uniform':
        log_low = math.log(param['low'])
        log_high = math.log(param['high'])
        return [
            math.exp(log_low + (log_high - log_low) * i / max(1, steps - 1))
            for i in range(steps)
        ]
    elif ptype == 'int_uniform':
        rng_size = param['high'] - param['low'] + 1
        if rng_size <= steps:
            return list(range(param['low'], param['high'] + 1))
        arr = [
            param['low'] + round((param['high'] - param['low']) * i / max(1, steps - 1))
            for i in range(steps)
        ]
        return sorted(set(arr))
    elif ptype == 'int_log_uniform':
        log_low = math.log(param['low'])
        log_high = math.log(param['high'])
        arr = [
            round(math.exp(log_low + (log_high - log_low) * i / max(1, steps - 1)))
            for i in range(steps)
        ]
        return sorted(set(arr))
    else:
        raise ValueError(f'Unknown SearchParam type: "{ptype}"')
