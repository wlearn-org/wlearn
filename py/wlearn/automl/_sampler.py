"""Search space sampling matching JS automl/sampler.js."""

import math

from ..rng import make_lcg
from ._conditions import condition_order, condition_satisfied


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
    for key in condition_order(space):
        if condition_satisfied(space[key].get('condition'), config):
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
    combos = [{}]
    for key in condition_order(space):
        vals = _discretize(space[key], steps)
        combos = [expanded for combo in combos for expanded in (
            [{**combo, key: value} for value in vals]
            if condition_satisfied(space[key].get('condition'), combo) else [combo])]

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
