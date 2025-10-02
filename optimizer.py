
import numpy as np
import pandas as pd
from typing import Sequence

def alloc_sqrt_model(b: np.ndarray, v: np.ndarray, B: float) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    v = np.asarray(v, dtype=float)
    w = (v * b) ** 2
    if not np.isfinite(B) or B < 0 or w.sum() <= 0:
        return np.zeros_like(b)
    return B * w / w.sum()

def optimal_table_sqrt(platform_df: pd.DataFrame, budget: float) -> pd.DataFrame:
    b = platform_df['b'].to_numpy(float)
    v = platform_df['v'].to_numpy(float)
    x = alloc_sqrt_model(b, v, budget)
    leads = platform_df['a'].to_numpy(float) + b * np.sqrt(np.maximum(x, 0.0))
    share = x / x.sum() if x.sum() > 0 else np.zeros_like(x)
    marginal = v * (b / (2.0 * np.sqrt(np.maximum(x, 1e-12))))
    out = platform_df[['platform', 'v']].copy()
    out['leads'] = leads
    out['spend'] = x
    out['share'] = share
    out['marginal_at_opt'] = marginal
    return out

def _deterministic_weights_from_names(names: Sequence[str], seed_base: int) -> np.ndarray:
    # Produce stable weights based on platform names for reproducibility
    seeds = [abs(hash((seed_base, n))) % (2**32 - 1) for n in names]
    vals = []
    for s in seeds:
        rng = np.random.default_rng(s)
        vals.append(rng.uniform(0.5, 1.5))
    raw = np.array(vals, dtype=float)
    # Normalize L2 so that sum s_i^2 = 1
    s = raw / np.linalg.norm(raw)
    return s

def build_fixed_platforms(product_name: str,
                          a_prod: float,
                          b_prod: float,
                          platform_names: Sequence[str],
                          v_values: Sequence[float]) -> pd.DataFrame:
    '''
    Build per-platform parameters from product-level (a,b) and user-provided v per platform.
    - a is split evenly among platforms
    - b_i = b_prod * s_i where sum s_i^2 = 1 with deterministic s_i from platform names
    '''
    if len(platform_names) != len(v_values):
        raise ValueError('platform_names and v_values must have same length')
    n = len(platform_names)
    s = _deterministic_weights_from_names(platform_names, seed_base=abs(hash(product_name)) % (2**32 - 1))
    b_i = b_prod * s
    a_i = np.full(n, float(a_prod) / n)
    df = pd.DataFrame({
        'product': product_name,
        'platform': list(platform_names),
        'a': a_i,
        'b': b_i,
        'v': np.asarray(v_values, dtype=float)
    })
    return df
