import numpy as np
import pandas as pd
from typing import Sequence

# -------------------------------
# Asignación sin restricciones (legacy)
# -------------------------------
def alloc_sqrt_model(b: np.ndarray, v: np.ndarray, B: float) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    v = np.asarray(v, dtype=float)
    w = (v * b) ** 2
    if not np.isfinite(B) or B < 0 or w.sum() <= 0:
        return np.zeros_like(b)
    return B * w / w.sum()

# -------------------------------
# Asignación con mínimos y máximos
# -------------------------------
def _sanitize_bounds(n, mins, maxs):
    """
    Devuelve (lo, hi) con:
      - lo >= 0, NaN/None -> 0
      - hi >= lo, NaN/None/inf -> +inf
    """
    lo = np.zeros(n, dtype=float) if mins is None else np.asarray(mins, dtype=float)
    hi = np.full(n, np.inf, dtype=float) if maxs is None else np.asarray(maxs, dtype=float)

    lo = np.where(np.isfinite(lo) & (lo > 0), lo, np.where(np.isfinite(lo) & (lo <= 0), 0.0, 0.0))
    hi = np.where(np.isfinite(hi) & (hi >= 0), hi, np.inf)

    # asegurar hi >= lo
    hi = np.maximum(hi, lo)
    return lo, hi

def alloc_sqrt_with_bounds(b: np.ndarray,
                           v: np.ndarray,
                           B: float,
                           mins=None,
                           maxs=None) -> np.ndarray:
    """
    Asigna con modelo raíz y pesos w = (v*b)^2 cumpliendo límites por plataforma.
    Reglas de esquina:
      - B <= 0 -> todo 0.
      - sum(mins) > B -> asigna proporcional a mins (no es factible cumplirlos).
      - sum(maxs) < B -> llena hasta máximos y deja remanente sin asignar.
      - Si w.sum() == 0 -> reparte el remanente equitativamente entre plataformas libres.
    """
    b = np.asarray(b, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(b)
    if not np.isfinite(B) or B <= 0 or n == 0:
        return np.zeros(n, dtype=float)

    lo, hi = _sanitize_bounds(n, mins, maxs)

    min_sum = float(np.nansum(lo))
    if min_sum >= B and min_sum > 0:
        # No alcanza para cumplir mínimos: distribuir proporcional a los mínimos
        return (B / min_sum) * lo

    # Punto de partida: mínimos
    x = lo.copy()
    remaining = float(B - min_sum)
    if remaining <= 0:
        return x

    # Capacidad adicional por plataforma
    cap = hi - lo  # puede incluir +inf
    free = np.arange(n)  # índices candidatos

    # Pesos
    w_full = (v * b) ** 2
    # En plataformas con cap=0 no se podrá asignar de todos modos
    w_full = np.where(cap > 0, w_full, 0.0)

    # Si todos los pesos son 0, repartir igual entre las que tengan capacidad
    if not np.isfinite(w_full.sum()) or w_full.sum() <= 0:
        # Asignación equitativa con clipping por cap
        alive = cap > 0
        if not np.any(alive):
            return x  # no hay capacidad
        shares = np.zeros(n, dtype=float)
        shares[alive] = 1.0 / alive.sum()
        # Water-filling simple con shares iguales
        while remaining > 1e-12 and np.any(cap > 1e-12):
            step = remaining
            add = step * shares
            # clip por cap
            over = add > cap
            if np.any(over):
                add = np.minimum(add, cap)
            x += add
            remaining -= float(add.sum())
            cap -= add
            shares = np.where(cap > 1e-12, shares, 0.0)
            tot = shares.sum()
            if tot <= 0:
                break
            shares = shares / tot
        return x

    # Water-filling ponderado por w = (v*b)^2
    # Reasignando iterativamente cuando alguna plataforma toca su máximo.
    active = cap > 1e-12
    while remaining > 1e-12 and np.any(active):
        w = np.where(active, w_full, 0.0)
        W = w.sum()
        if W <= 0:
            # fallback: repartir equitativo entre activos
            cnt = float(active.sum())
            add = np.where(active, remaining / cnt, 0.0)
        else:
            add = remaining * (w / W)

        # Clip por capacidad
        add = np.minimum(add, cap)
        total_add = float(add.sum())
        if total_add <= 1e-12:
            break

        x += add
        remaining -= total_add
        cap -= add
        active = cap > 1e-12

    # Si quedó remanente pero no hay capacidad (sum(max) < B), lo ignoramos.
    return x

# -------------------------------
# Tablas de salida (legacy + bounds)
# -------------------------------
def optimal_table_sqrt(platform_df: pd.DataFrame, budget: float) -> pd.DataFrame:
    b = platform_df['b'].to_numpy(float)
    v = platform_df['v'].to_numpy(float)
    x = alloc_sqrt_model(b, v, budget)
    resultados = platform_df['a'].to_numpy(float) + b * np.sqrt(np.maximum(x, 0.0))
    share = x / x.sum() if x.sum() > 0 else np.zeros_like(x)
    marginal = v * (b / (2.0 * np.sqrt(np.maximum(x, 1e-12))))
    out = platform_df[['platform', 'v']].copy()
    out['resultados'] = resultados
    out['spend'] = x
    out['share'] = share
    out['marginal_at_opt'] = marginal
    return out

def optimal_table_sqrt_bounds(platform_df: pd.DataFrame,
                              budget: float,
                              mins=None,
                              maxs=None) -> pd.DataFrame:
    """
    Igual a optimal_table_sqrt, pero respetando mínimos y máximos por plataforma.
    """
    b = platform_df['b'].to_numpy(float)
    v = platform_df['v'].to_numpy(float)
    lo = None if mins is None else np.asarray(mins, dtype=float)
    hi = None if maxs is None else np.asarray(maxs, dtype=float)

    x = alloc_sqrt_with_bounds(b, v, budget, lo, hi)

    resultados = platform_df['a'].to_numpy(float) + b * np.sqrt(np.maximum(x, 0.0))
    total_x = x.sum()
    share = x / total_x if total_x > 0 else np.zeros_like(x)
    marginal = np.where(x > 0, v * (b / (2.0 * np.sqrt(x))), np.inf)

    out = platform_df[['platform', 'v']].copy()
    out['resultados'] = resultados
    out['spend'] = x
    out['share'] = share
    out['marginal_at_opt'] = marginal
    # Para transparencia, devolvemos también los límites (si estaban)
    if mins is not None:
        out['min_spend'] = lo
    if maxs is not None:
        out['max_spend'] = hi
    return out

# -------------------------------
# Construcción de plataformas
# -------------------------------
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
