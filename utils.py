import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# --- Carga de parámetros ---
def load_params(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        clean = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                clean[k] = (float(v[0]), float(v[1]))
        return clean
    except Exception:
        return {}

# --- Carga de datos (acepta LONG o WIDE y devuelve LONG estandarizado) ---
# Formato LONG esperado (mínimo): date | product | investment | leads
# Formato WIDE (alternativo): "Product X - Investment", "Product X - Conversions", + fecha
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # 1) Normalizaciones de nombres básicas
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "date"})

    cols_lower = {c.lower(): c for c in df.columns}
    # posibles alias
    date_col  = next((cols_lower[c] for c in ["date", "fecha", "dt"] if c in cols_lower), None)
    prod_col  = next((cols_lower[c] for c in ["product", "producto", "item"] if c in cols_lower), None)
    inv_col   = next((cols_lower[c] for c in ["investment", "inversion", "spend", "gasto", "ad_spend", "inv"] if c in cols_lower), None)
    leads_col = next((cols_lower[c] for c in ["leads", "conversions", "objetivo", "ventas"] if c in cols_lower), None)

    # 2) Caso LONG ya provisto
    if prod_col and inv_col and leads_col:
        out = df.copy()
        out = out.rename(columns={
            prod_col: "product",
            inv_col: "investment",
            leads_col: "leads"
        })
        if date_col:
            out = out.rename(columns={date_col: "date"})
        else:
            # si no hay fecha, creamos una artificial incremental
            out["date"] = pd.RangeIndex(start=1, stop=len(out) + 1, step=1)
        # tipos
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["investment"] = pd.to_numeric(out["investment"], errors="coerce")
        out["leads"] = pd.to_numeric(out["leads"], errors="coerce")
        out = out[["date", "product", "investment", "leads"]].dropna(subset=["investment", "leads"])
        return out

    # 3) Caso WIDE → LONG (por compatibilidad hacia atrás)
    inv_cols = [c for c in df.columns if str(c).endswith(" - Investment")]
    conv_cols = [c for c in df.columns if str(c).endswith(" - Conversions")]

    products = {}
    for c in inv_cols:
        prod = str(c).replace(" - Investment", "")
        products.setdefault(prod, {})["inv"] = c
    for c in conv_cols:
        prod = str(c).replace(" - Conversions", "")
        products.setdefault(prod, {})["conv"] = c

    long_parts = []
    for prod, cols in products.items():
        inv = cols.get("inv")
        conv = cols.get("conv")
        if inv is None or conv is None:
            continue
        use_cols = [inv, conv]
        if date_col:
            use_cols = [date_col] + use_cols
        tmp = df[use_cols].copy()
        ren = {inv: "investment", conv: "leads"}
        if date_col:
            ren[date_col] = "date"
        tmp = tmp.rename(columns=ren)
        tmp["product"] = prod
        long_parts.append(tmp)

    if not long_parts:
        # no pudimos detectar estructura, devolvemos DF vacío estándar
        return pd.DataFrame(columns=["date", "product", "investment", "leads"])

    out = pd.concat(long_parts, ignore_index=True)
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
    out["investment"] = pd.to_numeric(out["investment"], errors="coerce")
    out["leads"] = pd.to_numeric(out["leads"], errors="coerce")
    out = out[["date", "product", "investment", "leads"]].dropna(subset=["investment", "leads"])
    return out

# --- Detección (ya estandarizado a LONG) ---
def detect_columns(df: pd.DataFrame):
    return "date", "product", "investment", "leads"

# --- Modelo: y = a + b * sqrt(x) ---
def predict_leads(inversion_array, intercepto, coef):
    inv = np.maximum(np.asarray(inversion_array, dtype=float), 0.0)
    return intercepto + coef * np.sqrt(inv)

def build_model_curve(inv_values, intercepto, coef, num=200):
    inv_min = float(np.nanmin(inv_values)) if len(inv_values) else 0.0
    inv_max = float(np.nanmax(inv_values)) if len(inv_values) else 1000.0
    if not np.isfinite(inv_min):
        inv_min = 0.0
    if (not np.isfinite(inv_max)) or (inv_max <= inv_min):
        inv_max = inv_min + 1.0
    x = np.linspace(inv_min, inv_max, num=num)
    y = predict_leads(x, intercepto, coef)
    return pd.DataFrame({"inversion": x, "leads_modelo": y})

# Agregar al final de utils.py (junto a predict_leads y build_model_curve)
def investment_for_target_leads(target_leads, intercepto, coef):
    """
    Devuelve la inversión necesaria para alcanzar 'target_leads'
    bajo y = a + b*sqrt(x). Reglas:
      - Si coef <= 0 -> NaN (modelo inválido para invertir).
      - Si target_leads <= a -> 0 (según el modelo, no hace falta inversión).
    """
    import numpy as np

    if coef is None or not np.isfinite(coef) or coef <= 0:
        return np.nan
    if intercepto is None or not np.isfinite(intercepto):
        return np.nan

    gap = float(target_leads) - float(intercepto)
    if gap <= 0:
        return 0.0  # el modelo dice que con 0 inversión ya estás en ese nivel o superior

    inv = (gap / coef) ** 2
    return float(max(inv, 0.0))

def incremental_cost_one_more_lead_at_target(target_leads, intercepto, coef):
    """
    Costo DISCRETO de obtener 1 lead extra pasando de X -> X+1 leads.
    Devuelve inv(X+1) - inv(X). Maneja casos borde.
    """
    if coef is None or not np.isfinite(coef) or coef <= 0:
        return np.nan
    if intercepto is None or not np.isfinite(intercepto):
        return np.nan

    # Si X <= a, el modelo dice que X se logra con 0 inversión.
    # Igual calculamos el salto a X+1.
    inv_x = investment_for_target_leads(target_leads, intercepto, coef)
    inv_x1 = investment_for_target_leads(target_leads + 1.0, intercepto, coef)
    if not (np.isfinite(inv_x) and np.isfinite(inv_x1)):
        return np.nan
    return float(max(inv_x1 - inv_x, 0.0))

def marginal_cost_per_lead_at_target(target_leads, intercepto, coef):
    """
    Costo MARGINAL aproximado (derivada) en el punto X:
      d(inv)/d(leads) = 2*sqrt(inv(X)) / b
    """
    if coef is None or not np.isfinite(coef) or coef <= 0:
        return np.nan
    inv_x = investment_for_target_leads(target_leads, intercepto, coef)
    if not np.isfinite(inv_x) or inv_x < 0:
        return np.nan
    # Evitar singularidad en 0 con una epsilon chica:
    inv_x = max(inv_x, 1e-9)
    return float(2.0 * np.sqrt(inv_x) / coef)

def optimal_point_for_lead_value(lead_value, intercepto, coef):
    """
    Dado el valor por lead (V), retorna el punto óptimo donde el costo marginal = V.
    Fórmulas (modelo y = a + b*sqrt(x)):
      X* = a + (V * b^2) / 2
      inv* = (V * b / 2)^2
      cmg* = V  (por construcción)
    También calcula métricas útiles en ese punto: leads, inversión, CPL nominal,
    CPL incremental (sobre (X*-a)), revenue, cost y profit.
    """
    if lead_value is None or not np.isfinite(lead_value) or lead_value <= 0:
        return {"ok": False, "msg": "Valor por lead (V) debe ser > 0."}
    if coef is None or not np.isfinite(coef) or coef <= 0:
        return {"ok": False, "msg": "Coeficiente b inválido (debe ser > 0)." }
    if intercepto is None or not np.isfinite(intercepto):
        return {"ok": False, "msg": "Intercepto a inválido."}

    a = float(intercepto); b = float(coef); V = float(lead_value)

    X_star   = a + (V * (b**2)) / 2.0
    inv_star = (V * b / 2.0) ** 2
    cmg_star = V  # por definición en el óptimo

    # Chequeos y métricas
    incr_leads = max(X_star - a, 0.0)
    cpl_nom    = inv_star / X_star if X_star > 0 else np.nan
    cpl_incr   = inv_star / incr_leads if incr_leads > 0 else np.nan

    revenue    = V * X_star
    cost       = inv_star
    profit     = revenue - cost

    return {
        "ok": True,
        "a": a, "b": b, "V": V,
        "X_star": X_star,
        "inv_star": inv_star,
        "cmg_star": cmg_star,
        "cpl_nominal": cpl_nom,
        "cpl_incremental": cpl_incr,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
    }

def marginal_cost_curve(a, b, L_min, L_max, num=200):
    """
    Devuelve una curva de costo marginal d(inv)/dX = 2*(X-a)/b^2
    entre L_min y L_max.
    """
    import pandas as pd
    X = np.linspace(L_min, L_max, num=num)
    cmg = 2.0 * (X - a) / (b**2)
    cmg = np.where(X >= a, cmg, np.nan)  # antes de a no tiene sentido
    return pd.DataFrame({"leads": X, "cmg": cmg})

def required_investment_for_probability(target_leads: float, a: float, b: float,
                                        residuals: np.ndarray,
                                        p: float) -> float:
    """
    Calcula la inversión mínima necesaria para que P(Leads >= target) >= p,
    bajo L = a + b*sqrt(I) + ε.
    Método: empírico con cuantiles de residuales.
    """
    if not (0.5 < p < 0.999999):
        return np.nan
    if b is None or not np.isfinite(b) or b <= 0:
        return np.nan
    if a is None or not np.isfinite(a):
        return np.nan

    if residuals.size < 5:  # necesitamos al menos unos datos
        return np.nan

    # cuantil (1-p) de ε
    q = np.quantile(residuals, 1.0 - p)
    needed_mean = target_leads - q

    gap = max(needed_mean - a, 0.0)
    inv = (gap / b) ** 2
    return float(inv)

def investment_range_for_two_probabilities(target_leads: float, a: float, b: float,
                                           residuals: np.ndarray,
                                           p_low: float, p_high: float) -> Tuple[float, float]:
    """
    Devuelve (I_min_low, I_min_high) para niveles de seguridad p_low < p_high.
    """
    if p_low >= p_high:
        p_low, p_high = min(p_low, 0.95), min(max(p_low + 0.05, 0.55), 0.99)

    I_low  = required_investment_for_probability(target_leads, a, b, residuals, p_low)
    I_high = required_investment_for_probability(target_leads, a, b, residuals, p_high)

    I_low  = 0.0 if (not np.isfinite(I_low))  else max(I_low,  0.0)
    I_high = 0.0 if (not np.isfinite(I_high)) else max(I_high, 0.0)
    if I_low > I_high:
        I_low, I_high = I_high, I_low
    return I_low, I_high

def compute_residuals_for_product(df: pd.DataFrame, product_col: str, spend_col: str, leads_col: str,
                                  product: str, intercepto: float, coef: float) -> np.ndarray:
    """
    Calcula residuales = leads_observados - leads_modelo para un producto.
    Devuelve un array (puede ser vacío si no hay datos).
    """
    sub = df[df[product_col] == product].dropna(subset=[spend_col, leads_col]).copy()
    if sub.empty or not np.isfinite(intercepto) or not np.isfinite(coef):
        return np.array([])
    y_hat = intercepto + coef * np.sqrt(np.maximum(sub[spend_col].values.astype(float), 0.0))
    res = sub[leads_col].values.astype(float) - y_hat
    return res