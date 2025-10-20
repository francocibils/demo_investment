import numpy as np
import pandas as pd
import requests
import streamlit as st
from utils import (
    load_params, load_raw_data, detect_columns,
    predict_resultados, build_model_curve, investment_for_target_resultados,
    incremental_cost_one_more_resultado_at_target, marginal_cost_per_resultado_at_target,
    optimal_point_for_resultado_value, marginal_cost_curve,
    compute_residuals_for_product,
    required_investment_for_probability,
    investment_range_for_two_probabilities
)
import altair as alt

from optimizer import build_fixed_platforms, optimal_table_sqrt

st.set_page_config(page_title = "Investment optimizer", page_icon="📈", layout="wide")

# --- Sidebar navigation ---
st.sidebar.title("📚 Menú")
section = st.sidebar.radio(
    "Ir a:",
    [
        "Página principal",
        "Visualizador de datos",
        "Calculadora de resultados",
        "Presupuesto para objetivo de resultados",
        "Costo de un resultado extra",
        "Punto rentable del próximo resultado",
        "Rango de inversión para lograr X resultados",
        "Asignación óptima multi-plataforma"
    ]
)

# --- Carga fija de archivos en el ambiente ---
DATA_PATH = "https://raw.githubusercontent.com/francocibils/demo_investment/main/investment_data.csv"
PARAMS_PATH = "https://raw.githubusercontent.com/francocibils/demo_investment/main/params_dict.json"

# Carga archivos
df = load_raw_data(DATA_PATH)
#params = load_params(PARAMS_PATH)
params = requests.get(PARAMS_PATH).json()

if not params:
    st.error(f"No pude leer parámetros desde {PARAMS_PATH}. ¿Ruta/JSON válidos?")
    st.stop()

# Detección de columnas (ya estandarizadas)
col_date, col_product, col_spend, col_resultados = detect_columns(df)

# --- Página principal ---
if section == "Página principal":
    st.title("Investment optimizer - Herramientas")

    st.markdown(
        """
        La herramienta de Optimización de Inversión y resultados permite analizar y planificar la asignación de presupuesto de marketing de manera eficiente.
        
        A partir de modelos que capturan los rendimientos marginales decrecientes de la inversión, la aplicación ayuda a estimar cuántos resultados se pueden obtener, cuál es el costo marginal de cada resultados adicional y hasta qué punto es rentable seguir invirtiendo.
        
        Además, ofrece una sección de optimización multi-plataforma que distribuye automáticamente el presupuesto entre distintos canales según el valor asignado a cada resultado, maximizando el retorno esperado.
        
        **Secciones:**
        - **Visualizador de datos:** dispersión inversión–resultados y curva del modelo.
        - **Calculadora de resultados:** dado un monto de inversión, estima los resultados esperados.
        - **Presupuesto para objetivo de resultados:** dado un objetivo de resultados, estima cuánta inversión hace falta en promedio.
        - **Costo de un resultado extra:** dado un objetivo de resultados, estima cuál es el costo de un resultado extra a partir de ese objetivo.
        - **Punto rentable del próximo resultado:** dado un valor de un resultado, estima hasta qué punto es rentable seguir invirtiendo.
        - **Rango de inversión para lograr X resultados:** dado un objetivo de resultados, estima el rango de inversión que garantiza con cierta confianza que se logre el objetivo.
        - **Asignación óptima multi-plataforma:** dado un presupuesto y valor de resultado según plataforma, estima cómo debería la distribución de presupuesto según plataforma para maximizar la eficiencia.
        """
    )
    with st.expander("Vista previa de datos (normalizados)"):
        st.dataframe(df.head(20))

# --- Visualizador ---
elif section == "Visualizador de datos":
    st.title("Visualizador de datos")
    st.subheader("¿Cómo se relaciona la inversión con los resultados?")

    st.markdown(
    """
    **Qué hace:**  
    - Permite explorar la relación entre la inversión y los resultados generados a través de gráficos interactivos.  
    - Cada punto en la dispersión representa un **día de inversión y sus resultados correspondientes**, lo que permite observar la variabilidad diaria y cómo se comporta la relación inversión–resultado.  
    - El gráfico muestra tanto la **dispersión real de datos** (inversión–resultados) como la **curva estimada del modelo**, lo que facilita identificar patrones, validar la calidad del ajuste y entender cómo crecen los resultados a medida que aumenta la inversión.
    - Esta visualización ayuda a comparar el comportamiento observado con las proyecciones teóricas y sirve como base para tomar decisiones sobre escenarios futuros de inversión.

    """
    )
    
    products_in_data = sorted(df[col_product].dropna().unique().tolist())
    products_in_params = sorted(list(params.keys()))
    st.markdown(products_in_data)
    st.markdown(products_in_params)
    valid_products = [p for p in products_in_data if p in products_in_params]

    if not valid_products:
        st.warning("No hay intersección entre productos del dataset y del JSON. Verificá los nombres.")
    else:
        product = st.selectbox("Elegí un producto", valid_products, index=0)
        sub = df[df[col_product] == product].dropna(subset=[col_spend, col_resultados]).copy()

        intercepto, coef = params[product]
        curve_df = build_model_curve(sub[col_spend].values, intercepto, coef)

        scatter = alt.Chart(sub).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X(col_spend, title="Inversión"),
            y=alt.Y(col_resultados, title="resultados"),
            tooltip=[col_date, col_spend, col_resultados]
        ).properties(width=800, height=450)

        line = alt.Chart(curve_df).mark_line().encode(
            x=alt.X("inversion:Q", title="Inversión"),
            y=alt.Y("resultados_modelo:Q", title="Resultados (modelo)"),
            tooltip=["inversion", "resultados_modelo"]
        )

        st.altair_chart(scatter + line, use_container_width=True)

        y_true = sub[col_resultados].to_numpy(dtype=float)
        y_pred = predict_resultados(sub[col_spend].to_numpy(dtype=float), intercepto, coef)
        mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else np.nan

        st.markdown("**Métricas del ajuste sobre los puntos observados:**")
        c1, c3 = st.columns(2)
        c1.metric("Observaciones", len(sub))
        c3.metric("RMSE", f"{rmse:,.2f}")

        st.markdown("""
        **Interpretación del RMSE:**  
        El RMSE (Root Mean Squared Error) indica, en promedio, cuánto se desvían las predicciones del modelo respecto a los datos reales. Cuanto más bajo sea el RMSE, mejor es el ajuste del modelo a los datos observados.
        """)

# --- Calculadora ---
elif section == "Calculadora de resultados":

    st.title("Calculadora de resultados")
    st.subheader("Si invierto $, ¿cuántos resultados puedo obtener?")

    st.markdown(
    """
    **Qué hace:** Te permite estimar cuántos resultados podrías generar a partir de un monto de inversión en un producto o canal.  

    **Cómo funciona:**  
    - Ingresás la inversión que estás dispuesto a hacer.  
    - La herramienta calcula la cantidad esperada de resultados que podrías obtener con ese presupuesto.  
    - También muestra la **sensibilidad marginal**, es decir, cuántos resultados adicionales podrías conseguir si sumaras un poco más de inversión en ese mismo punto.  

    **Cómo leer los resultados:**  
    - El valor principal es la **estimación de resultados** para el monto ingresado.  
    - La **sensibilidad marginal** siempre va bajando a medida que subís la inversión, porque los rendimientos son decrecientes.  
    - Es un valor estimado promedio: en la práctica puede haber variaciones, pero sirve como referencia para la planificación.  

    **Para qué sirve:**  
    - Responder rápidamente “¿Cuántos resultados me da $X de inversión?”.  
    - Evaluar si conviene aumentar o reducir la inversión en un canal.  
    - Visualizar cómo cambia el rendimiento al mover el presupuesto.  
    """
    )

    # Productos disponibles (intersección entre datos y params)
    products_in_data = sorted(df[col_product].dropna().unique().tolist())
    products_in_params = sorted(list(params.keys()))
    valid_products = [p for p in products_in_data if p in products_in_params] or products_in_params

    product = st.selectbox("Elegí un producto", valid_products, index=0)
    intercepto, coef = params.get(product, (np.nan, np.nan))

    inversion = st.number_input(
        "Ingresá un monto de inversión",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help="El modelo utiliza la raíz cuadrada de la inversión."
    )

    # Botón de cálculo
    if st.button("Calcular"):
        # Predicción usando tu helper existente
        resultados_est = predict_resultados(np.array([inversion]), intercepto, coef)[0]

        # Sensibilidad marginal f'(x) = b / (2√x)
        if inversion > 0 and np.isfinite(coef):
            dldy = coef / (2.0 * np.sqrt(inversion))
            st.caption(f"Sensibilidad marginal (aprox): **{dldy:,.4f} resultados por dólar** adicional alrededor de ${inversion:,.0f}.")
        else:
            st.caption("Ingresá una inversión > 0 para ver la sensibilidad marginal.")

        # Filtrar datos históricos del producto
        df_prod = df[df[col_product] == product].copy()
        scatter_df = df_prod[[col_spend, col_resultados]].rename(columns={
            col_spend: "Inversión",
            col_resultados: "resultados"
        })

        # Curva del modelo en un grid
        max_inv_hist = float(scatter_df["Inversión"].max()) if len(scatter_df) else 0.0
        x_max = max(max_inv_hist, inversion) * 1.15 + 1e-9  # margen
        grid = np.linspace(0.0, x_max, 200)
        curve_df = pd.DataFrame({
            "Inversión": grid,
            "resultados_modelo": intercepto + coef * np.sqrt(grid)
        })

        # Punto de la calculadora
        point_df = pd.DataFrame({"Inversión": [inversion], "resultados": [resultados_est]})

        # KPIs arriba del gráfico
        c1, c2, c3 = st.columns(3)
        c1.metric("Inversión", f"${inversion:,.0f}")
        c2.metric("resultados estimados", f"{resultados_est:,.0f}")
        marg = (coef / (2*np.sqrt(inversion))) if inversion > 0 else float("inf")
        c3.metric("Sensibilidad marginal", "∞" if not np.isfinite(marg) else f"{marg:,.3f}")

        # --- Gráficos Altair (sin usar .data(...) para evitar el UndefinedType callable) ---
        scatter = alt.Chart(scatter_df).mark_circle(size=60, opacity=0.35).encode(
            x=alt.X("Inversión:Q", title="Inversión", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("resultados:Q", title="resultados", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("Inversión:Q", format="$,.0f"),
                     alt.Tooltip("resultados:Q", format=",.0f")]
        )

        line = alt.Chart(curve_df).mark_line(strokeWidth=3).encode(
            x=alt.X("Inversión:Q", title="Inversión"),
            y=alt.Y("resultados_modelo:Q", title="resultados"),
            tooltip=[alt.Tooltip("Inversión:Q", format="$,.0f"),
                    alt.Tooltip("resultados_modelo:Q", title="resultados (modelo)", format=",.0f")]
        )

        point = alt.Chart(point_df).mark_point(size=500, filled=True, shape="diamond", color="#FF4B4B").encode(
            x=alt.X("Inversión:Q"),
            y=alt.Y("resultados:Q"),
            tooltip=[alt.Tooltip("Inversión:Q", format="$,.0f"),
                     alt.Tooltip("resultados:Q", format=",.0f")]
        )

        rule_x = alt.Chart(point_df).mark_rule(strokeDash=[4,4], opacity=1, color="#FF4B4B").encode(x="Inversión:Q")
        rule_y = alt.Chart(point_df).mark_rule(strokeDash=[4,4], opacity=1, color="#FF4B4B").encode(y="resultados:Q")

        chart = (scatter + line + point + rule_x + rule_y).properties(
            height=420,
            title=f"Dispersión, curva del modelo y punto estimado — {product}"
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "La línea muestra la **curva del modelo**; el **rombo** indica la predicción en tu inversión; "
            "cada **punto** del scatter es un día observado."
        )

elif section == "Presupuesto para objetivo de resultados":
    st.title("Presupuesto para objetivo de resultados")
    st.subheader("Dado que quiero conseguir X resultados, ¿cuánto me costarán aproximadamente estos resultados?")

    st.markdown(
    """
    **Qué hace:** Te dice cuánta inversión mínima necesitás para alcanzar un objetivo de `X` resultados.  

    **Cómo funciona:**  
    - Ingresás la cantidad de resultados que querés conseguir.  
    - La herramienta estima el presupuesto requerido para llegar a esa meta.  
    - También muestra el **CPL promedio** (costo por resultado) que resultaría en ese escenario.  

    **Cómo leer los resultados:**  
    - Si tu objetivo de resultados es muy bajo, puede que el modelo indique que **no hace falta inversión** porque ya se alcanzan con la base actual.  
    - El **CPL promedio** te sirve para comparar qué tan eficiente es el presupuesto estimado frente a otros escenarios.  
    - Si aparece una advertencia de **extrapolación**, significa que el presupuesto necesario está fuera del rango histórico de datos, por lo que el cálculo debe tomarse con cautela.  

    **Para qué sirve:**  
    - Planear cuánta inversión se necesita para lograr una meta específica de resultados.  
    - Comparar la eficiencia de distintos objetivos en términos de CPL.  
    - Evitar invertir de más o quedarse corto en la planificación de campañas.  
    """
    )
    # Productos disponibles (preferentemente la intersección data/params)
    products_in_data = sorted(df[col_product].dropna().unique().tolist())
    products_in_params = sorted(list(params.keys()))
    valid_products = [p for p in products_in_data if p in products_in_params] or products_in_params

    product = st.selectbox("Elegí un producto", valid_products, index=0)
    intercepto, coef = params.get(product, (np.nan, np.nan))

    target_resultados = st.number_input(
        "Ingresá el objetivo de resultados (X)",
        min_value=0.0,
        value=100.0,
        step=50.0,
        help="Cantidad de resultados que querés alcanzar."
    )

    if st.button("Calcular presupuesto"):
        inv_req = investment_for_target_resultados(target_resultados, intercepto, coef)

        # Rango histórico para advertir extrapolaciones
        sub = df[df[col_product] == product]
        hist_max = float(sub[col_spend].max()) if not sub.empty else np.nan
        hist_min = float(sub[col_spend].min()) if not sub.empty else np.nan

        if not np.isfinite(inv_req):
            st.error("No se pudo calcular el presupuesto: el coeficiente del modelo no es válido (b ≤ 0) o hay parámetros faltantes.")
        else:
            # Mensaje principal
            if inv_req == 0.0 and target_resultados <= intercepto:
                st.success(
                    f"Según el modelo, para **{product}** no hace falta inversión para llegar a **{target_resultados:,.1f}** resultados "
                    f"(intercepto = {intercepto:,.1f})."
                )
            else:

                # -------- MÉTRICAS (c1, c2, c3) --------
                c1, c2 = st.columns(2)

                # c1: Presupuesto necesario
                c1.metric("Presupuesto necesario", f"${inv_req:,.2f}")

                # c2: CPL nominal = inv / target
                cpl_nominal = (inv_req / target_resultados) if target_resultados > 0 else np.nan
                c2.metric("CPL nominal", f"${cpl_nominal:,.2f}" if np.isfinite(cpl_nominal) else "—")

                # c3: resultados esperados (modelo) con la inversión calculada (≈ target)
                resultados_check = predict_resultados(np.array([inv_req]), intercepto, coef)[0]
                # c3.metric("resultados esperados (modelo)", f"{resultados_check:,.1f}")
                # ---------------------------------------

            # Advertencias útiles
            if np.isfinite(hist_max) and inv_req > hist_max:
                st.warning(
                    f"⚠️ El presupuesto requerido (\${inv_req:,.2f}) excede el máximo histórico observado para {product} (${hist_max:,.2f}). Estás **extrapolando** fuera del rango de datos. Interpretá los resultados con cautela."
                )
            if np.isfinite(hist_min) and inv_req < hist_min:
                st.caption(
                    f"Nota: el presupuesto requerido (${inv_req:,.2f}) es inferior al mínimo histórico (${hist_min:,.2f})."
                )

            # Datos del producto
            df_prod = df[df[col_product] == product].copy()
            scatter_df = df_prod[[col_spend, col_resultados]].rename(columns={
                col_spend: "Inversión",
                col_resultados: "resultados"
            })

            # Curva del modelo (línea más gruesa)
            max_inv_hist = float(scatter_df["Inversión"].max()) if len(scatter_df) else 0.0
            x_max = max(max_inv_hist, inv_req if np.isfinite(inv_req) else 0.0) * 1.15 + 1e-9
            grid = np.linspace(0.0, x_max, 200)
            curve_df = pd.DataFrame({
                "Inversión": grid,
                "resultados_modelo": intercepto + coef * np.sqrt(grid)
            })

            # Punto en el presupuesto requerido y su predicción
            point_df = pd.DataFrame({"Inversión": [inv_req], "resultados": [predict_resultados(np.array([inv_req]), intercepto, coef)[0]]})

            # Construcción de los layers
            scatter = alt.Chart(scatter_df).mark_circle(size=60, opacity=0.55).encode(
                x=alt.X("Inversión:Q", title="Inversión", axis=alt.Axis(format="$,.0f")),
                y=alt.Y("resultados:Q", title="resultados", axis=alt.Axis(format=",.0f")),
                tooltip=[alt.Tooltip("Inversión:Q", format="$,.0f"),
                         alt.Tooltip("resultados:Q", format=",.0f")]
            )

            line = alt.Chart(curve_df).mark_line(strokeWidth=3).encode(
                x=alt.X("Inversión:Q", title="Inversión"),
                y=alt.Y("resultados_modelo:Q", title="resultados"),
                tooltip=[alt.Tooltip("Inversión:Q", format="$,.0f"),
                         alt.Tooltip("resultados_modelo:Q", title="resultados (modelo)", format=",.0f")]
            )

            point = alt.Chart(point_df).mark_point(size=500, filled=True, shape="diamond", color="#FF4B4B").encode(
                x="Inversión:Q",
                y="resultados:Q",
                tooltip=[alt.Tooltip("Inversión:Q", format="$,.0f"),
                         alt.Tooltip("resultados:Q", format=",.0f")]
            )

            rule_x = alt.Chart(point_df).mark_rule(strokeDash=[4,4], opacity=1, color="#FF4B4B").encode(x="Inversión:Q")
            rule_y = alt.Chart(point_df).mark_rule(strokeDash=[4,4], opacity=1, color="#FF4B4B").encode(y="resultados:Q")

            chart = (scatter + line + point + rule_x + rule_y).properties(
                height=420,
                title=f"Dispersión, curva del modelo y punto requerido — {product}"
            )

            st.altair_chart(chart, use_container_width=True)

            st.caption(
                "La línea muestra la **curva del modelo**; el **rombo** marca el **presupuesto requerido** para alcanzar X; "
                "cada **punto** del scatter es un día observado."
            )

elif section == "Costo de un resultado extra":

    st.title("Costo de un resultado extra")
    st.subheader("Dado que ya estoy consiguiendo X resultados, ¿cuánto costará conseguir un resultado extra?")

    st.markdown(
    """
    **Qué hace:** Muestra cuánto te costaría conseguir **un resultado adicional** si ya estás apuntando a un objetivo de `X` resultados.  

    **Cómo funciona:**  
    - Definís un objetivo de resultados.  
    - La herramienta calcula cuánto presupuesto extra necesitarías para pasar de `X` a `X+1` resultados.  
    - Ese valor es el **costo del próximo resultado**.  

    **Cómo leer los resultados:**  
    - Si el costo del resultado extra es **menor que el valor que aporta un resultado**, conviene invertir un poco más.  
    - Si es **mayor**, significa que ese siguiente resultado ya no es rentable.  
    - Si aparece una advertencia de **extrapolación**, quiere decir que el cálculo está fuera del rango de inversión observado históricamente, por lo que debe tomarse con cautela.  

    **Para qué sirve:**  
    - Decidir si conviene **empujar la inversión un poco más** para conseguir algunos resultados extra.  
    - Identificar el punto en que sumar un resultado adicional deja de ser eficiente.  
    - Complementar la planificación de presupuestos con una visión detallada de los costos marginales.  
    """
    )

    # Productos (ideal: intersección entre data y params)
    products_in_data = sorted(df[col_product].dropna().unique().tolist())
    products_in_params = sorted(list(params.keys()))
    valid_products = [p for p in products_in_data if p in products_in_params] or products_in_params

    product = st.selectbox("Elegí un producto", valid_products, index=0)
    intercepto, coef = params.get(product, (np.nan, np.nan))

    target_resultados = st.number_input(
        "resultados que ya estoy buscando conseguir (X)",
        min_value=0.0, value=100.0, step=50.0
    )

    if st.button("Calcular costo de 1 resultado extra"):
        # Costo DISCRETO: inv(X+1) - inv(X)
        cost_discrete = incremental_cost_one_more_resultado_at_target(target_resultados, intercepto, coef)
        # Costo MARGINAL (derivada) en X
        cost_marginal = marginal_cost_per_resultado_at_target(target_resultados, intercepto, coef)

        if not (np.isfinite(cost_discrete) or np.isfinite(cost_marginal)):
            st.error("No se pudo calcular el costo marginal. Verificá que el coeficiente (b) sea > 0 y que X > a, o usá otro X.")
        else:
            c1, c2 = st.columns(2)
            inv_x = investment_for_target_resultados(target_resultados, intercepto, coef)
            c1.metric(f"Inversión para {target_resultados} resultados", f"${inv_x:,.2f}" if np.isfinite(inv_x) else "—")
            c2.metric("Costo DISCRETO de +1 resultado", f"${cost_discrete:,.2f}" if np.isfinite(cost_discrete) else "—")

            st.caption(
                "• **DISCRETO**: diferencia exacta de presupuesto entre apuntar a X y a X+1 resultados.\n"
            )

        # Contexto: rango histórico de inversión observado para este producto
        sub = df[df[col_product] == product]
        if not sub.empty:
            hist_min, hist_max = float(sub[col_spend].min()), float(sub[col_spend].max())
            with st.expander("Rango histórico de inversión (observado en datos)"):
                st.write({"mínimo": hist_min, "máximo": hist_max})
            if np.isfinite(inv_x) and np.isfinite(hist_max) and inv_x > hist_max:
                st.warning(
                    f"⚠️ Estás evaluando el costo en un objetivo {target_resultados} resultados cuya inversión base (\${inv_x:,.2f}) ya excede el máximo histórico (${hist_max:,.2f}). Estás extrapolando por lo que los resultados deben ser tomados con cautela."
                )

        # ================== GRÁFICO: CPL promedio con punto + líneas ==================
        try:
            def inv_from_resultados(L, a, b):
                L = np.asarray(L, dtype=float)
                if not np.isfinite(b) or b <= 0:
                    return np.full_like(L, np.nan, dtype=float)
                inv = np.where(L > a, ((L - a) / b) ** 2, 0.0)
                return inv

            start_L = max(1.0, float(intercepto) + 1.0) if np.isfinite(intercepto) else 1.0
            end_L   = max(start_L + 1.0, (target_resultados * 1.5) if np.isfinite(target_resultados) else start_L + 50.0)
            resultados_range = np.linspace(start_L, end_L, 220)

            inv_range = inv_from_resultados(resultados_range, intercepto, coef)
            cpl_curve = pd.DataFrame({
                "resultados": resultados_range,
                "CPL": inv_range / resultados_range
            }).replace([np.inf, -np.inf], np.nan).dropna()

            # Punto en X
            cpl_at_x = (inv_x / target_resultados) if (np.isfinite(inv_x) and target_resultados > 0) else np.nan
            point_cpl = pd.DataFrame({"resultados": [target_resultados], "CPL": [cpl_at_x]})

            cpl_chart = alt.Chart(cpl_curve).mark_line(strokeWidth=3).encode(
                x=alt.X("resultados:Q", title="resultados"),
                y=alt.Y("CPL:Q", title="CPL promedio (inversión / resultados)")
            )

            dot_cpl = alt.Chart(point_cpl).mark_point(
                size=140, filled=True, shape="diamond", color="#FF4B4B"  # rojo brillante
            ).encode(
                x="resultados:Q",
                y="CPL:Q",
                tooltip=[alt.Tooltip("resultados:Q", format=",.0f"),
                         alt.Tooltip("CPL:Q", format="$,.2f")]
            )

            # Línea vertical en X
            line_x = alt.Chart(point_cpl).mark_rule(strokeDash=[4,4], color="#FF4B4B").encode(x="resultados:Q")

            # Línea horizontal en CPL(X)
            line_y = alt.Chart(point_cpl).mark_rule(strokeDash=[4,4], color="#FF4B4B").encode(y="CPL:Q")

            st.subheader("CPL promedio vs. resultados (con punto en X)")
            st.altair_chart(cpl_chart + dot_cpl + line_x + line_y, use_container_width=True)
            st.caption("El **rombo** marca el CPL en tu objetivo X; las líneas punteadas indican el CPL y los resultados correspondientes.")

            st.caption(
                """
                **Cómo interpretar el gráfico de CPL:**  
                - La **curva** muestra cómo evoluciona el costo promedio por resultado (CPL) a medida que aumenta el objetivo de resultados.  
                - El **rombo rojo** marca tu objetivo `X` y el CPL asociado.  
                - Si el CPL en `X` es menor que el valor que le asignás a cada resultado, todavía conviene invertir.  
                - Si el CPL en `X` supera ese valor, significa que llegar a esa meta deja de ser rentable.
                """
            )
        except Exception:
            pass

elif section == "Punto rentable del próximo resultado":
    st.title("Punto rentable del próximo resultado")
    st.subheader("Dado que cada resultado cuesta más que el anterior, ¿hasta que punto es rentable conseguir un resultado extra?")

    st.markdown(
    """
    **Qué hace:** Ayuda a identificar hasta qué punto conviene seguir invirtiendo en un producto o canal, comparando el costo de conseguir un resultado adicional con el valor que ese resultado genera.  

    **Cómo funciona:**  
    - Ingresás el **valor promedio de un resultado** (lo que aporta en ingresos).  
    - La herramienta calcula el **punto óptimo**, es decir, el nivel de resultados y de inversión donde el siguiente resultado deja de ser rentable porque cuesta lo mismo o más de lo que vale.  

    **Cómo leer los resultados:**  
    - Si estás **por debajo del punto óptimo**, todavía conviene invertir (cada resultado extra aporta más de lo que cuesta).  
    - Si estás **por encima**, seguir invirtiendo deja de ser eficiente (los próximos resultados cuestan más de lo que valen).  
    - También se muestra el **profit estimado** en ese punto: ingresos menos inversión.  
    - Si aparece la advertencia de **extrapolación**, significa que el óptimo calculado queda fuera del rango de inversión histórica observada, por lo que hay que tomarlo con cautela.  

    **Para qué sirve:**  
    - Saber hasta dónde conviene invertir en una campaña antes de que deje de ser rentable.  
    - Comparar diferentes productos o canales según su punto óptimo.  
    - Definir presupuestos basados en el valor real que aporta cada resultado.  
    """
    )

    # Productos (ideal: intersección datos/params)
    products_in_data = sorted(df[col_product].dropna().unique().tolist())
    products_in_params = sorted(list(params.keys()))
    valid_products = [p for p in products_in_data if p in products_in_params] or products_in_params

    product = st.selectbox("Elegí un producto", valid_products, index=0)
    intercepto, coef = params.get(product, (np.nan, np.nan))

    resultado_value = st.number_input(
        "Valor por resultado (V) — ingreso por cada resultado",
        min_value=0.0, value=10.0, step=1.0, help="Usá el ingreso promedio por resultado o LTV ponderado."
    )

    if st.button("Calcular punto rentable"):
        res = optimal_point_for_resultado_value(resultado_value, intercepto, coef)
        if not res["ok"]:
            st.error(res["msg"])
        else:
            X_star   = res["X_star"]
            inv_star = res["inv_star"]
            cpl_nom  = res["cpl_nominal"]
            cpl_incr = res["cpl_incremental"]
            revenue  = res["revenue"]
            cost     = res["cost"]
            profit   = res["profit"]

            c1, c2 = st.columns(2)
            c1.metric("resultados óptimos (X*)", f"{X_star:,.1f}")
            c2.metric("Inversión óptima", f"${inv_star:,.2f}")
            #c3.metric("Costo marginal en X* (dInv/dX)", f"${resultado_value:,.2f}")

            c4, c6 = st.columns(2)
            c4.metric("CPL promedio en X*", f"${cpl_nom:,.2f}" if np.isfinite(cpl_nom) else "—")
            #c5.metric("CPL incremental en X*", f"${cpl_incr:,.2f}" if np.isfinite(cpl_incr) else "—")
            c6.metric("Profit en X*", f"${profit:,.2f}")

            # Advertencias de rango histórico (extrapolación)
            sub = df[df[col_product] == product]
            if not sub.empty:
                hist_min_inv, hist_max_inv = float(sub[col_spend].min()), float(sub[col_spend].max())
                if np.isfinite(hist_max_inv) and inv_star > hist_max_inv:
                    st.warning(
                        f"⚠️ El óptimo sugiere invertir ${inv_star:,.2f}, que excede el máximo histórico observado "
                        f"(${hist_max_inv:,.2f}). **Extrapolación**."
                    )

            # Gráfico opcional: costo marginal vs resultados y línea V
            try:
                L_min = float(max(sub[col_resultados].min(), res["a"])) if not sub.empty else res["a"]
                L_max = float(max(sub[col_resultados].max(), X_star)) if not sub.empty else X_star
                curve = marginal_cost_curve(res["a"], res["b"], L_min, L_max)
                rule  = pd.DataFrame({"resultados": [L_min, L_max], "V": [resultado_value, resultado_value]})
                point = pd.DataFrame({"resultados": [X_star], "cmg": [resultado_value]})

                cmg_chart = alt.Chart(curve).mark_line().encode(
                    x=alt.X("resultados:Q", title="resultados"),
                    y=alt.Y("cmg:Q", title="Costo marginal")
                )
                vline = alt.Chart(rule).mark_rule().encode(y="V:Q")
                dot   = alt.Chart(point).mark_point(size=80).encode(x="resultados:Q", y="cmg:Q")

                st.altair_chart(cmg_chart + vline + dot, use_container_width=True)
            except Exception:
                pass

elif section == "Rango de inversión para lograr X resultados":
    st.title("Rango de inversión para obtener X resultados")
    st.subheader("Dado que quiero X resultados, ¿cuál es el rango de inversión que me garantiza con cierta seguridad que los vaya a conseguir?")

    st.markdown(
        """
    **Qué hace:** Te ayuda a estimar cuánto presupuesto necesitás para alcanzar un objetivo de resultados con distintos **niveles de seguridad**.  

    **Cómo funciona:**  
    - Definís un objetivo de resultados `X`.  
    - Elegís dos niveles de confianza: uno más **arriesgado** y otro más **conservador**.  
    - La herramienta calcula un **rango de inversión**:  
    - El valor más bajo es lo que deberías invertir si estás dispuesto a correr más riesgo.  
    - El valor más alto es lo que asegura con mayor probabilidad que llegues al objetivo.  

    **Cómo leer los resultados:**  
    - El rango te da una idea de la **zona de inversión razonable** para alcanzar tu meta.  
    - La parte baja sirve para escenarios optimistas; la parte alta, para escenarios más seguros.  
    - En el gráfico, la **banda sombreada** muestra ese rango de inversión y las **líneas** indican los niveles de confianza elegidos.  

    **Para qué sirve:**  
    - Planear presupuestos con diferentes grados de seguridad.  
    - Entender cuánto más tenés que invertir si querés estar más seguro de alcanzar tus objetivos.  
    - Comparar escenarios de riesgo vs. seguridad al definir un presupuesto de campaña.  
    """
    )

    products_in_data = sorted(df[col_product].dropna().unique().tolist())
    products_in_params = sorted(list(params.keys()))
    valid_products = [p for p in products_in_data if p in products_in_params] or products_in_params

    product = st.selectbox("Elegí un producto", valid_products, index=0)
    intercepto, coef = params.get(product, (np.nan, np.nan))

    target_resultados = st.number_input("Objetivo de resultados (X)", min_value=0.0, value=500.0, step=50.0)

    c1, c2 = st.columns(2)
    with c1:
        p_low = st.slider("Seguridad mínima (p_low)", min_value=0.60, max_value=0.99,
                          value=0.80, step=0.01)
    with c2:
        p_high = st.slider("Seguridad alta (p_high)", min_value=0.60, max_value=0.99,
                           value=0.95, step=0.01)

    if st.button("Calcular rango de inversión"):
        residuals = compute_residuals_for_product(
            df, col_product, col_spend, col_resultados, product, intercepto, coef
        )

        I_low, I_high = investment_range_for_two_probabilities(
            target_resultados, intercepto, coef, residuals, p_low, p_high
        )

        # métricas
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Inversión mínima (p ≥ {p_low:.0%})", f"${I_low:,.2f}")
        c2.metric(f"Inversión (p ≥ {p_high:.0%})", f"${I_high:,.2f}")
        I_mid = 0.5 * (I_low + I_high)
        c3.metric("Sugerencia (punto medio)", f"${I_mid:,.2f}")

        # check extrapolación
        sub = df[df[col_product] == product]
        if not sub.empty and np.isfinite(sub[col_spend].max()):
            hist_max = float(sub[col_spend].max())
            if I_high > hist_max:
                st.warning(
                    f"⚠️ El rango calculado (${I_low:,.2f} → ${I_high:,.2f}) excede "
                    f"el máximo histórico observado (${hist_max:,.2f}). Estás extrapolando."
                )

        # gráfico empírico: P(resultados ≥ X) vs inversión
        try:
            grid = np.linspace(0, max(I_high * 1.5, 1.0), 200)
            mu = intercepto + coef * np.sqrt(np.maximum(grid, 0.0))

            # Probabilidad empírica: P(ε ≥ target - mu)
            vals = target_resultados - mu
            probs = []
            for v in vals:
                # Fε(v) = P(ε ≤ v)
                F = (residuals <= v).mean() if residuals.size > 0 else np.nan
                probs.append(1.0 - F)
            probs = np.array(probs)

            plot_df = pd.DataFrame({"inversion": grid, "prob": probs})

            line = alt.Chart(plot_df).mark_line().encode(
                x=alt.X("inversion:Q", title="Inversión"),
                y=alt.Y("prob:Q", title=f"P(resultados ≥ {target_resultados:,.0f})", scale=alt.Scale(domain=[0,1]))
            )

            band = pd.DataFrame({
                "inversion": [I_low, I_low, I_high, I_high],
                "prob": [0, 1, 1, 0]
            })
            box = alt.Chart(band).mark_area(opacity=0.1, color="green").encode(
                x="inversion:Q", y="prob:Q"
            )

            rules = alt.Chart(pd.DataFrame({"p": [p_low, p_high]})).mark_rule(
                strokeDash=[4,4], color="red"
            ).encode(y="p:Q")

            st.altair_chart(line + box + rules, use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo graficar: {e}")

elif section == "Asignación óptima multi-plataforma":

    st.title("Asignación óptima multi-plataforma")

    st.markdown(
    """
    
    **Qué hace:**  Ayuda a decidir cómo distribuir un **presupuesto total** entre distintas plataformas de medios (ejemplo: Google Ads, Instagram, TikTok, Facebook Ads, Twitter) para obtener la mejor combinación de resultados.  

    **Cómo funciona:**  
    - Ingresás cuánto vale para vos un resultado en cada plataforma.  
    - Definís el presupuesto total disponible.  
    - La herramienta calcula cómo conviene repartir ese presupuesto para que cada peso invertido rinda lo máximo posible según el valor de cada resultado.  

    **Qué muestran los resultados:**  
    - Una visión general con el **presupuesto total** y la **cantidad de resultados esperados**.  
    - El detalle por plataforma, que podés ver en **cantidades absolutas** o en **porcentajes**:  
    - **resultados** generados por cada plataforma.  
    - **Inversión** asignada a cada una.  

    **Cómo interpretar:**  
    - Si una plataforma recibe más inversión que otra, significa que allí el sistema encuentra una mejor relación entre el valor que le diste a sus resultados y el costo de conseguirlos.  
    - Comparar los shares de **spend** y **resultados** ayuda a ver dónde se está invirtiendo más y qué tan eficiente es esa inversión.  

    **Para qué sirve:**  
    - Planear asignaciones de presupuesto en campañas multicanal.  
    - Probar escenarios cambiando el valor de los resultados según calidad o retorno esperado.  
    - Visualizar cómo impacta un cambio en el presupuesto total sobre la distribución entre plataformas.  
    """
    )

    if not params:
        st.error("No hay parámetros de productos en params_dict.json")
        st.stop()

    # Productos disponibles
    products = sorted(list(params.keys()))
    product = st.selectbox("Producto", options = products, index = 0)

    # Sets fijos por producto
    fixed_sets = {
        "Product 1": ["Google Ads", "Instagram", "TikTok", "Facebook Ads"],
        "Product 2": ["Google Ads", "Facebook Ads", "Twitter"],
    }
    platforms = fixed_sets.get(product, [])

    if not platforms:
        st.warning("No hay plataformas configuradas para este producto.")
        st.stop()

    # Inputs: valor por resultado por plataforma (editable por el usuario)
    st.subheader("Valor por resultado por plataforma")
    v_values = []
    cols = st.columns(len(platforms))
    for i, name in enumerate(platforms):
        with cols[i]:
            v = st.number_input(
                label   = name,
                min_value = 0.01,
                value   = 10.00,
                step    = 1.,
                format  = "%.2f",
                key     = f"v_{product}_{name}"
            )
            v_values.append(v)

    # Presupuesto
    budget = st.number_input(
        "Presupuesto total a asignar",
        min_value = 0.0,
        value = 1000.0,
        step = 100.0
    )

    # Parámetros del producto (a, b)
    a_prod, b_prod = params[product]

    # Construcción de plataformas y optimización (modelo raíz)
    plat_df = build_fixed_platforms(
        product_name   = product,
        a_prod         = a_prod,
        b_prod         = b_prod,
        platform_names = platforms,
        v_values       = v_values
    )
    alloc = optimal_table_sqrt(plat_df, budget)

    # -----------------------------
    # KPIs superiores (c1, c2, c3)
    # -----------------------------
    c1, c2= st.columns(2)
    c1.metric("Presupuesto total", f"${budget:,.0f}")
    c2.metric("resultados totales obtenidos", f"{alloc['resultados'].sum():,.0f}")

    # -----------------------------
    # Modo de visualización
    # -----------------------------
    st.subheader("Vista por plataforma")
    view_mode = st.radio("Mostrar como:", ["Cantidades", "Porcentajes"], horizontal=True)

    # Prepara shares
    total_resultados = float(alloc["resultados"].sum())
    total_spend = float(alloc["spend"].sum())
    alloc = alloc.copy()
    alloc["resultados_share"] = alloc["resultados"] / total_resultados if total_resultados > 0 else 0.0
    alloc["spend_share"] = alloc["spend"] / total_spend if total_spend > 0 else 0.0

    # -----------------------------
    # Métricas por plataforma: resultados
    # -----------------------------
    st.markdown("**resultados por plataforma**")
    cols_resultados = st.columns(len(platforms))
    for i, (name, resultados, lshare) in enumerate(zip(alloc["platform"], alloc["resultados"], alloc["resultados_share"])):
        with cols_resultados[i]:
            if view_mode == "Cantidades":
                st.metric(label=name, value=f"{resultados:,.0f}")
            else:
                st.metric(label=name, value=f"{lshare:,.1%}")

    # -----------------------------
    # Métricas por plataforma: Spend
    # -----------------------------
    st.markdown("**Spend por plataforma**")
    cols_spend = st.columns(len(platforms))
    for i, (name, spend, sshare) in enumerate(zip(alloc["platform"], alloc["spend"], alloc["spend_share"])):
        with cols_spend[i]:
            if view_mode == "Cantidades":
                st.metric(label=name, value=f"${spend:,.0f}")
            else:
                st.metric(label=name, value=f"{sshare:,.1%}")
