"""DataNet Analytics - Optimización multiflujo de costo mínimo (PuLP + Streamlit)."""
import pandas as pd
import pulp
import streamlit as st



TOL = 1e-6

# ----------------------------------------------------------------- DATOS
EDGES0 = pd.DataFrame(
    [(1, 2, 4, 12), (1, 3, 6, 10), (2, 3, 2, 8), (2, 4, 5, 10), (2, 5, 7, 8),
     (3, 4, 3, 12), (3, 6, 5, 10), (4, 5, 2, 10), (4, 7, 6, 12), (5, 6, 3, 8),
     (5, 8, 5, 10), (6, 7, 2, 10), (6, 8, 4, 12), (7, 8, 3, 14)],
    columns=["Origen", "Destino", "Costo", "Capacidad"])
EDGES0["Activo"] = True
FLOWS0 = pd.DataFrame(
    [("F1", 1, 8, 6.0), ("F2", 2, 7, 5.0), ("F3", 3, 8, 4.0)],
    columns=["Flujo", "Origen", "Destino", "Demanda"])
NODES = list(range(1, 9))
POS = {1: (0, 1), 2: (1, 2), 3: (1, 0), 4: (2, 2), 5: (2, 1),
       6: (3, 0), 7: (3.2, 2), 8: (4, 1)}


# ---------------------------------------------------------------- MODELO
def solve(edges: pd.DataFrame, flows: pd.DataFrame):
    """Construye y resuelve el multiflujo. Devuelve dict con estado y resultados."""
    act = edges[edges["Activo"]]
    A = [(int(r.Origen), int(r.Destino)) for r in act.itertuples()]      # enlaces existentes y activos
    cost = {(int(r.Origen), int(r.Destino)): float(r.Costo) for r in act.itertuples()}
    cap = {(int(r.Origen), int(r.Destino)): float(r.Capacidad) for r in act.itertuples()}
    K = list(flows["Flujo"])
    s = dict(zip(flows["Flujo"], flows["Origen"].astype(int)))
    t = dict(zip(flows["Flujo"], flows["Destino"].astype(int)))
    d = dict(zip(flows["Flujo"], flows["Demanda"].astype(float)))

    prob = pulp.LpProblem("Multiflujo", pulp.LpMinimize)
    # Variables x[k,i,j] >= 0 (no negatividad)
    x = {(k, i, j): pulp.LpVariable(f"x_{k}_{i}_{j}", lowBound=0) for k in K for (i, j) in A}
    # Objetivo
    prob += pulp.lpSum(cost[i, j] * x[k, i, j] for k in K for (i, j) in A)
    # Conservación de flujo por comunicación y nodo (+ satisfacción de demanda)
    for k in K:
        for n in NODES:
            out = pulp.lpSum(x[k, i, j] for (i, j) in A if i == n)
            inn = pulp.lpSum(x[k, i, j] for (i, j) in A if j == n)
            rhs = d[k] if n == s[k] else (-d[k] if n == t[k] else 0)
            prob += out - inn == rhs, f"cons_{k}_{n}"
    # Capacidad compartida
    for (i, j) in A:
        prob += pulp.lpSum(x[k, i, j] for k in K) <= cap[i, j], f"cap_{i}_{j}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    res = {"status": status, "A": A, "K": K, "cost": cost, "cap": cap,
           "s": s, "t": t, "d": d, "obj": None, "x": None}
    if status == "Optimal":
        res["x"] = {key: (v.value() or 0.0) for key, v in x.items()}
        res["obj"] = pulp.value(prob.objective)
    return res


def validate(res):
    """Verifica restricciones y recalcula la función objetivo."""
    x, A, K = res["x"], res["A"], res["K"]
    ok = all(v >= -TOL for v in x.values())
    for k in K:
        for n in NODES:
            net = sum(x[k, i, j] for (i, j) in A if i == n) - sum(x[k, i, j] for (i, j) in A if j == n)
            rhs = res["d"][k] if n == res["s"][k] else (-res["d"][k] if n == res["t"][k] else 0)
            ok &= abs(net - rhs) <= TOL
    for (i, j) in A:
        ok &= sum(x[k, i, j] for k in K) <= res["cap"][i, j] + TOL
    recomputed = sum(res["cost"][i, j] * x[k, i, j] for k in K for (i, j) in A)
    return bool(ok), abs(recomputed - res["obj"]) <= 1e-5


def link_table(res):
    rows = []
    for (i, j) in res["A"]:
        row = {"Enlace": f"{i}→{j}"}
        tot = 0.0
        for k in res["K"]:
            row[k] = round(res["x"][k, i, j], 4)
            tot += res["x"][k, i, j]
        row["Total"] = round(tot, 4)
        row["Capacidad"] = res["cap"][i, j]
        row["Utilización %"] = round(100 * tot / res["cap"][i, j], 1) if res["cap"][i, j] > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


import plotly.graph_objects as go

C_OK, C_SAT, C_AV, C_OFF = "#16a34a", "#dc2626", "#94a3b8", "#e2e8f0"


def network_fig(edges, res):
    used = {}
    if res and res["x"]:
        for (i, j) in res["A"]:
            used[i, j] = {k: res["x"][k, i, j] for k in res["K"]}
    fig = go.Figure()
    mx, my, mt = [], [], []
    for r in edges.itertuples():
        e = (int(r.Origen), int(r.Destino))
        (x0, y0), (x1, y1) = POS[e[0]], POS[e[1]]
        tot = sum(used[e].values()) if e in used else 0
        if not r.Activo:
            col, w, state = C_OFF, 1, "desactivado"
        elif tot > TOL and tot >= r.Capacidad - TOL:
            col, w, state = C_SAT, 4, "saturado"
        elif tot > TOL:
            col, w, state = C_OK, 2 + 4 * tot / max(r.Capacidad, 1), "utilizado"
        else:
            col, w, state = C_AV, 1.5, "disponible"
        dx, dy = x1 - x0, y1 - y0
        L = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / L, dy / L
        sx, sy, tx, ty = x0 + ux * .2, y0 + uy * .2, x1 - ux * .2, y1 - uy * .2
        fig.add_annotation(x=tx, y=ty, ax=sx, ay=sy, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=w, arrowcolor=col)
        detail = "<br>".join(f"{k}: {v:.2f} Gbps" for k, v in used.get(e, {}).items() if v > TOL)
        mx.append((x0 + x1) / 2); my.append((y0 + y1) / 2)
        mt.append(f"<b>{e[0]}→{e[1]}</b> ({state})<br>Costo: {r.Costo:g} | Cap: {r.Capacidad:g}"
                  f"<br>Flujo total: {tot:.2f} ({100*tot/r.Capacidad if r.Capacidad else 0:.0f}%)"
                  + (f"<br>{detail}" if detail else ""))
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=16, color="rgba(0,0,0,0)"),
                             hovertext=mt, hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[POS[n][0] for n in NODES], y=[POS[n][1] for n in NODES],
                             mode="markers+text", text=[str(n) for n in NODES],
                             textfont=dict(color="white", size=15),
                             marker=dict(size=40, color="#1e3a8a", line=dict(color="white", width=2)),
                             hoverinfo="skip", showlegend=False))
    for name, c in [("Disponible", C_AV), ("Utilizado", C_OK), ("Saturado", C_SAT), ("Desactivado", C_OFF)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=c, width=4), name=name))
    fig.update_layout(height=430, margin=dict(l=5, r=5, t=5, b=5), plot_bgcolor="white",
                      xaxis=dict(visible=False, range=[-.4, 4.4]), yaxis=dict(visible=False, range=[-.4, 2.4]),
                      legend=dict(orientation="h", y=-0.02))
    return fig


def apply_preset(name, edges, flows):
    e, f = edges.copy(), flows.copy()
    if name == "Falla: desactivar 3→6":
        e.loc[(e.Origen == 3) & (e.Destino == 6), "Activo"] = False
    elif name == "Tráfico: F1 = 10 Gbps":
        f.loc[f.Flujo == "F1", "Demanda"] = 10.0
    elif name == "Ampliación: cap 3→6 = 14":
        e.loc[(e.Origen == 3) & (e.Destino == 6), "Capacidad"] = 14.0
    return e, f


def scenarios(edges, flows):
    out = {"Base": (edges, flows)}
    for key, n in [("1. Falla", "Falla: desactivar 3→6"), ("2. Tráfico", "Tráfico: F1 = 10 Gbps"),
                   ("3. Ampliación", "Ampliación: cap 3→6 = 14")]:
        out[key] = apply_preset(n, edges, flows)
    f = flows.copy(); f["Demanda"] = f["Demanda"] * 2
    out["4. Validación ×2"] = (edges.copy(), f)
    return out


# ------------------------------------------------------------------- UI
st.set_page_config(page_title="DataNet Analytics", page_icon="🌐", layout="wide")
st.markdown("""<style>
.hero{background:linear-gradient(90deg,#1e3a8a,#0ea5e9);padding:22px 28px;border-radius:14px;color:white;margin-bottom:14px}
.hero h1{color:white;margin:0;font-size:2rem}.hero p{margin:4px 0 0;opacity:.9}
div[data-testid="stMetric"]{background:#f1f5f9;border-radius:12px;padding:12px 16px;border:1px solid #e2e8f0}
</style>
<div class="hero"><h1>🌐 DataNet Analytics</h1>
<p>Optimización de transmisión: multiflujo de costo mínimo con capacidad compartida (Gbps)</p></div>""",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Control")
    preset = st.radio("Escenario rápido", ["Base (datos del editor)", "Falla: desactivar 3→6",
                                           "Tráfico: F1 = 10 Gbps", "Ampliación: cap 3→6 = 14"])
    st.caption("Se aplica sobre los datos editados en la pestaña Optimización.")
    run = st.button("🚀 Optimizar", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["🛠️ Datos", "📊 Resultados", "🔀 Comparar escenarios"])

with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Enlaces")
        edges = st.data_editor(EDGES0, key="edges", use_container_width=True, hide_index=True,
                               column_config={"Costo": st.column_config.NumberColumn(min_value=0.0),
                                              "Capacidad": st.column_config.NumberColumn(min_value=0.0),
                                              "Activo": st.column_config.CheckboxColumn("¿Activo?")})
    with c2:
        st.subheader("Comunicaciones")
        flows = st.data_editor(FLOWS0, key="flows", num_rows="dynamic", use_container_width=True,
                               hide_index=True,
                               column_config={"Demanda": st.column_config.NumberColumn(min_value=0.0)})
    flows = flows.dropna()
    st.subheader("Vista previa de la red")
    st.plotly_chart(network_fig(edges, None), use_container_width=True)

if run:
    e2, f2 = apply_preset(preset, edges, flows)
    with st.spinner("Resolviendo con CBC..."):
        st.session_state["res"] = solve(e2, f2)
    st.session_state["edges_used"], st.session_state["preset"] = e2, preset

with tab2:
    res = st.session_state.get("res")
    if not res:
        st.info("Pulsa **Optimizar** en la barra lateral.")
    else:
        st.caption(f"Escenario: {st.session_state['preset']}")
        if res["status"] == "Optimal":
            tb = link_table(res)
            ok, ok_obj = validate(res)
            m = st.columns(5)
            m[0].metric("Estado", res["status"])
            m[1].metric("Costo total", f"{res['obj']:.2f}")
            m[2].metric("Enlaces usados", int((tb["Total"] > TOL).sum()))
            m[3].metric("Saturados", int((tb["Utilización %"] >= 99.99).sum()))
            m[4].metric("Validación", "✅ OK" if ok and ok_obj else "❌ Falla")
            st.plotly_chart(network_fig(st.session_state["edges_used"], res), use_container_width=True)
            used = tb[tb["Total"] > TOL]
            a, b = st.columns(2)
            with a:
                fig = go.Figure([go.Bar(name=k, x=used["Enlace"], y=used[k]) for k in res["K"]])
                fig.update_layout(barmode="stack", title="Flujo por enlace y comunicación (Gbps)", height=340)
                st.plotly_chart(fig, use_container_width=True)
            with b:
                colors = [C_SAT if u >= 99.99 else C_OK for u in used["Utilización %"]]
                fig = go.Figure(go.Bar(x=used["Enlace"], y=used["Utilización %"], marker_color=colors))
                fig.add_hline(y=100, line_dash="dash", line_color=C_SAT)
                fig.update_layout(title="Utilización de capacidad (%)", height=340, yaxis_range=[0, 110])
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(tb, use_container_width=True, hide_index=True)
        elif res["status"] == "Infeasible":
            st.error("⛔ Modelo INFACTIBLE: no se pueden satisfacer todas las demandas con los enlaces y "
                     "capacidades actuales. No se muestra ninguna solución.")
        else:
            st.warning(f"El solucionador terminó con estado '{res['status']}'. No hay solución válida.")

with tab3:
    st.write("Cada escenario se resuelve de forma independiente desde los datos del editor.")
    if st.button("▶️ Ejecutar todos los escenarios"):
        rows, util = [], {}
        for name, (e, f) in scenarios(edges, flows).items():
            r = solve(e, f)
            row = {"Escenario": name, "Estado": r["status"],
                   "Costo total": round(r["obj"], 2) if r["obj"] is not None else None}
            if r["status"] == "Optimal":
                tb = link_table(r)
                util[name] = tb.set_index("Enlace")["Utilización %"]
                row["Util. máx %"] = tb["Utilización %"].max()
                row["Saturados"] = ", ".join(tb[tb["Utilización %"] >= 99.99]["Enlace"]) or "-"
            rows.append(row)
        comp = pd.DataFrame(rows)
        base = comp.loc[0, "Costo total"]
        comp["Δ vs base"] = comp["Costo total"].apply(lambda v: None if pd.isna(v) else round(v - base, 2))
        st.session_state["comp"], st.session_state["util"] = comp, util
    if "comp" in st.session_state:
        comp, util = st.session_state["comp"], st.session_state["util"]
        st.dataframe(comp, use_container_width=True, hide_index=True)
        ok = comp.dropna(subset=["Costo total"])
        fig = go.Figure(go.Bar(x=ok["Escenario"], y=ok["Costo total"], text=ok["Costo total"],
                               marker_color="#0ea5e9"))
        fig.update_layout(title="Costo total por escenario", height=340)
        st.plotly_chart(fig, use_container_width=True)
        if util:
            u = pd.DataFrame(util).fillna(0)
            u = u[(u > 0).any(axis=1)]
            fig = go.Figure([go.Bar(name=c, x=u.index, y=u[c]) for c in u.columns])
            fig.update_layout(barmode="group", title="Utilización (%) por enlace y escenario", height=380)
            st.plotly_chart(fig, use_container_width=True)
