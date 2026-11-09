"""DataNet Analytics - Optimización multiflujo de costo mínimo (PuLP + Streamlit)."""
import pandas as pd
import pulp
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

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


def draw(edges, res):
    G = nx.DiGraph()
    G.add_nodes_from(NODES)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    nx.draw_networkx_nodes(G, POS, node_color="#dbe7f5", edgecolors="k", node_size=700, ax=ax)
    nx.draw_networkx_labels(G, POS, ax=ax)
    used = {}
    if res and res["x"]:
        for (i, j) in res["A"]:
            used[i, j] = sum(res["x"][k, i, j] for k in res["K"])
    for r in edges.itertuples():
        e = (int(r.Origen), int(r.Destino))
        if not r.Activo:
            color, style, w = "#cccccc", "dotted", 1
        elif e in used and used[e] >= r.Capacidad - TOL and used[e] > TOL:
            color, style, w = "red", "solid", 3          # saturado
        elif e in used and used[e] > TOL:
            color, style, w = "green", "solid", 3        # utilizado
        else:
            color, style, w = "gray", "solid", 1         # disponible
        nx.draw_networkx_edges(G, POS, edgelist=[e], edge_color=color, style=style, width=w,
                               arrows=True, arrowsize=18, node_size=700, ax=ax,
                               connectionstyle="arc3,rad=0.08")
        lbl = f"c={r.Costo:g}\n" + (f"{used[e]:.1f}/{r.Capacidad:g}" if e in used else f"cap {r.Capacidad:g}")
        nx.draw_networkx_edge_labels(G, POS, edge_labels={e: lbl}, font_size=7, ax=ax,
                                     label_pos=0.5, bbox=dict(alpha=0, edgecolor="none"))
    ax.set_title("Gris: disponible | Verde: utilizado | Rojo: saturado | Punteado: desactivado")
    ax.axis("off")
    return fig


# ------------------------------------------------------------------- UI
st.set_page_config(page_title="DataNet Analytics", layout="wide")
st.title("DataNet Analytics – Optimización de transmisión")
st.caption("Multiflujo de costo mínimo: 8 servidores, enlaces dirigidos, capacidad compartida (Gbps).")

tab1, tab2 = st.tabs(["Optimización", "Escenarios"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Enlaces (editable)")
        edges = st.data_editor(EDGES0, key="edges", use_container_width=True, hide_index=True,
                               column_config={"Costo": st.column_config.NumberColumn(min_value=0.0),
                                              "Capacidad": st.column_config.NumberColumn(min_value=0.0)})
    with c2:
        st.subheader("Comunicaciones (editable)")
        flows = st.data_editor(FLOWS0, key="flows", num_rows="dynamic", use_container_width=True,
                               hide_index=True,
                               column_config={"Demanda": st.column_config.NumberColumn(min_value=0.0)})
    flows = flows.dropna()
    if st.button("Optimizar", type="primary"):
        st.session_state["res"] = solve(edges, flows)
        st.session_state["edges_used"] = edges.copy()

    res = st.session_state.get("res")
    if res:
        st.markdown(f"**Estado del solucionador:** `{res['status']}`")
        if res["status"] == "Optimal":
            st.success(f"Costo total mínimo: {res['obj']:.2f}")
            ok, ok_obj = validate(res)
            st.write(f"Validación restricciones: {'✅' if ok else '❌'} | "
                     f"Validación función objetivo: {'✅' if ok_obj else '❌'}")
            st.pyplot(draw(st.session_state["edges_used"], res))
            tbl = link_table(res)
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        elif res["status"] == "Infeasible":
            st.error("Modelo INFACTIBLE: con los enlaces/capacidades actuales no se pueden "
                     "satisfacer todas las demandas. No se muestra ninguna solución.")
        else:
            st.warning(f"El solucionador terminó con estado '{res['status']}'. No hay solución válida.")


def scenarios(edges, flows):
    out = {}
    out["Base"] = (edges.copy(), flows.copy())
    e = edges.copy(); e.loc[(e.Origen == 3) & (e.Destino == 6), "Activo"] = False
    out["1. Falla 3→6"] = (e, flows.copy())
    f = flows.copy(); f.loc[f.Flujo == "F1", "Demanda"] = 10.0
    out["2. Tráfico F1=10"] = (edges.copy(), f)
    e = edges.copy(); e.loc[(e.Origen == 3) & (e.Destino == 6), "Capacidad"] = 14.0
    out["3. Ampliación 3→6=14"] = (e, flows.copy())
    # Prueba adicional de validación: demandas x2 (debe ser infactible o más cara)
    f = flows.copy(); f["Demanda"] = f["Demanda"] * 2
    out["4. Validación: demandas ×2"] = (edges.copy(), f)
    return out


with tab2:
    st.write("Se resuelven de forma independiente a partir de los datos actuales del editor.")
    if st.button("Ejecutar todos los escenarios"):
        rows, details = [], {}
        for name, (e, f) in scenarios(edges, flows).items():
            r = solve(e, f)
            row = {"Escenario": name, "Estado": r["status"],
                   "Costo total": round(r["obj"], 2) if r["obj"] is not None else None}
            if r["status"] == "Optimal":
                tb = link_table(r)
                row["Util. máx %"] = tb["Utilización %"].max()
                row["Util. prom % (enlaces usados)"] = round(tb[tb["Total"] > TOL]["Utilización %"].mean(), 1)
                row["Enlaces saturados"] = ", ".join(tb[tb["Utilización %"] >= 99.99]["Enlace"]) or "-"
                details[name] = tb
            rows.append(row)
        comp = pd.DataFrame(rows)
        base = comp.loc[0, "Costo total"]
        comp["Δ vs base"] = comp["Costo total"].apply(lambda v: None if pd.isna(v) else round(v - base, 2))
        st.dataframe(comp, use_container_width=True, hide_index=True)
        for name, tb in details.items():
            with st.expander(f"Detalle: {name}"):
                st.dataframe(tb, use_container_width=True, hide_index=True)
