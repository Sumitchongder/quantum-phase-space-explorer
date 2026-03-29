"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  app.py  —  Density Matrix & Wigner Function Dashboard  v4.0               ║
║  IIT Jodhpur · m25iqt013                                                    ║
║  Xanadu-grade professional CV quantum visualization                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
 
import pickle, math, os
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Quantum Phase Space Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Density Matrix & Wigner Function · IIT Jodhpur · m25iqt013"},
)
 
# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — vibrant, bright, professional
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');
 
:root {
  --bg:        #07090f;
  --bg2:       #0d1117;
  --bg3:       #161b27;
  --bg4:       #1c2333;
  --border:    #21293d;
  --border2:   #2d3f60;
  --violet:    #7c3aed;
  --indigo:    #4f46e5;
  --cyan:      #06b6d4;
  --sky:       #38bdf8;
  --pink:      #ec4899;
  --rose:      #f43f5e;
  --amber:     #f59e0b;
  --lime:      #84cc16;
  --emerald:   #10b981;
  --text:      #f1f5f9;
  --text2:     #94a3b8;
  --text3:     #475569;
  --glow-v:    rgba(124,58,237,0.35);
  --glow-c:    rgba(6,182,212,0.35);
  --glow-p:    rgba(236,72,153,0.35);
}
 
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(79,70,229,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(6,182,212,0.06) 0%, transparent 60%);
}
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border2);
}
[data-testid="stHeader"]          { background: transparent !important; }
html, body, [class*="css"]        { color: var(--text); font-family: 'Inter', sans-serif; }
.block-container                  { padding-top: 1.5rem !important; max-width: 1400px; }
 
/* ── Sidebar ── */
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] label    { font-family: 'Inter', sans-serif !important; font-size: 0.8rem !important; color: var(--text2) !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--text3); font-size: 0.73rem; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
 
/* ── Metrics ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, var(--bg3) 0%, var(--bg4) 100%);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 16px 18px;
    transition: transform 0.15s, box-shadow 0.15s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(79,70,229,0.2);
}
[data-testid="stMetricValue"]  { font-family: 'JetBrains Mono', monospace; color: var(--sky); font-size: 1.4rem !important; font-weight: 700; }
[data-testid="stMetricLabel"]  { color: var(--text2); font-size: 0.72rem; font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 0.06em; }
 
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 5px 6px;
    gap: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text2);
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 8px 18px;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--indigo), var(--violet)) !important;
    color: #fff !important;
    font-weight: 600;
    box-shadow: 0 2px 12px rgba(79,70,229,0.4);
}
 
/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--indigo), var(--violet));
    color: white; border: none; border-radius: 10px;
    font-family: 'Inter', sans-serif; font-size: 0.82rem; font-weight: 600;
    padding: 10px 24px; letter-spacing: 0.02em;
    transition: all 0.2s; box-shadow: 0 4px 14px rgba(79,70,229,0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--violet), #9333ea);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(124,58,237,0.45);
}
 
/* ── Selectbox / Slider ── */
[data-baseweb="select"] > div {
    background: var(--bg3) !important;
    border-color: var(--border2) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--text) !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.2) !important;
}
.stSlider > div > div > div { background: var(--border2) !important; }
.stSlider [data-testid="stThumb"] { background: var(--indigo) !important; box-shadow: 0 0 8px var(--glow-v); }
 
/* ── Checkboxes / Radio ── */
[data-testid="stCheckbox"] label span { color: var(--text) !important; font-size: 0.82rem; }
 
/* ── Cards ── */
.qcard {
    background: linear-gradient(135deg, var(--bg3), var(--bg4));
    border: 1px solid var(--border2);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
}
.qcard::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--indigo), var(--cyan), var(--pink));
}
.qcard-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--cyan);
    margin-bottom: 8px;
}
.qcard p { margin: 0; font-size: 0.84rem; color: var(--text2); line-height: 1.65; }
 
.eq-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-left: 3px solid var(--indigo);
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.83rem;
    color: #c4b5fd;
    line-height: 1.7;
}
 
.hero-banner {
    background: linear-gradient(135deg, var(--bg2) 0%, var(--bg3) 50%, var(--bg4) 100%);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79,70,229,0.18), transparent 70%);
    pointer-events: none;
}
.hero-banner::before {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(6,182,212,0.1), transparent 70%);
    pointer-events: none;
}
.hero-banner h1 {
    margin: 0 0 6px 0;
    font-family: 'Inter', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.03em;
}
.hero-banner .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.77rem;
    color: var(--text2);
    margin: 4px 0 14px 0;
    line-height: 1.6;
}
.pill {
    display: inline-block;
    background: rgba(79,70,229,0.15);
    border: 1px solid rgba(79,70,229,0.4);
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--sky);
    margin: 3px 3px 0 0;
}
.pill.pink   { background: rgba(236,72,153,0.12); border-color: rgba(236,72,153,0.35); color: #f9a8d4; }
.pill.cyan   { background: rgba(6,182,212,0.12);  border-color: rgba(6,182,212,0.35);  color: var(--cyan); }
.pill.lime   { background: rgba(132,204,22,0.12); border-color: rgba(132,204,22,0.35); color: var(--lime); }
.pill.amber  { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.35); color: var(--amber); }
 
.nc-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 14px; border-radius: 100px;
    font-family: 'Inter', sans-serif; font-size: 0.73rem; font-weight: 700;
    letter-spacing: 0.04em;
}
.nc-badge.quantum  { background: rgba(236,72,153,0.15); color: #f472b6; border: 1px solid rgba(236,72,153,0.4); }
.nc-badge.classical{ background: rgba(245,158,11,0.12); color: var(--amber); border: 1px solid rgba(245,158,11,0.35); }
 
.state-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.2s;
}
.state-card:hover { border-color: var(--indigo); box-shadow: 0 0 20px rgba(79,70,229,0.2); }
 
hr.qdiv { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
.stDataFrame { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
[data-testid="stExpander"] { background: var(--bg3); border: 1px solid var(--border); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
_LAY = dict(
    paper_bgcolor="#07090f",
    plot_bgcolor="#0d1117",
    font=dict(color="#f1f5f9", family="Inter, sans-serif", size=11),
    xaxis=dict(gridcolor="#161b27", zerolinecolor="#21293d", title_font=dict(size=11), tickfont=dict(size=9)),
    yaxis=dict(gridcolor="#161b27", zerolinecolor="#21293d", title_font=dict(size=11), tickfont=dict(size=9)),
    legend=dict(bgcolor="#0d1117", bordercolor="#21293d", borderwidth=1, font=dict(size=9)),
    margin=dict(l=48, r=24, t=50, b=40),
    colorway=["#4f46e5","#06b6d4","#ec4899","#84cc16","#f59e0b","#38bdf8","#f43f5e"],
)
 
# Vivid colormaps
W_CS   = "RdBu_r"          # classic diverging — red=positive, blue=negative
W_CS2  = [[0,"#0f172a"],[0.25,"#1e3a5f"],[0.5,"#07090f"],[0.75,"#4a0d2e"],[1,"#e11d48"]]
Q_CS   = [[0,"#07090f"],[0.3,"#1e1b4b"],[0.6,"#4c1d95"],[0.8,"#7c3aed"],[1,"#fbbf24"]]
DM_CS  = "RdBu_r"
PLASMA = "plasma"
 
# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).parent / "data"
 
@st.cache_resource(show_spinner=False)
def load_data():
    out = {}
    for name in ("states", "channels", "gbs"):
        p = DATA_DIR / f"{name}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                out[name] = pickle.load(f)
        else:
            out[name] = None
    return out
 
DATA = load_data()
 
def _check(key):
    if DATA[key] is None:
        st.error(f"**`data/{key}.pkl` not found.** Run `python generate_data.py` locally, then commit `data/` to GitHub.")
        return False
    return True
 
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_NC_GROUPS = {"fock", "squeezed", "cat", "displaced_squeezed", "gkp"}
 
def _wigner_range(W):
    wmin, wmax = float(W.min()), float(W.max())
    if wmin >= 0: wmin = -1e-9
    if wmax <= 0: wmax = 1e-9
    lim = max(abs(wmin), abs(wmax))
    return -lim, lim        # symmetric so zero = white
 
def fig_wigner(W_list, xvec, title="W(x,p)", height=440, colorscale=None):
    W  = np.array(W_list, dtype=float)
    xv = np.array(xvec)
    wmin, wmax = _wigner_range(W)
    cs = colorscale or W_CS
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=W, x=xv, y=xv, colorscale=cs, zmid=0, zmin=wmin, zmax=wmax,
        colorbar=dict(title=dict(text="W(x,p)", font=dict(size=10)), len=0.85, thickness=14,
                      tickfont=dict(size=9, color="#94a3b8"), outlinecolor="#21293d", outlinewidth=1),
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}<br>W=%{z:.5f}<extra></extra>",
    ))
    fig.add_trace(go.Contour(
        z=W, x=xv, y=xv,
        contours=dict(start=0, end=0, size=1, coloring="none"),
        line=dict(color="rgba(255,255,255,0.55)", width=1.5, dash="dot"),
        showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#38bdf8"), x=0.5),
        xaxis_title="Quadrature x", yaxis_title="Quadrature p",
        xaxis=dict(**_LAY["xaxis"], scaleanchor="y"),
        yaxis=dict(**_LAY["yaxis"]),
    )
    return fig
 
def fig_husimi(Q_list, xvec, title="Q(α)", height=440):
    Q  = np.array(Q_list)
    xv = np.array(xvec)
    fig = go.Figure(go.Heatmap(
        z=Q, x=xv, y=xv, colorscale=Q_CS,
        colorbar=dict(title=dict(text="Q(α)", font=dict(size=10)), len=0.85, thickness=14,
                      tickfont=dict(size=9, color="#94a3b8"), outlinecolor="#21293d", outlinewidth=1),
        hovertemplate="Re(α)=%{x:.2f}  Im(α)=%{y:.2f}<br>Q=%{z:.5f}<extra></extra>",
    ))
    fig.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#a78bfa"), x=0.5),
        xaxis_title="Re(α)", yaxis_title="Im(α)",
        xaxis=dict(**_LAY["xaxis"], scaleanchor="y"),
    )
    return fig
 
def fig_density_matrix(rho_list, title="ρ", height=400):
    rho = np.array(rho_list)
    sub = make_subplots(
        1, 2,
        subplot_titles=["Re(ρ<sub>mn</sub>)", "Im(ρ<sub>mn</sub>)"],
        horizontal_spacing=0.1,
    )
    for ci, dat in enumerate([np.real(rho), np.imag(rho)], 1):
        vm = max(float(np.abs(dat).max()), 1e-6)
        sub.add_trace(go.Heatmap(
            z=dat, colorscale=DM_CS, zmid=0, zmin=-vm, zmax=vm,
            showscale=(ci == 1),
            colorbar=dict(title=dict(text="ρ", font=dict(size=10)), x=1.02, len=0.9, thickness=12,
                          tickfont=dict(size=9, color="#94a3b8")),
            hovertemplate="n=%{y}  m=%{x}<br>val=%{z:.5f}<extra></extra>",
        ), row=1, col=ci)
    sub.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>Density Matrix {title}</b>", font=dict(size=13, color="#06b6d4"), x=0.5),
    )
    sub.update_xaxes(title_text="m (column)", title_font=dict(size=10))
    sub.update_yaxes(title_text="n (row)", title_font=dict(size=10))
    return sub
 
def fig_photon_dist(probs_list, mean_n, title="P(n)", height=340):
    probs = np.array(probs_list)
    k     = np.arange(len(probs))
    poi   = np.array([
        math.exp(-max(mean_n, 1e-9)) * (max(mean_n, 1e-9)**ki) / math.factorial(int(ki))
        if int(ki) < 170 else 0.0 for ki in k
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k, y=probs, name="P(n)",
        marker=dict(
            color=probs,
            colorscale=[[0,"#1e1b4b"],[0.5,"#4f46e5"],[1,"#06b6d4"]],
            showscale=False, line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        opacity=0.9,
    ))
    if mean_n > 0.01:
        fig.add_trace(go.Scatter(
            x=k, y=poi, mode="lines+markers", name="Poisson ref",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=4, color="#f59e0b"),
        ))
    fig.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=12, color="#38bdf8"), x=0.5),
        xaxis_title="Photon number n", yaxis_title="P(n)", bargap=0.12,
    )
    return fig
 
def fig_wigner_3d(W_list, xvec, title="W(x,p) — 3D Surface", height=500):
    W  = np.array(W_list, dtype=float)
    xv = np.array(xvec)
    wa = max(float(np.abs(W).max()), 1e-6)
    fig = go.Figure(go.Surface(
        z=W, x=xv, y=xv, colorscale=W_CS, cmin=-wa, cmax=wa,
        showscale=True, opacity=0.92, contours_z=dict(show=True, usecolormap=True, project_z=True),
        colorbar=dict(title=dict(text="W(x,p)", font=dict(size=10)), len=0.7, thickness=14,
                      tickfont=dict(size=9, color="#94a3b8")),
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}<br>W=%{z:.5f}<extra></extra>",
    ))
    fig.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#38bdf8"), x=0.5),
        scene=dict(
            xaxis=dict(title="x", gridcolor="#161b27", backgroundcolor="#07090f", showbackground=True),
            yaxis=dict(title="p", gridcolor="#161b27", backgroundcolor="#07090f", showbackground=True),
            zaxis=dict(title="W(x,p)", gridcolor="#161b27", backgroundcolor="#07090f", showbackground=True),
            bgcolor="#07090f",
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.85)),
        )
    )
    return fig
 
def metrics_row(m: dict, wnv: float, is_nc: bool):
    cols = st.columns(6)
    mq_val = m.get('mandel_Q')
    mq_str = f"{mq_val:.4f}" if mq_val is not None else "N/A"
    items = [
        ("⟨n̂⟩", f"{m.get('mean_n', 0):.4f}", "Mean photon number", "#38bdf8"),
        ("Purity", f"{m.get('purity', 0):.5f}", "Tr(ρ²)  [1 = pure state]", "#a78bfa"),
        ("Entropy", f"{m.get('entropy', 0):.4f}", "von Neumann S(ρ) in bits", "#06b6d4"),
        ("Mandel Q", mq_str, "< 0 = sub-Poissonian (quantum)", "#f472b6"),
        ("Δx · Δp", f"{m.get('heis_prod', 0):.5f}", "Heisenberg product (≥ 0.5)", "#fbbf24"),
        ("WNV", f"{wnv:.5f}", "Wigner Negativity Volume (>0 = quantum)", "#84cc16" if wnv > 0.001 else "#94a3b8"),
    ]
    for col, (lbl, val, hlp, color) in zip(cols, items):
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1117,#161b27);border:1px solid #21293d;
                    border-top:2px solid {color};border-radius:14px;padding:14px 16px;
                    transition:transform 0.15s;">
          <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
                      color:#475569;margin-bottom:6px;font-family:'Inter',sans-serif;">{lbl}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.15rem;font-weight:700;
                      color:{color};">{val}</div>
          <div style="font-size:0.67rem;color:#334155;margin-top:4px;font-family:'Inter',sans-serif;">{hlp}</div>
        </div>
        """, unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.markdown("""
    <div style="padding:20px 16px 14px;text-align:center;">
      <div style="font-size:2.8rem;line-height:1;filter:drop-shadow(0 0 16px rgba(79,70,229,0.7));">⚛️</div>
      <div style="font-family:'Inter',sans-serif;font-weight:900;font-size:1.0rem;color:#f1f5f9;
                  letter-spacing:-0.02em;margin-top:10px;line-height:1.2;">
        Quantum Phase Space<br>
        <span style="font-size:0.75rem;font-weight:500;color:#4f46e5;">Dashboard v4.0</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.64rem;color:#334155;margin-top:6px;">
        IIT Jodhpur · m25iqt013
      </div>
    </div>
    <hr style="border-color:#21293d;margin:0 0 12px 0;">
    <div style="padding:0 8px 6px;font-family:'Inter',sans-serif;font-size:0.65rem;
                font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#334155;">
      Navigate
    </div>
    """, unsafe_allow_html=True)
 
    page = st.sidebar.radio(
        "nav",
        ["🔬 State Explorer",
         "🌌 Phase Space Zoo",
         "🧪 Witness Lab",
         "⚡ Channel Simulator",
         "🔭 GBS & Quantum ML"],
        label_visibility="collapsed",
    )
 
    st.sidebar.markdown("<hr style='border-color:#21293d;margin:14px 0 10px'>", unsafe_allow_html=True)
 
    # Quick stats
    if DATA.get("states"):
        SD = DATA["states"]
        n_states = sum(len(v) for k, v in SD.items() if isinstance(v, dict))
        st.sidebar.markdown(f"""
        <div style="padding:10px 12px;background:#0d1117;border:1px solid #21293d;border-radius:12px;
                    font-family:'JetBrains Mono',monospace;font-size:0.7rem;line-height:2.0;color:#475569;">
          <span style="color:#4f46e5;">●</span> {n_states} precomputed states<br>
          <span style="color:#06b6d4;">●</span> Wigner · Husimi · ρ<br>
          <span style="color:#ec4899;">●</span> QuTiP + Strawberry Fields<br>
          <span style="color:#84cc16;">●</span> IIT Jodhpur CVQIP 2025
        </div>
        """, unsafe_allow_html=True)
 
    st.sidebar.markdown("""
    <div style="padding:12px 0 4px;font-family:'Inter',sans-serif;font-size:0.63rem;
                color:#1e293b;text-align:center;">
    Density Matrix & Wigner Function<br>for Various Quantum States
    </div>
    """, unsafe_allow_html=True)
    return page
 
# ══════════════════════════════════════════════════════════════════════════════
# STATE INFO
# ══════════════════════════════════════════════════════════════════════════════
STATE_INFO = {
    "Fock |n⟩": {
        "emoji": "🎯", "color": "#f472b6", "classical": False,
        "eq": "|n⟩  —  â†â|n⟩ = n|n⟩  (Fock / number state)",
        "short": "Exact photon number, maximum phase uncertainty.",
        "desc": "Fock states have exactly n photons — perfect number certainty, maximum phase uncertainty. Their Wigner function goes negative, a rigorous proof of non-classicality. Used as building blocks in quantum optics.",
    },
    "Coherent |α⟩": {
        "emoji": "💡", "color": "#fbbf24", "classical": True,
        "eq": "|α⟩ = e^{−|α|²/2} Σ (αⁿ/√n!) |n⟩",
        "short": "Most classical quantum state — laser light.",
        "desc": "Coherent states are the most classical quantum states — they model laser light. Their Wigner function is a Gaussian, always positive. They saturate the Heisenberg uncertainty: ΔxΔp = ½.",
    },
    "Squeezed |r,φ⟩": {
        "emoji": "🔧", "color": "#818cf8", "classical": False,
        "eq": "|r,φ⟩ = S(ξ)|0⟩,  S(ξ) = exp(½(ξ*â² − ξ↠²))",
        "short": "Noise below shot-noise limit in one quadrature.",
        "desc": "Squeezing trades noise between quadratures. One direction is squeezed below the vacuum noise level — used in LIGO gravitational-wave detectors and CV-QKD. The Wigner function is an elliptical Gaussian.",
    },
    "Thermal ρ̂_th": {
        "emoji": "🌡️", "color": "#fb923c", "classical": True,
        "eq": "ρ_th = Σₙ [n̄ⁿ/(1+n̄)ⁿ⁺¹] |n⟩⟨n|",
        "short": "Blackbody radiation — classical mixed state.",
        "desc": "Thermal states model blackbody radiation. They are mixed states with super-Poissonian photon statistics (Mandel Q > 0). The Wigner function is a broad Gaussian — classical but noisy.",
    },
    "Cat State": {
        "emoji": "🐱", "color": "#4ade80", "classical": False,
        "eq": "|cat⟩ = N(|α⟩ ± |−α⟩)  (Schrödinger's cat)",
        "short": "Macroscopic quantum superposition with interference fringes.",
        "desc": "Superposition of two coherent states — macroscopic quantum superposition. Shows spectacular interference fringes in the Wigner function. The fringes are a direct signature of quantum coherence and non-classicality.",
    },
    "Displaced-Squeezed": {
        "emoji": "🌀", "color": "#22d3ee", "classical": False,
        "eq": "|α,r⟩ = D̂(α)Ŝ(r)|0⟩",
        "short": "Squeezed state shifted in phase space — CV-QKD resource.",
        "desc": "A squeezed state displaced in phase space. Combines displacement D̂(α) and squeezing Ŝ(r). The Wigner function is an elliptical Gaussian shifted from the origin. Central to CV quantum key distribution protocols.",
    },
    "GKP State": {
        "emoji": "🛡️", "color": "#c084fc", "classical": False,
        "eq": "|GKP⟩ ∝ Σₙ e^{−δ²n²} D̂(n√π)|0⟩",
        "short": "Gottesman-Kitaev-Preskill — photonic error correction.",
        "desc": "GKP states are a grid of displaced vacuum states in phase space. They encode a logical qubit in the continuous-variable space and are the leading candidate for fault-tolerant photonic quantum computing.",
    },
}
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STATE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def page_state_explorer():
    if not _check("states"): return
    SD   = DATA["states"]
    xvec = SD["xvec"]
 
    st.markdown("""
    <div class="hero-banner">
      <h1>🔬 State Explorer</h1>
      <div class="subtitle">
        Interactive exploration of continuous-variable quantum states.<br>
        Wigner function W(x,p) · Husimi Q(α) · Density matrix ρ · Photon statistics P(n)
      </div>
      <span class="pill">Wigner Function</span>
      <span class="pill cyan">Husimi Q</span>
      <span class="pill pink">Density Matrix ρ</span>
      <span class="pill lime">Photon Statistics</span>
      <span class="pill amber">7 Quantum States</span>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Layout: sidebar controls | main content ──
    col_ctrl, col_main = st.columns([1, 3.4], gap="large")
 
    with col_ctrl:
        # State selector with colored pills
        st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.72rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:8px;">
                    Select Quantum State</div>""", unsafe_allow_html=True)
        state_type = st.selectbox("state", list(STATE_INFO.keys()), label_visibility="collapsed")
        info = STATE_INFO[state_type]
        is_nc = not info["classical"]
 
        # State info card
        nc_html = (
            '<span class="nc-badge quantum">⚡ NON-CLASSICAL</span>'
            if is_nc else
            '<span class="nc-badge classical">☀️ CLASSICAL / GAUSSIAN</span>'
        )
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba({
            '124,58,237' if is_nc else '245,158,11'
        },0.08),rgba({
            '79,70,229' if is_nc else '161,98,7'
        },0.05));border:1px solid rgba({
            '124,58,237' if is_nc else '245,158,11'
        },0.3);border-radius:14px;padding:16px;margin:10px 0;">
          <div style="font-size:1.8rem;margin-bottom:8px;">{info['emoji']}</div>
          {nc_html}
          <div style="font-size:0.82rem;color:#cbd5e1;line-height:1.6;margin-top:10px;">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown(f'<div class="eq-box">{info["eq"]}</div>', unsafe_allow_html=True)
 
        st.markdown("<hr class='qdiv'>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.72rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:8px;">
                    Parameters</div>""", unsafe_allow_html=True)
 
        if state_type == "Fock |n⟩":
            n_val = st.select_slider("Photon number n", options=[0, 1, 2, 3, 5], value=1)
            key, group = n_val, "fock"
        elif state_type == "Coherent |α⟩":
            opts = ["0,0","1,0","2,0","1,1","2,2","-2,0","0,2"]
            preset = st.selectbox("α = Re + i·Im", opts)
            key, group = preset, "coherent"
        elif state_type == "Squeezed |r,φ⟩":
            opts = ["0.5,0","1.0,0","1.5,0","2.0,0","1.0,1.57"]
            labels = ["r=0.5, φ=0","r=1.0, φ=0","r=1.5, φ=0","r=2.0, φ=0","r=1.0, φ=π/2"]
            preset = st.selectbox("(r, φ)", opts, format_func=lambda x: labels[opts.index(x)])
            key, group = preset, "squeezed"
        elif state_type == "Thermal ρ̂_th":
            n_bar = st.select_slider("Mean photon number n̄", options=[0.5, 1, 2, 5, 10])
            key, group = n_bar, "thermal"
        elif state_type == "Cat State":
            ctype = st.radio("Cat type", ["Even  |α⟩+|−α⟩", "Odd  |α⟩−|−α⟩",
                                          "Even α=2", "Odd α=2"], index=0)
            km = {"Even  |α⟩+|−α⟩":"even","Odd  |α⟩−|−α⟩":"odd",
                  "Even α=2":"even_2","Odd α=2":"odd_2"}
            key, group = km[ctype], "cat"
        elif state_type == "Displaced-Squeezed":
            opts = ["(1+i, 0.5, 0)","(2, 1.0, 0)","(1+2i, 0.8, π/2)"]
            preset = st.selectbox("(α, r, φ)", opts)
            km = {"(1+i, 0.5, 0)":"(1+1j),0.5,0",
                  "(2, 1.0, 0)":"(2+0j),1.0,0",
                  "(1+2i, 0.8, π/2)":"(1+2j),0.8,1.57"}
            key, group = km[preset], "displaced_squeezed"
        else:  # GKP
            delta = st.radio("δ (envelope width)", ["δ = 0.3 (narrow)", "δ = 0.5 (wide)"])
            key, group = ("0.3,3" if "0.3" in delta else "0.5,3"), "gkp"
 
        st.markdown("<hr class='qdiv'>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Inter',sans-serif;font-size:0.72rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:8px;">
                    Display Options</div>""", unsafe_allow_html=True)
        view_3d   = st.toggle("🏔️ 3D Wigner Surface", value=False)
        show_dm   = st.toggle("🔢 Show Density Matrix", value=True)
        show_both = st.toggle("🔀 Wigner + Husimi side-by-side", value=False)
 
    # ── Load state data ──
    try:
        state_d = SD[group][key]
    except KeyError:
        state_d = SD[group][list(SD[group].keys())[0]]
 
    m   = state_d["metrics"]
    wnv = state_d.get("wnv", 0.0)
 
    with col_main:
        # Metrics row
        metrics_row(m, wnv, is_nc)
        st.markdown("<hr class='qdiv'>", unsafe_allow_html=True)
 
        if show_both:
            # Side-by-side Wigner + Husimi
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_wigner(state_d["W"], xvec,
                    title=f"Wigner W(x,p) — {state_type}", height=420), use_container_width=True)
            with c2:
                st.plotly_chart(fig_husimi(state_d["Q"], xvec,
                    title=f"Husimi Q(α) — {state_type}", height=420), use_container_width=True)
            # Density matrix below
            if show_dm and "rho" in state_d:
                st.plotly_chart(fig_density_matrix(state_d["rho"],
                    title=f"— {state_type}", height=380), use_container_width=True)
            st.plotly_chart(fig_photon_dist(m["probs"], m["mean_n"],
                title=f"Photon Number Distribution P(n) — {state_type}"), use_container_width=True)
 
        else:
            # Tabbed view
            tab1, tab2, tab3, tab4 = st.tabs([
                "🌊  Wigner W(x,p)", "🌀  Husimi Q(α)", "📊  Photon Statistics P(n)", "📐  Density Matrix ρ"
            ])
 
            with tab1:
                if view_3d:
                    st.plotly_chart(fig_wigner_3d(state_d["W"], xvec,
                        title=f"W(x,p) — {state_type}"), use_container_width=True)
                else:
                    st.plotly_chart(fig_wigner(state_d["W"], xvec,
                        title=f"Wigner Function W(x,p) — {state_type}"), use_container_width=True)
 
                neg_str = f"{wnv:.5f}" if wnv > 0.001 else "≈ 0"
                if wnv > 0.001:
                    nc_text = "⚡ <b>Non-classical</b> — Wigner function has negative regions (red areas above)."
                elif is_nc:
                    nc_text = "⚡ <b>Non-classical</b> — quantum features present (squeezed noise / coherence), no Wigner negativity for this preset."
                else:
                    nc_text = "☀️ <b>Classical / Gaussian</b> — Wigner function is everywhere non-negative."
                st.markdown(f"""
                <div class="qcard">
                  <div class="qcard-title">💡 Wigner Function</div>
                  <p>W(x,p) is a quasi-probability distribution in phase space.
                  <b>Negative regions (blue/dark)</b> are impossible classically and prove quantum behaviour.
                  WNV = <b>{neg_str}</b>. {nc_text}</p>
                </div>""", unsafe_allow_html=True)
 
            with tab2:
                st.plotly_chart(fig_husimi(state_d["Q"], xvec,
                    title=f"Husimi Q Function Q(α) — {state_type}"), use_container_width=True)
                st.markdown("""
                <div class="qcard">
                  <div class="qcard-title">💡 Husimi Q-Function</div>
                  <p>Q(α) = ⟨α|ρ|α⟩/π — always non-negative, unlike the Wigner function.
                  Brighter yellow regions show where the state "lives" in phase space.
                  Smoother than W(x,p) but loses some quantum information about coherences.</p>
                </div>""", unsafe_allow_html=True)
 
            with tab3:
                st.plotly_chart(fig_photon_dist(m["probs"], m["mean_n"],
                    title=f"Photon Number Distribution P(n) — {state_type}"), use_container_width=True)
                mq = m.get("mandel_Q")
                if mq is not None:
                    if mq < -0.001:
                        regime = f"<b style='color:#f472b6'>Q = {mq:.4f} → Sub-Poissonian</b> (non-classical, fewer fluctuations than a laser)"
                    elif mq > 0.001:
                        regime = f"<b style='color:#fbbf24'>Q = {mq:.4f} → Super-Poissonian</b> (classical noise, thermal-like bunching)"
                    else:
                        regime = f"<b style='color:#4ade80'>Q ≈ 0 → Poissonian</b> (coherent state statistics)"
                    st.markdown(f"""
                    <div class="qcard">
                      <div class="qcard-title">💡 Photon Statistics</div>
                      <p>Mandel Q parameter: {regime}.<br>
                      Yellow dashed = Poisson reference (coherent state).
                      Q &lt; 0 is a quantum signature: sub-shot-noise photon-number variance.</p>
                    </div>""", unsafe_allow_html=True)
 
            with tab4:
                if show_dm:
                    if "rho" in state_d:
                        st.plotly_chart(fig_density_matrix(state_d["rho"],
                            title=f"— {state_type}", height=420), use_container_width=True)
                        rho_arr = np.array(state_d["rho"])
                        diag    = np.real(np.diag(rho_arr))
                        off_max = float(np.abs(rho_arr - np.diag(np.diag(rho_arr))).max())
                        st.markdown(f"""
                        <div class="qcard">
                          <div class="qcard-title">💡 Density Matrix ρ</div>
                          <p>ρ encodes <b>everything</b> about the quantum state.
                          The diagonal ρ<sub>nn</sub> = P(n) gives photon probabilities.
                          Off-diagonal elements (coherences) signal quantum superposition — max |ρ<sub>mn</sub>| = <b>{off_max:.5f}</b>.
                          Purity Tr(ρ²) = <b>{m.get('purity', 0):.5f}</b>
                          {'(pure state ✅)' if m.get('purity', 0) > 0.999 else '(mixed state — some decoherence)'}.</p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        # Compute density matrix from Wigner for states without stored rho
                        st.info("ℹ️ Full density matrix not stored for this state (memory optimisation). "
                                "Showing photon distribution P(n) = ρₙₙ instead.")
                        st.plotly_chart(fig_photon_dist(m["probs"], m["mean_n"],
                            title=f"P(n) = ρₙₙ diagonal — {state_type}"), use_container_width=True)
                else:
                    st.info("Toggle **Show Density Matrix** in the sidebar to display ρ.")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PHASE SPACE ZOO
# ══════════════════════════════════════════════════════════════════════════════
def page_phase_space_zoo():
    if not _check("states"): return
    SD   = DATA["states"]
    xvec = SD["xvec"]
 
    st.markdown("""
    <div class="hero-banner">
      <h1>🌌 Phase Space Zoo</h1>
      <div class="subtitle">
        All 8 quantum states side-by-side. Switch between Wigner and Husimi representations.<br>
        Spot the non-classical signatures at a glance — interference fringes, negative regions, grid patterns.
      </div>
      <span class="pill">8 States</span>
      <span class="pill cyan">Wigner · Husimi</span>
      <span class="pill pink">Non-classicality</span>
      <span class="pill lime">Scorecard</span>
    </div>
    """, unsafe_allow_html=True)
 
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        rep = st.radio("**Representation**", ["🌊 Wigner W(x,p)", "🌀 Husimi Q(α)"], horizontal=True)
    with c2:
        cs_choice = st.radio("**Colormap**", ["RdBu (classic)", "Plasma", "Custom"], horizontal=True)
    with c3:
        n_cols = st.selectbox("Grid columns", [4, 3, 2], index=0)
 
    cs_map = {"RdBu (classic)": "RdBu_r", "Plasma": "plasma", "Custom": W_CS2}
    chosen_cs = cs_map[cs_choice]
 
    state_configs = [
        ("fock", 0,      "Fock |0⟩  (vacuum)",    False),
        ("fock", 1,      "Fock |1⟩",               False),
        ("fock", 2,      "Fock |2⟩",               False),
        ("coherent","2,0","Coherent |α=2⟩",         True),
        ("squeezed","1.0,0","Squeezed r=1",          False),
        ("thermal", 1,   "Thermal n̄=1",             True),
        ("cat","even",   "Cat (even, α=1.5)",        False),
        ("gkp","0.3,3",  "GKP δ=0.3",               False),
    ]
 
    cols = st.columns(n_cols)
    for idx, (group, key, label, classical) in enumerate(state_configs):
        try:
            sd = SD[group][key]
        except KeyError:
            sd = SD[group][list(SD[group].keys())[0]]
 
        wnv    = sd.get("wnv", 0.0)
        is_nc  = (group in _NC_GROUPS) or (wnv > 0.001)
        arr    = sd["W"] if "Wigner" in rep else sd["Q"]
        cs     = chosen_cs if "Wigner" in rep else Q_CS
        W_np   = np.array(arr, dtype=float)
        xv     = np.array(xvec)
 
        if "Wigner" in rep:
            wmin, wmax = _wigner_range(W_np)
        else:
            wmin, wmax = float(W_np.min()), float(W_np.max())
 
        badge_color = "#f472b6" if is_nc else "#f59e0b"
        badge_text  = "⚡ Quantum" if is_nc else "☀️ Classical"
 
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=W_np, x=xv, y=xv, colorscale=cs,
            zmin=wmin, zmax=wmax,
            zmid=0 if "Wigner" in rep else None,
            showscale=False,
            hovertemplate=f"<b>{label}</b><br>x=%{{x:.2f}}  p=%{{y:.2f}}<br>val=%{{z:.4f}}<extra></extra>",
        ))
        if "Wigner" in rep:
            fig.add_trace(go.Contour(
                z=W_np, x=xv, y=xv,
                contours=dict(start=0, end=0, size=1, coloring="none"),
                line=dict(color="rgba(255,255,255,0.6)", width=1.5),
                showscale=False, hoverinfo="skip",
            ))
 
        fig.update_layout(
            paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
            font=dict(color="#f1f5f9", size=9, family="Inter, sans-serif"),
            xaxis=dict(gridcolor="#161b27", tickfont=dict(size=7), title="x", title_font=dict(size=8),
                       scaleanchor="y"),
            yaxis=dict(gridcolor="#161b27", tickfont=dict(size=7), title="p", title_font=dict(size=8)),
            title=dict(
                text=f"<b>{label}</b>  <span style='color:{badge_color};font-size:9px'>{badge_text}</span>",
                font=dict(size=11, color="#f1f5f9"), x=0.5,
            ),
            margin=dict(l=30, r=8, t=46, b=28), height=255,
            annotations=[dict(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text=f"WNV={wnv:.3f}" if wnv > 0.001 else "WNV≈0",
                font=dict(size=8, color=badge_color, family="JetBrains Mono"),
                showarrow=False, xanchor="left", yanchor="top",
            )],
        )
        cols[idx % n_cols].plotly_chart(fig, use_container_width=True)
 
    st.markdown("<hr class='qdiv'>", unsafe_allow_html=True)
 
    # Scorecard table
    cola, colb = st.columns([2, 1])
    with cola:
        st.markdown("### 📊 Non-classicality Scorecard")
        rows = []
        for group, key, label, _ in state_configs:
            try:
                sd = SD[group][key]
            except KeyError:
                sd = list(SD[group].values())[0]
            m   = sd["metrics"]
            wnv = sd.get("wnv", 0.0)
            is_nc = (group in _NC_GROUPS) or (wnv > 0.001)
            mq = m.get("mandel_Q")
            rows.append({
                "State": label, "Purity": round(m.get("purity", 0), 4),
                "S(ρ) bits": round(m.get("entropy", 0), 4),
                "Mandel Q": round(mq, 4) if mq is not None else None,
                "WNV": round(wnv, 5), "Δx·Δp": round(m.get("heis_prod", 0), 4),
                "Quantum ✅": "⚡ Yes" if is_nc else "☀️ No",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=320)
 
    with colb:
        st.markdown("### 🏆 WNV Ranking")
        df_sorted = df[df["WNV"] > 0].sort_values("WNV", ascending=False)
        if not df_sorted.empty:
            fig_bar = go.Figure(go.Bar(
                x=df_sorted["WNV"], y=df_sorted["State"],
                orientation="h",
                marker=dict(
                    color=df_sorted["WNV"],
                    colorscale=[[0,"#4f46e5"],[0.5,"#ec4899"],[1,"#f59e0b"]],
                    showscale=False,
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"{v:.4f}" for v in df_sorted["WNV"]],
                textposition="outside", textfont=dict(size=9, color="#94a3b8"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
                font=dict(color="#f1f5f9", size=9),
                xaxis=dict(gridcolor="#161b27", title="WNV", title_font=dict(size=9)),
                yaxis=dict(gridcolor="#161b27", tickfont=dict(size=9)),
                margin=dict(l=10, r=60, t=20, b=30), height=280,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
 
    st.markdown("""
    <div class="qcard" style="margin-top:8px;">
      <div class="qcard-title">💡 Reading the table</div>
      <p><b>WNV</b> (Wigner Negativity Volume) = ∫|W|dxdp − 1. Any WNV > 0 is a <i>rigorous proof</i> of non-classicality.
      <b>Mandel Q &lt; 0</b> = sub-Poissonian statistics (quantum).
      <b>Purity = 1</b> = pure state. <b>S(ρ) = 0</b> = no classical uncertainty.
      Note: coherent and thermal states always have WNV = 0 (classical Gaussian Wigner functions).</p>
    </div>
    """, unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WITNESS LAB
# ══════════════════════════════════════════════════════════════════════════════
def page_witness_lab():
    if not _check("states"): return
    SD = DATA["states"]
 
    st.markdown("""
    <div class="hero-banner">
      <h1>🧪 Witness Lab</h1>
      <div class="subtitle">
        Quantum non-classicality witnesses across all states.<br>
        Wigner negativity · Mandel Q · Heisenberg uncertainty · Purity vs Entropy scatter
      </div>
      <span class="pill">WNV Witness</span>
      <span class="pill cyan">Mandel Q</span>
      <span class="pill pink">Heisenberg ΔxΔp</span>
      <span class="pill lime">Purity vs Entropy</span>
    </div>
    """, unsafe_allow_html=True)
 
    STATE_KEYS = [
        ("fock",0,"Fock |0⟩"),("fock",1,"Fock |1⟩"),("fock",2,"Fock |2⟩"),("fock",3,"Fock |3⟩"),
        ("coherent","0,0","Coherent |0⟩"),("coherent","1,0","Coherent |1⟩"),("coherent","2,0","Coherent |2⟩"),
        ("squeezed","0.5,0","Sq r=0.5"),("squeezed","1.0,0","Sq r=1"),("squeezed","2.0,0","Sq r=2"),
        ("thermal",0.5,"Thermal n̄=½"),("thermal",1,"Thermal n̄=1"),("thermal",5,"Thermal n̄=5"),
        ("cat","even","Cat even α=1.5"),("cat","odd","Cat odd α=1.5"),("cat","even_2","Cat even α=2"),
        ("displaced_squeezed","(1+1j),0.5,0","Disp-Sq"),
        ("gkp","0.3,3","GKP δ=0.3"),
    ]
 
    rows = []
    for group, key, label in STATE_KEYS:
        try:
            sd = SD[group][key]
        except KeyError:
            continue
        m   = sd["metrics"]
        wnv = sd.get("wnv", 0.0)
        is_nc = (group in _NC_GROUPS) or (wnv > 0.001)
        mq = m.get("mandel_Q")
        rows.append({
            "State": label, "⟨n̂⟩": m["mean_n"], "Purity": m["purity"],
            "Entropy": m["entropy"],
            "Mandel Q": mq if mq is not None else float("nan"),
            "Δx": m["delta_x"], "Δp": m["delta_p"], "ΔxΔp": m["heis_prod"],
            "WNV": round(wnv, 5), "Non-classical": is_nc,
            "Group": group,
        })
    df = pd.DataFrame(rows)
 
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Full Table", "🌊 WNV Chart", "🔭 Purity vs Entropy",
        "⚖️ Heisenberg", "📊 Mandel Q"
    ])
 
    with tab1:
        def color_nc(val):
            return "color: #84cc16; font-weight: 700" if val else "color: #475569"
        def fmt_mq(v):
            if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
            return f"{v:.4f}"
 
        st.dataframe(
            df.drop(columns=["Group"]).style
              .map(color_nc, subset=["Non-classical"])
              .format({
                  "⟨n̂⟩": "{:.3f}", "Purity": "{:.5f}", "Entropy": "{:.4f}",
                  "Mandel Q": fmt_mq,
                  "Δx": "{:.4f}", "Δp": "{:.4f}", "ΔxΔp": "{:.4f}", "WNV": "{:.5f}",
              }),
            use_container_width=True, height=520,
        )
 
    with tab2:
        # Gradient WNV bar chart
        df_wnv = df[df["WNV"] > 0].sort_values("WNV", ascending=False)
        df_zero = df[df["WNV"] <= 0]
        fig = go.Figure()
        if not df_wnv.empty:
            fig.add_trace(go.Bar(
                x=df_wnv["State"], y=df_wnv["WNV"],
                name="WNV > 0  (quantum)",
                marker=dict(
                    color=df_wnv["WNV"],
                    colorscale=[[0,"#4f46e5"],[0.5,"#ec4899"],[1,"#fbbf24"]],
                    showscale=True,
                    colorbar=dict(title="WNV", thickness=10, len=0.6, x=1.02),
                    line=dict(color="rgba(0,0,0,0)"),
                ),
                text=[f"{v:.4f}" for v in df_wnv["WNV"]],
                textposition="outside", textfont=dict(size=8),
            ))
        if not df_zero.empty:
            fig.add_trace(go.Bar(
                x=df_zero["State"], y=df_zero["WNV"],
                name="WNV = 0  (classical)",
                marker_color="#1e293b",
                marker_line=dict(color="#334155", width=1),
            ))
        fig.add_hline(y=0, line_color="#475569", line_dash="solid", line_width=1)
        fig.update_layout(**_LAY, height=460,
            title=dict(text="<b>Wigner Negativity Volume — WNV = ∫|W|dxdp − 1</b>",
                       font=dict(size=13, color="#38bdf8"), x=0.5),
            xaxis_tickangle=-38, xaxis_title="", yaxis_title="WNV",
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="qcard">
          <div class="qcard-title">💡 Wigner Negativity Volume</div>
          <p>WNV > 0 is a <b>rigorous necessary condition</b> for non-classicality — any state with negative Wigner
          function regions cannot be described by a classical phase-space distribution.
          Fock, Cat, and GKP states have WNV > 0. Squeezed and displaced-squeezed states are quantum
          but have positive Wigner functions (their non-classicality is revealed by other witnesses).</p>
        </div>""", unsafe_allow_html=True)
 
    with tab3:
        fig = go.Figure()
        group_colors = {
            "fock":"#f472b6", "coherent":"#fbbf24", "squeezed":"#818cf8",
            "thermal":"#fb923c", "cat":"#4ade80", "displaced_squeezed":"#22d3ee", "gkp":"#c084fc",
        }
        for grp, grp_df in df.groupby("Group"):
            color = group_colors.get(grp, "#94a3b8")
            is_nc_grp = grp in _NC_GROUPS
            fig.add_trace(go.Scatter(
                x=grp_df["Purity"], y=grp_df["Entropy"],
                mode="markers+text",
                name=grp.replace("_"," ").title(),
                marker=dict(color=color, size=14,
                            symbol="diamond" if is_nc_grp else "circle",
                            line=dict(color="white", width=1.5),
                            opacity=0.9),
                text=grp_df["State"], textposition="top center",
                textfont=dict(size=8, color=color),
            ))
        fig.update_layout(**_LAY, height=480,
            title=dict(text="<b>Purity  vs  von Neumann Entropy S(ρ)</b>",
                       font=dict(size=13, color="#38bdf8"), x=0.5),
            xaxis_title="Purity  Tr(ρ²)", yaxis_title="Entropy S(ρ)  [bits]",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="qcard">
          <div class="qcard-title">💡 Purity vs Entropy</div>
          <p>Pure states (Purity=1, S=0) sit at top-left. Mixed states spread toward lower-right.
          Thermal states are maximally mixed for their energy. Note: non-classicality ≠ purity —
          a pure coherent state is classical while a mixed GKP state can be highly non-classical.
          Diamonds = non-classical groups; circles = classical groups.</p>
        </div>""", unsafe_allow_html=True)
 
    with tab4:
        colors_heis = ["#ec4899" if row["Non-classical"] else "#fbbf24" for _, row in df.iterrows()]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["State"], y=df["ΔxΔp"],
            marker=dict(
                color=df["ΔxΔp"],
                colorscale=[[0,"#1e3a5f"],[0.4,"#4f46e5"],[0.7,"#ec4899"],[1,"#fbbf24"]],
                showscale=True,
                colorbar=dict(title="Δx·Δp", thickness=10, len=0.6),
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=[f"{v:.4f}" for v in df["ΔxΔp"]], textposition="outside",
            textfont=dict(size=8),
        ))
        fig.add_hline(y=0.5, line_color="#4ade80", line_dash="dash", line_width=2,
                      annotation_text="Heisenberg limit:  Δx·Δp = ½",
                      annotation_font=dict(color="#4ade80", size=10),
                      annotation_position="top right")
        fig.update_layout(**_LAY, height=460,
            title=dict(text="<b>Heisenberg Uncertainty Product  Δx · Δp  ≥  ½</b>",
                       font=dict(size=13, color="#38bdf8"), x=0.5),
            xaxis_tickangle=-38, xaxis_title="", yaxis_title="Δx · Δp",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="eq-box">
        Heisenberg Uncertainty Principle:   Δx · Δp ≥ ½<br>
        Minimum uncertainty (Δx·Δp = ½):  Coherent states, Squeezed states (along squeezed axis)<br>
        Thermal / mixed states:  Δx·Δp > ½  (excess classical noise above vacuum level)
        </div>""", unsafe_allow_html=True)
 
    with tab5:
        df_mq = df.dropna(subset=["Mandel Q"])
        colors_mq = ["#f472b6" if v < 0 else ("#fbbf24" if v > 0 else "#4ade80")
                     for v in df_mq["Mandel Q"]]
        fig = go.Figure(go.Bar(
            x=df_mq["State"], y=df_mq["Mandel Q"],
            marker=dict(color=colors_mq, line=dict(color="rgba(0,0,0,0)")),
            text=[f"{v:.4f}" for v in df_mq["Mandel Q"]],
            textposition="outside", textfont=dict(size=8),
        ))
        fig.add_hline(y=0, line_color="#475569", line_width=1.5)
        fig.add_hrect(y0=-10, y1=0, fillcolor="rgba(244,114,182,0.05)",
                      line=dict(color="rgba(0,0,0,0)"))
        fig.update_layout(**_LAY, height=460,
            title=dict(text="<b>Mandel Q Parameter  (< 0 = Sub-Poissonian / quantum)</b>",
                       font=dict(size=13, color="#38bdf8"), x=0.5),
            xaxis_tickangle=-38, xaxis_title="", yaxis_title="Mandel Q",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="qcard">
          <div class="qcard-title">💡 Mandel Q Parameter</div>
          <p>Q = (⟨n²⟩ − ⟨n⟩² − ⟨n⟩)/⟨n⟩.
          <b style='color:#f472b6'>Q &lt; 0</b> = sub-Poissonian (fewer fluctuations than a laser — quantum signature).
          <b style='color:#4ade80'>Q = 0</b> = Poissonian (coherent state).
          <b style='color:#fbbf24'>Q &gt; 0</b> = super-Poissonian (thermal bunching).
          Fock states have the most negative Q, approaching −1 for large n.</p>
        </div>""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHANNEL SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
def page_channel_simulator():
    if not _check("channels"): return
    CD   = DATA["channels"]
    xvec = CD["xvec"]
 
    st.markdown("""
    <div class="hero-banner">
      <h1>⚡ Quantum Channel Simulator</h1>
      <div class="subtitle">
        Watch a coherent state |α=2⟩ evolve through four quantum channels.<br>
        All Wigner functions pre-computed via the full Lindblad master equation (QuTiP).
      </div>
      <span class="pill">Displacement D(α)</span>
      <span class="pill cyan">Squeezing S(r)</span>
      <span class="pill pink">Phase Shift R(φ)</span>
      <span class="pill amber">Loss Channel (Lindblad)</span>
    </div>
    """, unsafe_allow_html=True)
 
    channel = st.radio(
        "**Select Quantum Channel**",
        ["💫  Displacement  D(α)", "🔧  Squeezing  S(r)",
         "🔄  Phase Shift  R(φ)", "📉  Loss Channel (Lindblad)"],
        horizontal=True,
    )
 
    if "Displacement" in channel:
        series = CD["displacement_sweep"]
        param  = [s["alpha_re"] for s in series]
        plabel = "Displacement  α (real part)"
        title_fn = lambda v: f"D(α={v:.1f}) — Displaced Coherent State"
        theory_eq  = "D̂(α) = exp(αâ† − α*â)  →  shifts W(x,p) rigidly"
        theory_txt = "Displacement moves the state rigidly in phase space without distortion. Purity is perfectly preserved — a pure state stays pure. Used in CV teleportation as a correction operation."
    elif "Squeezing" in channel:
        series = CD["squeezing_sweep"]
        param  = [s["r"] for s in series]
        plabel = "Squeezing parameter  r"
        title_fn = lambda v: f"S(r={v:.1f}) — Squeezed Coherent State"
        theory_eq  = "Ŝ(r) = exp(r(â² − â†²)/2)  →  elliptical Wigner"
        theory_txt = "Squeezing stretches the Wigner function along one quadrature and compresses along the other. Purity and area are conserved. Below-shot-noise noise is the key resource in LIGO and CV-QKD."
    elif "Phase" in channel:
        series = CD["phase_sweep"]
        param  = [round(s["phi"], 2) for s in series]
        plabel = "Phase shift  φ (radians)"
        title_fn = lambda v: f"R(φ={v:.2f} rad) — Phase-Rotated Coherent State"
        theory_eq  = "R̂(φ) = exp(iφ â†â)  →  rotates W(x,p) in phase space"
        theory_txt = "Phase rotation spins the Wigner function rigidly around the origin. Purity is perfectly conserved — a unitary operation. A full 2π rotation returns the state exactly to itself."
    else:
        series = CD["loss_sweep"]
        param  = [s["gamma_t"] for s in series]
        plabel = "Loss parameter  γt"
        title_fn = lambda v: f"Loss γt={v:.1f} — Attenuated Coherent State"
        theory_eq  = "dρ/dt = γ(âρâ† − ½â†âρ − ½ρâ†â)  →  Gaussian spreading"
        theory_txt = "The loss (amplitude damping) channel models photon absorption by the environment. Every lost photon increases entropy and decreases purity. The coherent amplitude decays as e^{−γt/2} while the state acquires thermal noise."
 
    tab1, tab2 = st.tabs(["🌊  Wigner Evolution", "📊  Metrics Evolution"])
 
    with tab1:
        st.markdown(f"""<div class="eq-box"><b>Channel:</b>  {theory_eq}</div>""", unsafe_allow_html=True)
 
        # Interactive slider
        idx  = st.slider(f"**{plabel}**", 0, len(series) - 1, 0,
                         format=f"step %d / {len(series)-1}")
        s    = series[idx]
        pval = param[idx]
 
        c_main, c_info = st.columns([3, 1])
        with c_main:
            st.plotly_chart(fig_wigner(s["W"], xvec, title=title_fn(pval), height=460),
                            use_container_width=True)
        with c_info:
            m = s["metrics"]
            st.markdown(f"""
            <div style="margin-top:40px;">
            <div style="font-family:'Inter',sans-serif;font-size:0.68rem;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:12px;">
                Metrics at step {idx}</div>
            {''.join(f"""
            <div style="background:#0d1117;border:1px solid #21293d;border-left:2px solid {c};
                        border-radius:8px;padding:10px 12px;margin:6px 0;">
              <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;">{l}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.0rem;font-weight:700;color:{c};">{v}</div>
            </div>""" for l, v, c in [
                ("Purity Tr(ρ²)", f"{m.get('purity',0):.5f}", "#a78bfa"),
                ("Entropy S(ρ)", f"{m.get('entropy',0):.4f}", "#06b6d4"),
                ("⟨n̂⟩", f"{m.get('mean_n',0):.4f}", "#38bdf8"),
                ("Δx·Δp", f"{m.get('heis_prod',0):.5f}", "#fbbf24"),
            ])}
            </div>
            """, unsafe_allow_html=True)
 
        # Three snapshot thumbnails
        st.markdown("**Evolution snapshots — start / mid / end:**")
        snap_cols = st.columns(3)
        for ci, si in enumerate([0, len(series)//2, len(series)-1]):
            sv   = series[si]
            pv   = param[si]
            W_np = np.array(sv["W"], dtype=float)
            xv   = np.array(xvec, dtype=float)
            if W_np.ndim != 2: continue
            wmin, wmax = _wigner_range(W_np)
            fig = go.Figure(go.Heatmap(
                z=W_np, x=xv, y=xv, colorscale=W_CS,
                zmin=wmin, zmax=wmax, zmid=0, showscale=False,
            ))
            fig.add_trace(go.Contour(z=W_np, x=xv, y=xv,
                contours=dict(start=0,end=0,size=1,coloring="none"),
                line=dict(color="rgba(255,255,255,0.5)",width=1), showscale=False, hoverinfo="skip"))
            fig.update_layout(
                paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
                xaxis=dict(gridcolor="#161b27", tickfont=dict(size=7), scaleanchor="y"),
                yaxis=dict(gridcolor="#161b27", tickfont=dict(size=7)),
                title=dict(text=f"<b>{title_fn(pv)}</b>", font=dict(size=9,color="#38bdf8"), x=0.5),
                margin=dict(l=24,r=8,t=34,b=20), height=230,
            )
            snap_cols[ci].plotly_chart(fig, use_container_width=True)
 
        st.markdown(f"""
        <div class="qcard">
          <div class="qcard-title">💡 Physical Intuition</div>
          <p>{theory_txt}</p>
        </div>""", unsafe_allow_html=True)
 
    with tab2:
        purities  = [s["metrics"]["purity"]  for s in series]
        entropies = [s["metrics"]["entropy"]  for s in series]
        mean_ns   = [s["metrics"]["mean_n"]   for s in series]
        delta_xs  = [s["metrics"]["delta_x"]  for s in series]
        delta_ps  = [s["metrics"]["delta_p"]  for s in series]
 
        fig = make_subplots(2, 2,
            subplot_titles=["Purity Tr(ρ²)", "von Neumann Entropy S(ρ)",
                            "⟨n̂⟩ Mean Photon Number", "Δx and Δp Quadrature Noise"],
            horizontal_spacing=0.12, vertical_spacing=0.16)
 
        for (row, col), (ys, color, name) in zip(
            [(1,1),(1,2),(2,1)],
            [(purities,"#a78bfa","Purity"),(entropies,"#06b6d4","Entropy"),(mean_ns,"#38bdf8","⟨n⟩")],
        ):
            fig.add_trace(go.Scatter(x=param, y=ys, mode="lines+markers",
                line=dict(color=color, width=2.5), marker=dict(size=5, color=color),
                name=name), row=row, col=col)
 
        fig.add_trace(go.Scatter(x=param, y=delta_xs, mode="lines+markers", name="Δx",
            line=dict(color="#ec4899", width=2.5), marker=dict(size=5)), row=2, col=2)
        fig.add_trace(go.Scatter(x=param, y=delta_ps, mode="lines+markers", name="Δp",
            line=dict(color="#fbbf24", width=2.5, dash="dot"), marker=dict(size=5)), row=2, col=2)
 
        fig.update_layout(**_LAY, height=520,
            title=dict(text=f"<b>Quantum Metrics vs {plabel}</b>",
                       font=dict(size=13,color="#38bdf8"), x=0.5))
        st.plotly_chart(fig, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — GBS & QUANTUM ML
# ══════════════════════════════════════════════════════════════════════════════
def page_gbs_sampler():
    if not _check("gbs"): return
    GD = DATA["gbs"]
 
    st.markdown("""
    <div class="hero-banner">
      <h1>🔭 Gaussian Boson Sampling & CV-QML</h1>
      <div class="subtitle">
        Xanadu-grade GBS circuits · Hafnian complexity · Photon statistics · CV quantum machine learning.<br>
        Real Strawberry Fields simulation data for SF configs — exact Gaussian computation for analytic configs.
      </div>
      <span class="pill">Strawberry Fields</span>
      <span class="pill cyan">Hafnian (#P-hard)</span>
      <span class="pill pink">Photon Statistics</span>
      <span class="pill lime">CV-QML</span>
      <span class="pill amber">Quantum Advantage</span>
    </div>
    """, unsafe_allow_html=True)
 
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔭  GBS Circuit", "📊  Photon Statistics", "∑  Hafnian", "🤖  CV-QML"
    ])
 
    avail = {k: v for k, v in GD.items() if isinstance(v, dict) and "mean_photons" in v}
    mode_options = list(avail.keys())
 
    with tab1:
        if mode_options:
            c1, c2 = st.columns([2, 3])
            with c1:
                sel_key = st.selectbox(
                    "**GBS Configuration**", mode_options,
                    format_func=lambda k: f"{'🍓 SF' if 'sf' in k else '📐 Analytic'} — {GD[k]['N_modes']} modes  r≈{GD[k]['r_vals'][0]:.2f}",
                )
            gbs = GD[sel_key]
            is_sf = gbs.get("source") == "strawberryfields"
            with c2:
                src_color = "#4ade80" if is_sf else "#38bdf8"
                src_label = "🍓 Strawberry Fields — Real Gaussian backend simulation" if is_sf else "📐 Analytic — Exact Gaussian state computation (NumPy)"
                st.markdown(f"""
                <div style="background:rgba({('74,222,128' if is_sf else '56,189,248')},0.08);
                            border:1px solid rgba({('74,222,128' if is_sf else '56,189,248')},0.3);
                            border-radius:10px;padding:10px 16px;margin-top:8px;">
                  <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:{src_color};">
                    {src_label}
                  </span>
                </div>""", unsafe_allow_html=True)
 
            N = gbs["N_modes"]
 
            # Circuit diagram
            fig_circ = go.Figure()
            for i in range(N):
                fig_circ.add_shape(type="line", x0=0, x1=5.2, y0=i, y1=i,
                                   line=dict(color="#1e293b", width=2.5))
                fig_circ.add_annotation(x=-0.25, y=i, text="|0⟩", showarrow=False,
                                        font=dict(color="#38bdf8", size=12, family="JetBrains Mono"))
                r_i = gbs["r_vals"][i] if i < len(gbs["r_vals"]) else gbs["r_vals"][0]
                # Squeezing gate
                fig_circ.add_shape(type="rect", x0=0.2, x1=1.1, y0=i-0.3, y1=i+0.3,
                    line=dict(color="#4f46e5", width=2), fillcolor="rgba(79,70,229,0.25)")
                fig_circ.add_annotation(x=0.65, y=i, text=f"Ŝ({r_i:.1f})", showarrow=False,
                    font=dict(color="#c4b5fd", size=9, family="JetBrains Mono"))
                # Output annotation
                fig_circ.add_annotation(x=5.5, y=i, text=f"n̄={gbs['mean_photons'][i]:.2f}",
                    showarrow=False, font=dict(color="#fbbf24", size=10, family="JetBrains Mono"))
                # Detector
                fig_circ.add_shape(type="rect", x0=4.2, x1=5.0, y0=i-0.28, y1=i+0.28,
                    line=dict(color="#ec4899", width=2), fillcolor="rgba(236,72,153,0.12)")
                fig_circ.add_annotation(x=4.6, y=i, text="📷", showarrow=False,
                    font=dict(color="#f9a8d4", size=10))
 
            # Interferometer box
            fig_circ.add_shape(type="rect", x0=1.3, x1=4.0, y0=-0.55, y1=N-0.45,
                line=dict(color="#06b6d4", width=2), fillcolor="rgba(6,182,212,0.06)")
            fig_circ.add_annotation(x=2.65, y=(N-1)/2, text="Interferometer  Û",
                showarrow=False, font=dict(color="#06b6d4", size=13, family="JetBrains Mono"))
            if N > 2:
                fig_circ.add_annotation(x=2.65, y=(N-1)/2 - 0.55,
                    text="(Clements decomposition)", showarrow=False,
                    font=dict(color="#334155", size=8, family="JetBrains Mono"))
 
            fig_circ.update_layout(
                paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
                xaxis=dict(visible=False, range=[-0.6, 6.2]),
                yaxis=dict(visible=False, range=[-0.9, N + 0.2]),
                height=max(260, N * 60 + 60),
                title=dict(text=f"<b>GBS Circuit — {N} Modes</b>",
                           font=dict(size=13, color="#38bdf8"), x=0.5),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_circ, use_container_width=True)
 
            # Wigner functions per mode
            st.markdown("**Marginal Wigner functions per mode (after interferometer):**")
            xvec_sf = np.array(gbs["xvec"])
            n_show  = min(N, 4)
            w_cols  = st.columns(n_show)
            wig_keys = list(gbs["wigners"].keys())
            for i in range(n_show):
                wk   = i if isinstance(wig_keys[0], int) else str(i)
                W_i  = np.array(gbs["wigners"][wk], dtype=float)
                wmin, wmax = _wigner_range(W_i)
                fig_wi = go.Figure(go.Heatmap(
                    z=W_i, x=xvec_sf, y=xvec_sf, colorscale=W_CS,
                    zmin=wmin, zmax=wmax, zmid=0, showscale=False,
                ))
                fig_wi.update_layout(
                    paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
                    xaxis=dict(gridcolor="#161b27", tickfont=dict(size=7), scaleanchor="y"),
                    yaxis=dict(gridcolor="#161b27", tickfont=dict(size=7)),
                    title=dict(text=f"<b>Mode {i}  n̄={gbs['mean_photons'][i]:.2f}</b>",
                               font=dict(size=10, color="#22d3ee"), x=0.5),
                    margin=dict(l=20, r=6, t=36, b=18), height=210,
                )
                w_cols[i].plotly_chart(fig_wi, use_container_width=True)
 
            st.markdown("""<div class="qcard">
              <div class="qcard-title">💡 GBS Architecture</div>
              <p>GBS = N squeezed vacuum inputs → random linear-optical interferometer → photon-number detection.
              The output probability P(S) ∝ |Haf(A_S)|² where Haf is the hafnian — a #P-hard function.
              Applications: molecular vibronic spectra, dense subgraph optimisation, drug discovery, quantum advantage demonstration.
              Xanadu's Borealis (2022) demonstrated quantum computational advantage using GBS.</p>
            </div>""", unsafe_allow_html=True)
 
    with tab2:
        if mode_options:
            sel2 = st.selectbox("GBS config", mode_options, key="gbs_tab2",
                format_func=lambda k: f"{'SF' if 'sf' in k else 'Analytic'} — {GD[k]['N_modes']} modes")
            gbs2 = GD[sel2]
            ca, cb = st.columns(2)
            with ca:
                fig_mn = go.Figure(go.Bar(
                    x=[f"Mode {i}" for i in range(len(gbs2["mean_photons"]))],
                    y=[float(v) for v in gbs2["mean_photons"]],
                    marker=dict(
                        color=list(range(len(gbs2["mean_photons"]))),
                        colorscale=[[0,"#4f46e5"],[0.5,"#06b6d4"],[1,"#ec4899"]],
                        showscale=False, line=dict(color="rgba(0,0,0,0)"),
                    ),
                    text=[f"{v:.3f}" for v in gbs2["mean_photons"]], textposition="outside",
                ))
                fig_mn.update_layout(**_LAY, height=320,
                    title=dict(text="<b>⟨n̂⟩ per mode</b>", font=dict(size=12,color="#38bdf8"), x=0.5),
                    xaxis_title="Mode", yaxis_title="⟨n̂⟩")
                st.plotly_chart(fig_mn, use_container_width=True)
 
            with cb:
                pn  = np.array(gbs2["photon_dist"]); ns = np.array(gbs2["photon_ns"])
                nbar = float(gbs2["mean_photons"][0])
                fig_pn = go.Figure(go.Bar(x=ns, y=pn,
                    marker=dict(color=pn, colorscale=[[0,"#1e1b4b"],[0.5,"#4f46e5"],[1,"#06b6d4"]],
                                showscale=False, line=dict(color="rgba(0,0,0,0)")),
                    name="P(n)"))
                if nbar > 0:
                    thermal = (nbar/(1+nbar))**ns / (1+nbar)
                    fig_pn.add_trace(go.Scatter(x=ns, y=thermal, mode="lines",
                        name="Thermal ref", line=dict(color="#fbbf24", dash="dot", width=2)))
                fig_pn.update_layout(**_LAY, height=320,
                    title=dict(text="<b>Mode 0 — P(n)</b>", font=dict(size=12,color="#ec4899"), x=0.5),
                    xaxis_title="n", yaxis_title="P(n)")
                st.plotly_chart(fig_pn, use_container_width=True)
 
            st.markdown("""<div class="qcard">
              <div class="qcard-title">💡 GBS Photon Statistics</div>
              <p>Each mode after the interferometer follows a <b>thermal-like marginal distribution</b> with n̄ = sinh²(r).
              However the <b>joint</b> multi-mode distribution is governed by the hafnian and is
              exponentially hard to sample classically — this is the source of quantum advantage.</p>
            </div>""", unsafe_allow_html=True)
 
    with tab3:
        st.markdown("### Hafnian — the #P-Hard Function at the Heart of GBS")
        st.markdown("""<div class="eq-box">
        Haf(A) = Σ_{perfect matchings M} Π_{(i,j)∈M} A_{ij}<br>
        GBS probability:   P(S) = |Haf(A_S)|² / (s₁! · s₂! · … · √det(σ_Q))<br>
        Classical complexity:  O(2ⁿ · n²)  via Ryser formula  →  intractable for n ≳ 50
        </div>""", unsafe_allow_html=True)
 
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hafnian verification table (brute-force):**")
            df_haf = pd.DataFrame(GD.get("hafnian_table", []))
            if not df_haf.empty:
                st.dataframe(df_haf, use_container_width=True, hide_index=True)
 
        with c2:
            if mode_options:
                gbs0   = GD[mode_options[0]]
                n_arr  = np.array(gbs0["hafnian_n"])
                t_ryser = np.array(gbs0["hafnian_ops"])
                fig_haf = go.Figure(go.Scatter(
                    x=n_arr, y=t_ryser, mode="lines+markers",
                    line=dict(color="#4f46e5", width=2.5),
                    marker=dict(size=7, color="#ec4899", line=dict(color="white", width=1)),
                    fill="tozeroy", fillcolor="rgba(79,70,229,0.08)",
                    name="Ryser  O(2ⁿ · n²)",
                ))
                fig_haf.update_layout(**_LAY, height=340, yaxis_type="log",
                    title=dict(text="<b>Hafnian Classical Complexity  O(2ⁿ · n²)</b>",
                               font=dict(size=12, color="#38bdf8"), x=0.5),
                    xaxis_title="Matrix size n", yaxis_title="Operations (log scale)")
                st.plotly_chart(fig_haf, use_container_width=True)
 
    with tab4:
        st.markdown("### Continuous-Variable Quantum Machine Learning")
        st.markdown("""<div class="eq-box">
        CV QNode:   Û(θ) = D̂(α) · Ŝ(r) · R̂(φ)   acting on |0⟩<br>
        Expectation:   ⟨X̂⟩(θ) = √2 · r · cos(θ)<br>
        Parameter-shift rule:   ∂⟨Ô⟩/∂θ = ½[⟨Ô⟩_{θ+π/2} − ⟨Ô⟩_{θ−π/2}]
        </div>""", unsafe_allow_html=True)
 
        theta  = np.array(GD["qml_theta"])
        exp_x  = np.array(GD["qml_exp_x"])
        grad   = np.array(GD["qml_gradient"])
        steps  = np.array(GD["qml_steps"])
        loss   = np.array(GD["qml_loss"])
 
        fig_qml = make_subplots(1, 2,
            subplot_titles=["⟨X̂⟩(θ) — Expectation value", "∂⟨X̂⟩/∂θ — Parameter-shift gradient"],
            horizontal_spacing=0.12)
        fig_qml.add_trace(go.Scatter(x=theta, y=exp_x, mode="lines",
            line=dict(color="#4f46e5", width=2.5), fill="tozeroy",
            fillcolor="rgba(79,70,229,0.08)", name="⟨X̂⟩"), row=1, col=1)
        fig_qml.add_trace(go.Scatter(x=theta, y=grad, mode="lines",
            line=dict(color="#06b6d4", width=2.5), name="gradient"), row=1, col=2)
        fig_qml.add_hline(y=0, line_color="#1e293b", line_width=1, row=1, col=1)
        fig_qml.add_hline(y=0, line_color="#1e293b", line_width=1, row=1, col=2)
        fig_qml.update_layout(**_LAY, height=340,
            title=dict(text="<b>CV QNode Parameter Landscape</b>",
                       font=dict(size=13,color="#38bdf8"), x=0.5))
        st.plotly_chart(fig_qml, use_container_width=True)
 
        fig_tr = go.Figure(go.Scatter(
            x=steps, y=loss, mode="lines",
            line=dict(color="#ec4899", width=2.5),
            fill="tozeroy", fillcolor="rgba(236,72,153,0.08)",
        ))
        fig_tr.add_trace(go.Scatter(
            x=steps, y=np.convolve(loss, np.ones(10)/10, mode="same"),
            mode="lines", line=dict(color="#fbbf24", width=1.5, dash="dot"),
            name="10-step moving avg",
        ))
        fig_tr.update_layout(**_LAY, height=300, yaxis_type="log",
            title=dict(text="<b>CV-QNN Training Loss — Adam Optimizer</b>",
                       font=dict(size=12,color="#ec4899"), x=0.5),
            xaxis_title="Training step", yaxis_title="MSE Loss  (log scale)")
        st.plotly_chart(fig_tr, use_container_width=True)
 
        st.markdown("""<div class="qcard">
          <div class="qcard-title">💡 CV Quantum Machine Learning</div>
          <p>CV quantum neural networks use <b>Gaussian gates</b> (displacement, squeezing, phase rotation)
          and <b>non-Gaussian gates</b> (Kerr interaction) as trainable layers.
          The parameter-shift rule provides exact quantum gradients directly on hardware — no finite-difference approximation needed.
          CV-QML is naturally suited for regression, generative modelling, and quantum chemistry.
          Xanadu's PennyLane framework supports end-to-end CV differentiable programming.</p>
        </div>""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    page = render_sidebar()
    if   "State Explorer"    in page: page_state_explorer()
    elif "Phase Space Zoo"   in page: page_phase_space_zoo()
    elif "Witness Lab"       in page: page_witness_lab()
    elif "Channel Simulator" in page: page_channel_simulator()
    elif "GBS"               in page: page_gbs_sampler()
 
if __name__ == "__main__":
    main()
