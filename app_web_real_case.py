"""
KNN Real-Case Web Demo — Enhanced Edition
Features: multiple real datasets, rich stats, confusion matrix, k-sensitivity,
responsive layout, dark glassmorphism UI.
Authors: Ouahid Samrani & Yassir Mrabti
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def hex_to_rgba(hex_color, alpha=1.0):
    """Convert #RRGGBB to rgba(r,g,b,a) string."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config(page_title="KNN Real Case Studio", page_icon="🩺", layout="wide")

# ════════════════════════════════════════════════════════════
#  PREMIUM DARK THEME + GLASSMORPHISM
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg-primary:   #0a0e1a;
  --bg-secondary: #111827;
  --bg-card:      rgba(17, 24, 39, 0.65);
  --glass-border: rgba(99, 179, 237, 0.18);
  --accent-blue:  #63b3ed;
  --accent-cyan:  #22d3ee;
  --accent-orange:#fb923c;
  --accent-purple:#a78bfa;
  --text-primary: #e2e8f0;
  --text-muted:   #94a3b8;
  --text-bright:  #f1f5f9;
  --glow-blue:    0 0 25px rgba(99,179,237,0.25);
}

@keyframes fadeInUp { from {opacity:0;transform:translateY(20px)} to {opacity:1;transform:translateY(0)} }
@keyframes scaleIn  { from {opacity:0;transform:scale(0.9)} to {opacity:1;transform:scale(1)} }
@keyframes shimmer  { 0%{background-position:-200% center} 100%{background-position:200% center} }
@keyframes glowPulse {
  0%,100% { box-shadow: 0 0 12px rgba(99,179,237,0.12); }
  50%     { box-shadow: 0 0 22px rgba(99,179,237,0.25); }
}

/* Global */
.stApp, .main, [data-testid="stAppViewContainer"] {
  background: var(--bg-primary) !important;
  background-image:
    radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99,179,237,0.06) 0%, transparent 55%),
    radial-gradient(ellipse 60% 50% at 85% 75%, rgba(167,139,250,0.05) 0%, transparent 50%) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text-primary) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1rem !important; max-width: 1440px !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1321 0%, #111b2e 50%, #0d1321 100%) !important;
  border-right: 1px solid rgba(99,179,237,0.12) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
  background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;
}

/* Headings */
h1 { font-family:'Inter',sans-serif!important; font-weight:900!important; font-size:2.2rem!important;
     background: linear-gradient(135deg, #63b3ed, #22d3ee, #a78bfa);
     background-size:200% auto; -webkit-background-clip:text!important;
     -webkit-text-fill-color:transparent!important;
     animation: shimmer 4s linear infinite, fadeInUp 0.7s ease-out;
     letter-spacing:-.4px!important; }
h2,h3 { color:var(--text-bright)!important; font-family:'Inter',sans-serif!important; font-weight:700!important; }
[data-testid="stCaptionContainer"] p { color:var(--text-muted)!important; }
[data-testid="stMarkdownContainer"] p { color:var(--text-primary)!important; }
hr { border-color: rgba(99,179,237,0.12) !important; }

/* Hero */
.hero { border:1px solid var(--glass-border); border-radius:16px; padding:1rem 1.4rem;
        background: var(--bg-card); backdrop-filter:blur(16px);
        animation:fadeInUp .6s ease-out; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; top:0;left:0;right:0; height:2px;
  background:linear-gradient(90deg,transparent,var(--accent-cyan),var(--accent-blue),transparent); opacity:.7; }
.hero h1 { margin:.1rem 0 .3rem 0; font-size:1.8rem; }
.hero p { color:var(--text-muted)!important; }
.badge { display:inline-block; padding:.2rem .6rem; border-radius:999px; font-size:.72rem;
         font-weight:600; border:1px solid rgba(34,211,238,0.3); color:var(--accent-cyan);
         background:rgba(34,211,238,0.08); }

/* KPI tiles */
.kpi { border:1px solid var(--glass-border); border-radius:12px; padding:.6rem .7rem;
       background:var(--bg-card); backdrop-filter:blur(12px); text-align:center;
       animation:scaleIn .6s ease-out; transition:all .3s ease; }
.kpi:hover { transform:translateY(-2px); border-color:rgba(99,179,237,0.35); }
.kpi .label { font-size:.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:.6px; }
.kpi .value { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.15rem; margin-top:.1rem;
              color:var(--accent-cyan); }

/* Cards */
.card { border:1px solid var(--glass-border); border-radius:14px; padding:.8rem 1rem;
        background:var(--bg-card); backdrop-filter:blur(16px);
        animation:fadeInUp .7s ease-out, glowPulse 4s ease-in-out infinite;
        transition:transform .25s,box-shadow .25s; }
.card:hover { transform:translateY(-2px); box-shadow:0 8px 30px rgba(99,179,237,.12); }
.card p { color:var(--text-primary)!important; margin:.3rem 0; font-size:.92rem; }
.card b { color:var(--accent-cyan); }

.section-title { font-family:'Inter',sans-serif; font-weight:700; font-size:1rem;
                 margin:.6rem 0 .4rem 0; color:var(--text-bright); }

[data-testid="stDataFrame"] { border-radius:10px; border:1px solid rgba(99,179,237,0.12); }

/* Footer */
.gradient-divider { height:1px; margin:1.5rem 0; opacity:.4;
  background:linear-gradient(90deg,transparent,var(--accent-blue),var(--accent-cyan),var(--accent-blue),transparent); }
.footer-text { text-align:center; font-size:.78rem; color:var(--text-muted); padding:.6rem 0 1.2rem 0; }
.footer-text .tech-tag { background:rgba(99,179,237,0.08); border:1px solid rgba(99,179,237,0.15);
  border-radius:6px; padding:2px 8px; font-size:.72rem; font-family:'JetBrains Mono',monospace;
  color:var(--text-muted); margin:0 3px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  DATASETS — 3 real-world inspired scenarios
# ════════════════════════════════════════════════════════════
DATASETS = {
    "🩺 Triage Métabolique (IMC + Glycémie)": {
        "desc": "Classer un patient en risque faible ou risque élevé de syndrome métabolique à partir de son IMC et sa glycémie à jeun.",
        "features": ("IMC (kg/m²)", "Glycémie (mg/dL)"),
        "classes": ("Risque faible", "Risque élevé"),
        "colors": ("#3b82f6", "#f97316"),
        "class_0": np.array([
            [20.2,82],[21.5,88],[22.1,91],[23.0,95],[24.2,99],
            [25.1,101],[21.0,86],[22.8,93],[24.7,97],[23.5,90],
            [19.8,84],[20.7,87],[22.5,94],[24.0,98],[23.8,92],
        ]),
        "class_1": np.array([
            [28.4,118],[29.6,124],[30.8,130],[31.5,135],[33.0,140],
            [27.9,116],[32.2,138],[34.0,145],[29.1,121],[31.0,133],
            [28.0,119],[30.2,128],[33.5,142],[27.5,114],[29.8,126],
        ]),
        "ranges": {"x": (18.0, 36.0), "y": (75.0, 150.0)},
        "default_point": (26.5, 108.0),
    },
    "🌸 Classification Iris (Pétale)": {
        "desc": "Distinguer Iris setosa de Iris versicolor par la longueur et la largeur du pétale — le dataset classique du machine learning.",
        "features": ("Longueur pétale (cm)", "Largeur pétale (cm)"),
        "classes": ("Setosa", "Versicolor"),
        "colors": ("#a78bfa", "#22d3ee"),
        "class_0": np.array([
            [1.4,0.2],[1.3,0.2],[1.5,0.2],[1.4,0.1],[1.7,0.4],
            [1.0,0.2],[1.5,0.1],[1.6,0.2],[1.4,0.3],[1.1,0.1],
            [1.2,0.2],[1.5,0.4],[1.3,0.3],[1.4,0.2],[1.6,0.2],
        ]),
        "class_1": np.array([
            [4.7,1.4],[4.5,1.5],[3.3,1.0],[4.0,1.3],[3.9,1.4],
            [4.2,1.5],[4.4,1.4],[4.1,1.0],[3.5,1.0],[4.5,1.5],
            [3.8,1.1],[4.0,1.2],[4.6,1.5],[3.7,1.0],[4.3,1.3],
        ]),
        "ranges": {"x": (0.5, 5.5), "y": (0.0, 2.0)},
        "default_point": (2.5, 0.8),
    },
    "🏠 Immobilier (Surface + Prix)": {
        "desc": "Classifier un bien immobilier comme Économique ou Premium selon sa surface et son prix au m².",
        "features": ("Surface (m²)", "Prix/m² (€)"),
        "classes": ("Économique", "Premium"),
        "colors": ("#22d3ee", "#fb923c"),
        "class_0": np.array([
            [35,2200],[42,2400],[50,2600],[38,2100],[55,2800],
            [45,2500],[30,2000],[48,2350],[52,2700],[40,2300],
            [33,2150],[47,2450],[36,2250],[44,2550],[39,2050],
        ]),
        "class_1": np.array([
            [85,5200],[92,5500],[78,4800],[105,6000],[95,5800],
            [88,5100],[110,6200],[82,4900],[98,5600],[75,4600],
            [90,5300],[100,5900],[83,5050],[96,5700],[87,5400],
        ]),
        "ranges": {"x": (25.0, 120.0), "y": (1800.0, 6500.0)},
        "default_point": (60.0, 3500.0),
    },
}

# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Dataset")
    dataset_name = st.selectbox("Choisir un cas réel", list(DATASETS.keys()), label_visibility="collapsed")
    ds = DATASETS[dataset_name]

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres KNN")
    k_user = st.slider("Nombre de voisins (k)", 1, 15, 5, step=2)
    metric = st.selectbox("Métrique de distance", ["euclidean", "manhattan", "minkowski"])
    weights = st.radio("Pondération", ["uniform", "distance"], horizontal=True)

    st.markdown("---")
    st.markdown(f"### 📍 Nouveau Point")
    feat_x, feat_y = ds["features"]
    rx, ry = ds["ranges"]["x"], ds["ranges"]["y"]
    dx, dy = ds["default_point"]
    x_pt = st.slider(feat_x, float(rx[0]), float(rx[1]), float(dx), step=round((rx[1]-rx[0])/200, 2))
    y_pt = st.slider(feat_y, float(ry[0]), float(ry[1]), float(dy), step=round((ry[1]-ry[0])/200, 2))

    st.markdown("---")
    st.markdown("### 🗺️ Affichage")
    show_boundary = st.toggle("Frontière de décision", True)
    show_confidence = st.toggle("Carte de confiance", True)
    show_connections = st.toggle("Lignes voisins", True)

    st.markdown("---")
    st.caption("🧠 KNN Real Case Studio v3.0")

# ════════════════════════════════════════════════════════════
#  DATA + MODEL (normalize for proper circle + distances)
# ════════════════════════════════════════════════════════════
c0, c1 = ds["class_0"], ds["class_1"]
X_raw = np.vstack([c0, c1])
y = np.array([0]*len(c0) + [1]*len(c1))

# Normalize data so KNN distances are scale-independent → circle stays round
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_raw)
new_point_raw = np.array([[x_pt, y_pt]])
new_point_norm = scaler.transform(new_point_raw)

k_safe = int(max(1, min(k_user, len(X_raw))))
knn = KNeighborsClassifier(n_neighbors=k_safe, metric=metric, weights=weights)
knn.fit(X_norm, y)
pred = int(knn.predict(new_point_norm)[0])
proba = knn.predict_proba(new_point_norm)[0]
conf = float(proba[pred]) * 100

dist_arr, idx_arr = knn.kneighbors(new_point_norm, n_neighbors=k_safe)
rayon_norm = float(dist_arr[0][-1])  # radius in normalized space
voisins_raw = X_raw[idx_arr[0]]
voisins_cls = y[idx_arr[0]]
votes_0 = int(np.sum(voisins_cls == 0))
votes_1 = int(np.sum(voisins_cls == 1))

# Cross-val accuracy
cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=k_safe, metric=metric, weights=weights),
    X_norm, y, cv=min(5, len(X_raw)), scoring="accuracy"
)

# Baseline
mid_y = (ry[0] + ry[1]) / 2
baseline_pred_arr = (X_raw[:, 1] >= mid_y).astype(int)
baseline_acc = accuracy_score(y, baseline_pred_arr)

# Confusion matrix on training data
train_pred = knn.predict(X_norm)
cm = confusion_matrix(y, train_pred)

# K sensitivity
k_vals = [k for k in range(1, 16, 2)]
k_preds, k_confs, k_accs = [], [], []
for kv in k_vals:
    km = KNeighborsClassifier(n_neighbors=kv, metric=metric, weights=weights)
    km.fit(X_norm, y)
    kp = int(km.predict(new_point_norm)[0])
    kpr = km.predict_proba(new_point_norm)[0]
    k_preds.append(kp)
    k_confs.append(float(kpr[kp]) * 100)
    k_accs.append(accuracy_score(y, km.predict(X_norm)) * 100)

cls_names = ds["classes"]
pred_label = cls_names[pred]
col0, col1 = ds["colors"]

# ════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <span class="badge">CAS RÉEL INTERACTIF</span>
  <h1>{dataset_name}</h1>
  <p>{ds["desc"]}</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  KPI ROW
# ════════════════════════════════════════════════════════════
st.markdown("")  # spacing
c1k, c2k, c3k, c4k, c5k = st.columns(5)
with c1k:
    st.markdown(f"<div class='kpi'><div class='label'>Prédiction</div><div class='value' style='color:{col0 if pred==0 else col1}'>{pred_label}</div></div>", unsafe_allow_html=True)
with c2k:
    st.markdown(f"<div class='kpi'><div class='label'>Confiance</div><div class='value'>{conf:.1f}%</div></div>", unsafe_allow_html=True)
with c3k:
    st.markdown(f"<div class='kpi'><div class='label'>Votes {cls_names[0]}</div><div class='value' style='color:{col0}'>{votes_0}/{k_safe}</div></div>", unsafe_allow_html=True)
with c4k:
    st.markdown(f"<div class='kpi'><div class='label'>Votes {cls_names[1]}</div><div class='value' style='color:{col1}'>{votes_1}/{k_safe}</div></div>", unsafe_allow_html=True)
with c5k:
    st.markdown(f"<div class='kpi'><div class='label'>CV Accuracy</div><div class='value'>{cv_scores.mean()*100:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("")  # spacing

# ════════════════════════════════════════════════════════════
#  MAIN CHART (normalized space → perfect circle) + INFO
# ════════════════════════════════════════════════════════════
left_col, right_col = st.columns([2.5, 1.0], gap="large")

with left_col:
    fig = go.Figure()

    # Grid in normalized space, then inverse-transform for display in real units
    gx_norm = np.linspace(X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1, 100)
    gy_norm = np.linspace(X_norm[:, 1].min() - 1, X_norm[:, 1].max() + 1, 100)
    xx_n, yy_n = np.meshgrid(gx_norm, gy_norm)
    grid_norm = np.c_[xx_n.ravel(), yy_n.ravel()]
    # Convert grid back to real coords for display
    grid_real = scaler.inverse_transform(grid_norm)
    gx_real = scaler.inverse_transform(np.c_[gx_norm, np.zeros_like(gx_norm)])[:, 0]
    gy_real = scaler.inverse_transform(np.c_[np.zeros_like(gy_norm), gy_norm])[:, 1]

    if show_boundary:
        zz = knn.predict(grid_norm).reshape(xx_n.shape)
        fig.add_trace(go.Contour(
            x=gx_real, y=gy_real, z=zz, showscale=False, opacity=0.25,
            colorscale=[[0, hex_to_rgba(col0, 0.25)], [0.499, hex_to_rgba(col0, 0.25)],
                        [0.5, hex_to_rgba(col1, 0.25)], [1, hex_to_rgba(col1, 0.25)]],
            contours=dict(showlines=False), hoverinfo="skip", name="Frontière",
        ))

    if show_confidence:
        conf_grid = knn.predict_proba(grid_norm)[:, 1].reshape(xx_n.shape)
        fig.add_trace(go.Contour(
            x=gx_real, y=gy_real, z=conf_grid, showscale=False, opacity=0.15,
            colorscale=[[0,"rgba(59,130,246,0.3)"],[0.5,"rgba(200,200,200,0.03)"],[1,"rgba(251,146,60,0.3)"]],
            contours=dict(start=0, end=1, size=0.1, coloring="heatmap", showlines=False),
            hoverinfo="skip", name="Confiance",
        ))

    # Data points (real coordinates)
    fig.add_trace(go.Scatter(x=c0[:,0], y=c0[:,1], mode="markers", name=cls_names[0],
        marker=dict(size=12, color=col0, line=dict(color="rgba(255,255,255,0.5)", width=1.5)),
        hovertemplate=f"{cls_names[0]}<br>({feat_x}: %{{x:.1f}}, {feat_y}: %{{y:.1f}})<extra></extra>"))
    fig.add_trace(go.Scatter(x=c1[:,0], y=c1[:,1], mode="markers", name=cls_names[1],
        marker=dict(size=12, color=col1, symbol="diamond", line=dict(color="rgba(255,255,255,0.5)", width=1.5)),
        hovertemplate=f"{cls_names[1]}<br>({feat_x}: %{{x:.1f}}, {feat_y}: %{{y:.1f}})<extra></extra>"))

    # Neighbor highlights
    fig.add_trace(go.Scatter(x=voisins_raw[:,0], y=voisins_raw[:,1], mode="markers", name=f"{k_safe} voisins",
        marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color="#22d3ee", width=2.5))))

    # New point
    fig.add_trace(go.Scatter(x=[x_pt], y=[y_pt], mode="markers+text", name="Nouveau point",
        text=["?"], textposition="middle center", textfont=dict(size=16, color="white", family="Inter"),
        marker=dict(size=28, color="#0891b2", line=dict(color="#164e63", width=2))))

    # Search radius — draw circle in NORMALIZED space, convert to real coords for display
    if show_connections:
        t = np.linspace(0, 2*np.pi, 200)
        circle_norm_x = new_point_norm[0, 0] + rayon_norm * np.cos(t)
        circle_norm_y = new_point_norm[0, 1] + rayon_norm * np.sin(t)
        circle_real = scaler.inverse_transform(np.c_[circle_norm_x, circle_norm_y])
        fig.add_trace(go.Scatter(
            x=circle_real[:, 0], y=circle_real[:, 1],
            mode="lines", name="Zone de recherche",
            line=dict(color="rgba(34,211,238,0.5)", width=1.8, dash="dash")))

        # Lines to neighbors
        for vc in voisins_raw:
            fig.add_trace(go.Scatter(x=[x_pt, vc[0]], y=[y_pt, vc[1]], mode="lines", showlegend=False,
                line=dict(color="rgba(34,211,238,0.18)", width=1, dash="dot"), hoverinfo="skip"))

    fig.update_layout(
        title=dict(text=f"<b>KNN Classification</b> — K={k_safe}, {metric}, {weights}",
                   font=dict(size=15, color="#e2e8f0", family="Inter"), x=0.01, y=0.97),
        height=680, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1729",
        margin=dict(l=50, r=20, t=45, b=45),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0,
                    font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#94a3b8", family="Inter"),
    )
    fig.update_xaxes(title=dict(text=feat_x, font=dict(size=12, color="#94a3b8")),
        showgrid=True, gridcolor="rgba(99,179,237,0.06)", gridwidth=1,
        tickfont=dict(color="#64748b"), zeroline=False, linecolor="rgba(99,179,237,0.15)",
        range=[rx[0], rx[1]])
    fig.update_yaxes(title=dict(text=feat_y, font=dict(size=12, color="#94a3b8")),
        showgrid=True, gridcolor="rgba(99,179,237,0.06)", gridwidth=1,
        tickfont=dict(color="#64748b"), zeroline=False, linecolor="rgba(99,179,237,0.15)",
        range=[ry[0], ry[1]])
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown("<div class='section-title'>📋 Voisins utilisés</div>", unsafe_allow_html=True)
    rows = []
    for rank, (coord, d, cls) in enumerate(zip(voisins_raw, dist_arr[0], voisins_cls), 1):
        rows.append({"#": rank, feat_x[:8]: round(coord[0],1), feat_y[:8]: round(coord[1],1),
                      "Dist": round(float(d),2), "Classe": cls_names[int(cls)]})
    st.dataframe(rows, hide_index=True, use_container_width=True)

    st.markdown("<div class='section-title'>🗳️ Répartition des votes</div>", unsafe_allow_html=True)
    vote_fig = go.Figure(data=[go.Pie(
        labels=[cls_names[0], cls_names[1]], values=[votes_0, votes_1], hole=0.55,
        marker=dict(colors=[col0, col1], line=dict(color="#0f172a", width=2)),
        textinfo="percent+label", textfont=dict(size=10, color="#e2e8f0"),
    )])
    vote_fig.update_layout(height=200, margin=dict(l=5,r=5,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(color="#94a3b8"),
        annotations=[dict(text=f"<b>{k_safe}</b>", x=0.5, y=0.5,
        font=dict(size=20, color="#22d3ee", family="JetBrains Mono"), showarrow=False)])
    st.plotly_chart(vote_fig, use_container_width=True)

    st.markdown(f"""
    <div class="card">
      <p style="margin:0;font-size:.78rem;color:var(--text-muted)">RÉSUMÉ</p>
      <p><b>Point:</b> ({x_pt:.1f}, {y_pt:.1f})</p>
      <p><b>Rayon (normalisé):</b> {rayon_norm:.2f}</p>
      <p><b>Métrique:</b> {metric}</p>
      <p><b>Pondération:</b> {weights}</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  STATISTICS ROW
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📊 Analyse Statistique Approfondie")

stat1, stat2, stat3 = st.columns(3, gap="medium")

# Dark chart styling helper
dark_chart = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1729",
                  font=dict(color="#94a3b8", family="Inter"))
dark_grid = dict(showgrid=True, gridcolor="rgba(99,179,237,0.06)", zeroline=False)

with stat1:
    st.markdown("<div class='section-title'>🔢 Matrice de Confusion</div>", unsafe_allow_html=True)
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm, x=[cls_names[0], cls_names[1]], y=[cls_names[0], cls_names[1]],
        colorscale=[[0,"#0f1729"],[1,"#3b82f6"]], showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=18, color="#e2e8f0"),
    ))
    cm_fig.update_layout(height=280, margin=dict(l=10,r=10,t=35,b=10), **dark_chart,
        xaxis_title="Prédit", yaxis_title="Réel",
        title=dict(text=f"Accuracy: {accuracy_score(y,train_pred)*100:.1f}%",
                   font=dict(size=12, color="#e2e8f0")))
    st.plotly_chart(cm_fig, use_container_width=True)

with stat2:
    st.markdown("<div class='section-title'>📈 Sensibilité au paramètre K</div>", unsafe_allow_html=True)
    sens_fig = make_subplots(specs=[[{"secondary_y": True}]])
    sens_fig.add_trace(go.Bar(x=k_vals, y=k_confs, name="Confiance %",
        marker_color=[col0 if p==0 else col1 for p in k_preds], opacity=0.7), secondary_y=False)
    sens_fig.add_trace(go.Scatter(x=k_vals, y=k_accs, name="Accuracy %", mode="lines+markers",
        line=dict(color="#22d3ee", width=2), marker=dict(size=7, color="#22d3ee")), secondary_y=True)
    sens_fig.update_layout(height=280, margin=dict(l=10,r=10,t=35,b=10), **dark_chart,
        showlegend=True, legend=dict(font=dict(size=9, color="#94a3b8"), y=1.15, orientation="h"),
        title=dict(text="Confiance & Accuracy par K", font=dict(size=12, color="#e2e8f0")))
    sens_fig.update_xaxes(title="k", tickmode="array", tickvals=k_vals, **dark_grid)
    sens_fig.update_yaxes(title="Confiance %", range=[0,105], secondary_y=False, **dark_grid)
    sens_fig.update_yaxes(title="Accuracy %", range=[80,105], secondary_y=True, **dark_grid)
    st.plotly_chart(sens_fig, use_container_width=True)

with stat3:
    st.markdown("<div class='section-title'>⚖️ KNN vs Règle Simple</div>", unsafe_allow_html=True)
    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(x=["KNN"], y=[accuracy_score(y,train_pred)*100], marker_color=col0,
        name="KNN", text=[f"{accuracy_score(y,train_pred)*100:.1f}%"], textposition="auto",
        textfont=dict(color="#e2e8f0")))
    comp_fig.add_trace(go.Bar(x=["Règle seuil"], y=[baseline_acc*100], marker_color="#475569",
        name="Baseline", text=[f"{baseline_acc*100:.1f}%"], textposition="auto",
        textfont=dict(color="#e2e8f0")))
    comp_fig.update_layout(height=280, margin=dict(l=10,r=10,t=35,b=10), **dark_chart,
        showlegend=True, barmode="group",
        title=dict(text="Accuracy Comparée", font=dict(size=12, color="#e2e8f0")))
    st.plotly_chart(comp_fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-text">
  <div>Built with ❤️ by <b>Ouahid Samrani</b> & <b>Yassir Mrabti</b></div>
  <div style="margin-top:6px;">
    <span class="tech-tag">Streamlit</span>
    <span class="tech-tag">scikit-learn</span>
    <span class="tech-tag">Plotly</span>
    <span class="tech-tag">NumPy</span>
  </div>
</div>
""", unsafe_allow_html=True)
