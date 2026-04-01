import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="KNN Studio", page_icon="🧠", layout="wide")

# ──────────────────────────────────────────────────────────
#  PREMIUM DARK THEME + GLASSMORPHISM + ANIMATIONS
# ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ── CSS Variables ───────────────────────────────────── */
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
      --glow-orange:  0 0 25px rgba(251,146,60,0.25);
    }

    /* ── Keyframe Animations ─────────────────────────────── */
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(28px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
      from { opacity: 0; transform: translateX(-30px); }
      to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes shimmer {
      0%   { background-position: -200% center; }
      100% { background-position: 200% center; }
    }
    @keyframes glowPulse {
      0%, 100% { box-shadow: 0 0 12px rgba(99,179,237,0.15), 0 0 30px rgba(99,179,237,0.06); }
      50%      { box-shadow: 0 0 20px rgba(99,179,237,0.30), 0 0 50px rgba(99,179,237,0.12); }
    }
    @keyframes borderGlow {
      0%, 100% { border-color: rgba(99,179,237,0.2); }
      50%      { border-color: rgba(99,179,237,0.5); }
    }
    @keyframes float {
      0%, 100% { transform: translateY(0px); }
      50%      { transform: translateY(-6px); }
    }
    @keyframes scaleIn {
      from { opacity: 0; transform: scale(0.88); }
      to   { opacity: 1; transform: scale(1); }
    }
    @keyframes gradientMove {
      0%   { background-position: 0% 50%; }
      50%  { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    /* ── Global Background ───────────────────────────────── */
    .stApp, .main, [data-testid="stAppViewContainer"] {
      background: var(--bg-primary) !important;
      background-image:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99,179,237,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 85% 75%, rgba(167,139,250,0.05) 0%, transparent 50%),
        radial-gradient(ellipse 50% 40% at 50% 10%, rgba(34,211,238,0.04) 0%, transparent 45%) !important;
      font-family: 'Inter', sans-serif !important;
      color: var(--text-primary) !important;
    }
    [data-testid="stHeader"] {
      background: transparent !important;
    }
    .block-container {
      padding-top: 1.2rem !important;
      max-width: 1400px !important;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0d1321 0%, #111b2e 50%, #0d1321 100%) !important;
      border-right: 1px solid rgba(99,179,237,0.12) !important;
    }
    section[data-testid="stSidebar"] * {
      color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] .stSlider > div > div {
      background: rgba(99,179,237,0.1) !important;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 700;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
      font-size: 0.92rem;
      color: var(--text-muted) !important;
    }

    /* ── Main Title ──────────────────────────────────────── */
    h1 {
      font-family: 'Inter', sans-serif !important;
      font-weight: 900 !important;
      font-size: 2.6rem !important;
      background: linear-gradient(135deg, #63b3ed 0%, #22d3ee 25%, #a78bfa 50%, #63b3ed 75%, #22d3ee 100%);
      background-size: 200% auto;
      -webkit-background-clip: text !important;
      -webkit-text-fill-color: transparent !important;
      animation: shimmer 4s linear infinite, fadeInUp 0.8s ease-out;
      letter-spacing: -0.5px !important;
      line-height: 1.1 !important;
      padding-bottom: 0.3rem !important;
    }

    /* ── Subheadings ─────────────────────────────────────── */
    h2, h3 {
      color: var(--text-bright) !important;
      font-family: 'Inter', sans-serif !important;
      font-weight: 700 !important;
      letter-spacing: -0.3px !important;
    }

    /* ── Caption ─────────────────────────────────────────── */
    [data-testid="stCaptionContainer"] p {
      color: var(--text-muted) !important;
      font-size: 0.9rem !important;
    }

    /* ── Glass Card ──────────────────────────────────────── */
    .glass-card {
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--glass-border);
      border-radius: 16px;
      padding: 1.3rem 1.4rem;
      animation: fadeInUp 0.7s ease-out, glowPulse 4s ease-in-out infinite;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
      position: relative;
      overflow: hidden;
    }
    .glass-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-blue), transparent);
      opacity: 0.7;
    }
    .glass-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 8px 35px rgba(99,179,237,0.18);
    }
    .glass-card p {
      color: var(--text-primary) !important;
      margin: 0.35rem 0;
      font-size: 0.95rem;
      line-height: 1.65;
    }
    .glass-card b {
      color: var(--accent-cyan);
      font-weight: 600;
    }

    /* ── Metric Tiles ────────────────────────────────────── */
    .metric-row {
      display: flex;
      gap: 12px;
      margin-bottom: 1rem;
    }
    .metric-tile {
      flex: 1;
      background: rgba(17,24,39,0.5);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(99,179,237,0.12);
      border-radius: 12px;
      padding: 0.8rem 0.9rem;
      text-align: center;
      animation: scaleIn 0.6s ease-out backwards;
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }
    .metric-tile::after {
      content: '';
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    .metric-tile:hover::after { opacity: 1; }
    .metric-tile:hover {
      transform: translateY(-2px);
      border-color: rgba(99,179,237,0.3);
    }
    .metric-tile:nth-child(1) { animation-delay: 0.1s; }
    .metric-tile:nth-child(2) { animation-delay: 0.2s; }
    .metric-tile:nth-child(3) { animation-delay: 0.3s; }
    .metric-tile .metric-value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.4rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .metric-tile .metric-label {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-muted);
      margin-top: 0.15rem;
    }

    /* ── Vote Bar ────────────────────────────────────────── */
    .vote-bar-container {
      margin: 0.5rem 0;
      animation: fadeInUp 0.8s ease-out backwards;
    }
    .vote-bar-container:nth-child(2) { animation-delay: 0.15s; }
    .vote-bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      margin-bottom: 4px;
      font-weight: 500;
    }
    .vote-bar-track {
      height: 10px;
      background: rgba(255,255,255,0.06);
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }
    .vote-bar-fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
      position: relative;
      overflow: hidden;
    }
    .vote-bar-fill::after {
      content: '';
      position: absolute;
      top: 0; left: -50%; width: 50%; height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
      animation: shimmer 2.5s linear infinite;
    }
    .vote-bar-fill.blue {
      background: linear-gradient(90deg, #2563eb, #63b3ed);
      box-shadow: 0 0 12px rgba(99,179,237,0.3);
    }
    .vote-bar-fill.orange {
      background: linear-gradient(90deg, #ea580c, #fb923c);
      box-shadow: 0 0 12px rgba(251,146,60,0.3);
    }

    /* ── Section Header ──────────────────────────────────── */
    .section-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 1.1rem 0 0.7rem 0;
      animation: slideInLeft 0.6s ease-out;
    }
    .section-header .icon {
      font-size: 1.2rem;
      animation: float 3s ease-in-out infinite;
    }
    .section-header .text {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-bright);
      letter-spacing: -0.2px;
    }

    /* ── Hero Badge ──────────────────────────────────────── */
    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(34,211,238,0.08);
      border: 1px solid rgba(34,211,238,0.2);
      border-radius: 100px;
      padding: 0.35rem 1rem;
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--accent-cyan);
      animation: fadeInUp 0.5s ease-out;
      margin-bottom: 0.4rem;
    }
    .hero-badge .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent-cyan);
      animation: glowPulse 2s ease-in-out infinite;
    }

    /* ── Prediction Badge ────────────────────────────────── */
    .pred-badge {
      display: inline-block;
      padding: 0.3rem 1rem;
      border-radius: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      font-size: 1.1rem;
      animation: scaleIn 0.5s ease-out;
    }
    .pred-badge.classe-a {
      background: linear-gradient(135deg, rgba(37,99,235,0.2), rgba(99,179,237,0.15));
      color: var(--accent-blue);
      border: 1px solid rgba(99,179,237,0.3);
      text-shadow: 0 0 20px rgba(99,179,237,0.3);
    }
    .pred-badge.classe-b {
      background: linear-gradient(135deg, rgba(234,88,12,0.2), rgba(251,146,60,0.15));
      color: var(--accent-orange);
      border: 1px solid rgba(251,146,60,0.3);
      text-shadow: 0 0 20px rgba(251,146,60,0.3);
    }

    /* ── Dataframe / table ───────────────────────────────── */
    [data-testid="stDataFrame"] {
      animation: fadeInUp 0.7s ease-out 0.2s backwards;
    }
    [data-testid="stDataFrame"] table {
      border-radius: 10px;
      overflow: hidden;
    }

    /* ── Streamlit overrides ─────────────────────────────── */
    .stProgress > div > div > div {
      background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
    }
    hr {
      border-color: rgba(99,179,237,0.12) !important;
    }
    [data-testid="stMarkdownContainer"] p {
      color: var(--text-primary) !important;
    }
    /* Toggle */
    [data-testid="stBaseButton-secondary"] {
      border-color: rgba(99,179,237,0.2) !important;
      color: var(--text-primary) !important;
    }

    /* ── Floating SVG decoration ─────────────────────────── */
    .deco-lines {
      position: fixed;
      top: 0;
      right: 0;
      width: 420px;
      height: 420px;
      pointer-events: none;
      opacity: 0.04;
      z-index: 0;
    }

    /* ── Divider ─────────────────────────────────────────── */
    .gradient-divider {
      height: 1px;
      background: linear-gradient(90deg, transparent 0%, var(--accent-blue) 30%, var(--accent-cyan) 50%, var(--accent-blue) 70%, transparent 100%);
      margin: 1.5rem 0;
      opacity: 0.4;
      animation: fadeInUp 1s ease-out 0.5s backwards;
    }

    /* ── Footer ──────────────────────────────────────────── */
    .footer-text {
      text-align: center;
      font-size: 0.78rem;
      color: var(--text-muted);
      padding: 0.6rem 0 1.2rem 0;
      animation: fadeInUp 1s ease-out 0.6s backwards;
    }
    .footer-text a {
      color: var(--accent-cyan);
      text-decoration: none;
    }
    .footer-text .tech-stack {
      display: inline-flex;
      gap: 6px;
      margin-top: 0.3rem;
    }
    .footer-text .tech-tag {
      background: rgba(99,179,237,0.08);
      border: 1px solid rgba(99,179,237,0.15);
      border-radius: 6px;
      padding: 2px 8px;
      font-size: 0.72rem;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-muted);
    }

    /* ── Info list ────────────────────────────────────────── */
    .info-list {
      animation: fadeInUp 0.8s ease-out 0.3s backwards;
    }
    .info-list .info-item {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 0.45rem 0;
      font-size: 0.88rem;
      color: var(--text-muted);
      line-height: 1.55;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      transition: color 0.2s ease;
    }
    .info-list .info-item:hover {
      color: var(--text-primary);
    }
    .info-list .info-item:last-child {
      border-bottom: none;
    }
    .info-list .info-icon {
      flex-shrink: 0;
      width: 22px;
      height: 22px;
      background: rgba(34,211,238,0.1);
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.7rem;
      margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Decorative SVG ───────────────────────────────────────
st.markdown(
    """
    <svg class="deco-lines" viewBox="0 0 420 420" xmlns="http://www.w3.org/2000/svg">
      <circle cx="210" cy="210" r="180" stroke="#63b3ed" stroke-width="0.5" fill="none" opacity="0.5"/>
      <circle cx="210" cy="210" r="130" stroke="#22d3ee" stroke-width="0.4" fill="none" opacity="0.4"/>
      <circle cx="210" cy="210" r="80"  stroke="#a78bfa" stroke-width="0.3" fill="none" opacity="0.3"/>
      <line x1="0" y1="210" x2="420" y2="210" stroke="#63b3ed" stroke-width="0.3" opacity="0.2"/>
      <line x1="210" y1="0" x2="210" y2="420" stroke="#63b3ed" stroke-width="0.3" opacity="0.2"/>
    </svg>
    """,
    unsafe_allow_html=True,
)

# ── Hero Badge ───────────────────────────────────────────
st.markdown(
    '<div class="hero-badge"><span class="dot"></span> Simulation Interactive en Temps Réel</div>',
    unsafe_allow_html=True,
)

st.title("🧠 KNN Studio")
st.caption("Ajustez les paramètres et observez le vote des k voisins — Visualisation professionnelle")

# ──────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="section-header"><span class="icon">⚙️</span><span class="text">Paramètres</span></div>',
        unsafe_allow_html=True,
    )
    k_user = st.slider("Nombre de voisins (k)", min_value=1, max_value=9, value=5, step=2)

    st.markdown(
        '<div class="section-header"><span class="icon">📍</span><span class="text">Nouveau Point</span></div>',
        unsafe_allow_html=True,
    )
    x_point = st.slider("Coordonnée X", min_value=0.5, max_value=9.0, value=4.8, step=0.1)
    y_point = st.slider("Coordonnée Y", min_value=1.0, max_value=5.0, value=2.8, step=0.1)

    st.markdown(
        '<div class="section-header"><span class="icon">🗺️</span><span class="text">Affichage</span></div>',
        unsafe_allow_html=True,
    )
    show_boundary = st.toggle("Frontière de décision", value=True)
    show_confidence = st.toggle("Carte de confiance", value=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem;color:var(--text-muted);text-align:center;padding-top:0.3rem;">'
        "🧠 KNN Studio v2.0<br/>Ouahid Samrani</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────
#  DATA + MODEL
# ──────────────────────────────────────────────────────────
classe_a = np.array(
    [
        [1.0, 4.2], [1.4, 3.0], [2.0, 3.6], [2.3, 4.8], [2.8, 3.4],
        [3.1, 4.4], [1.8, 4.6], [2.6, 3.8], [3.6, 4.0], [1.2, 3.6],
    ]
)
classe_b = np.array(
    [
        [6.2, 2.1], [6.8, 1.6], [7.4, 2.4], [7.8, 3.0], [8.2, 2.6],
        [6.9, 3.3], [7.5, 2.9], [8.3, 3.6], [6.3, 1.4], [7.0, 2.7],
    ]
)

X = np.vstack([classe_a, classe_b])
y_labels = np.array([0] * len(classe_a) + [1] * len(classe_b))
new_point = np.array([[x_point, y_point]])

k_user = int(max(1, min(k_user, len(X))))

knn = KNeighborsClassifier(n_neighbors=k_user)
knn.fit(X, y_labels)
pred = int(knn.predict(new_point)[0])
proba = knn.predict_proba(new_point)[0]

k_dist, k_idx = knn.kneighbors(new_point, n_neighbors=k_user)
rayon = float(k_dist[0][-1])
voisins = X[k_idx[0]]
voisins_class = y_labels[k_idx[0]]

votes_a = int(np.sum(voisins_class == 0))
votes_b = int(np.sum(voisins_class == 1))

# ──────────────────────────────────────────────────────────
#  LAYOUT
# ──────────────────────────────────────────────────────────
col_chart, col_info = st.columns([3.0, 1.0], gap="large")

# ── CHART COLUMN ─────────────────────────────────────────
with col_chart:
    fig = go.Figure()

    # Decision boundary
    if show_boundary:
        gx = np.linspace(0.4, 9.5, 110)
        gy = np.linspace(1.1, 5.2, 110)
        xx, yy = np.meshgrid(gx, gy)
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_pred = knn.predict(grid).reshape(xx.shape)

        fig.add_trace(
            go.Contour(
                x=gx, y=gy, z=grid_pred,
                colorscale=[
                    [0.0, "rgba(37,99,235,0.18)"], [0.499, "rgba(37,99,235,0.18)"],
                    [0.5, "rgba(251,146,60,0.18)"], [1.0, "rgba(251,146,60,0.18)"],
                ],
                showscale=False, opacity=0.6,
                contours=dict(showlines=False),
                hoverinfo="skip", name="Frontière",
            )
        )

    # Confidence heatmap
    if show_confidence:
        gx2 = np.linspace(0.4, 9.5, 80)
        gy2 = np.linspace(1.1, 5.2, 80)
        xx2, yy2 = np.meshgrid(gx2, gy2)
        grid2 = np.c_[xx2.ravel(), yy2.ravel()]
        conf = knn.predict_proba(grid2)[:, 1].reshape(xx2.shape)

        fig.add_trace(
            go.Contour(
                x=gx2, y=gy2, z=conf,
                colorscale=[
                    [0.0, "rgba(37,99,235,0.22)"],
                    [0.5, "rgba(200,200,200,0.03)"],
                    [1.0, "rgba(251,146,60,0.22)"],
                ],
                showscale=False, opacity=0.4,
                contours=dict(start=0.0, end=1.0, size=0.1, coloring="heatmap", showlines=False),
                hoverinfo="skip", name="Confiance",
            )
        )

    # Classe A markers
    fig.add_trace(
        go.Scatter(
            x=classe_a[:, 0], y=classe_a[:, 1], mode="markers", name="Classe A",
            marker=dict(
                size=13, color="#2563eb",
                line=dict(color="rgba(37,99,235,0.5)", width=2),
                symbol="circle",
            ),
            hovertemplate="Classe A<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )

    # Classe B markers
    fig.add_trace(
        go.Scatter(
            x=classe_b[:, 0], y=classe_b[:, 1], mode="markers", name="Classe B",
            marker=dict(
                size=13, color="#ea580c",
                line=dict(color="rgba(234,88,12,0.5)", width=2),
                symbol="diamond",
            ),
            hovertemplate="Classe B<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )

    # K-nearest neighbors highlight
    fig.add_trace(
        go.Scatter(
            x=voisins[:, 0], y=voisins[:, 1], mode="markers",
            name=f"{k_user} voisins",
            marker=dict(
                size=22, color="rgba(0,0,0,0)",
                line=dict(color="#0891b2", width=2.5),
            ),
            hovertemplate="Voisin<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )

    # New point (query)
    fig.add_trace(
        go.Scatter(
            x=[new_point[0, 0]], y=[new_point[0, 1]],
            mode="markers+text", name="Nouveau point",
            text=["?"], textposition="middle center",
            textfont=dict(size=18, color="#ffffff", family="Inter"),
            marker=dict(
                size=30, color="#0891b2",
                line=dict(color="#164e63", width=2),
                symbol="circle",
            ),
            hovertemplate="Nouveau point<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )

    # Search radius circle
    t = np.linspace(0, 2 * np.pi, 180)
    circle_x = new_point[0, 0] + rayon * np.cos(t)
    circle_y = new_point[0, 1] + rayon * np.sin(t)
    fig.add_trace(
        go.Scatter(
            x=circle_x, y=circle_y, mode="lines",
            name="Zone de recherche",
            line=dict(color="rgba(8,145,178,0.55)", width=1.8, dash="dash"),
        )
    )

    # Lines connecting new_point to each neighbor
    for v_coord in voisins:
        fig.add_trace(
            go.Scatter(
                x=[new_point[0, 0], v_coord[0]],
                y=[new_point[0, 1], v_coord[1]],
                mode="lines", showlegend=False,
                line=dict(color="rgba(8,145,178,0.25)", width=1, dash="dot"),
                hoverinfo="skip",
            )
        )

    # Dynamic axis ranges that always contain the full circle
    pad = 0.5
    x_min = min(0.2, new_point[0, 0] - rayon - pad)
    x_max = max(9.7, new_point[0, 0] + rayon + pad)
    y_min = min(0.9, new_point[0, 1] - rayon - pad)
    y_max = max(5.4, new_point[0, 1] + rayon + pad)

    # Equalise the ranges so the circle is always perfectly round
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span > y_span:
        mid_y = (y_min + y_max) / 2
        y_min = mid_y - x_span / 2
        y_max = mid_y + x_span / 2
    else:
        mid_x = (x_min + x_max) / 2
        x_min = mid_x - y_span / 2
        x_max = mid_x + y_span / 2

    fig.update_layout(
        title=dict(
            text=f"<b>Simulation KNN</b> — <span style='color:#0891b2'>K = {k_user}</span>",
            font=dict(size=16, color="#1e293b", family="Inter"),
        ),
        height=750,
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#475569", family="Inter"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color="#475569"),
            bgcolor="rgba(255,255,255,0)",
        ),
        margin=dict(l=25, r=25, t=60, b=30),
    )
    fig.update_xaxes(
        range=[x_min, x_max], showgrid=True,
        gridcolor="rgba(0,0,0,0.06)", gridwidth=1,
        title=dict(text="Caractéristique 1", font=dict(size=12, color="#334155")),
        tickfont=dict(color="#64748b"), zeroline=False,
        linecolor="rgba(0,0,0,0.1)",
        constrain="domain",
    )
    fig.update_yaxes(
        range=[y_min, y_max], showgrid=True,
        gridcolor="rgba(0,0,0,0.06)", gridwidth=1,
        title=dict(text="Caractéristique 2", font=dict(size=12, color="#334155")),
        tickfont=dict(color="#64748b"), zeroline=False,
        linecolor="rgba(0,0,0,0.1)",
        scaleanchor="x", scaleratio=1, constrain="domain",
    )

    st.plotly_chart(fig, use_container_width=True)

# ── INFO COLUMN ──────────────────────────────────────────
with col_info:
    pred_label = "Classe A" if pred == 0 else "Classe B"
    pred_css = "classe-a" if pred == 0 else "classe-b"
    confidence_val = float(proba[pred]) * 100

    # Prediction card
    st.markdown(
        '<div class="section-header"><span class="icon">🎯</span><span class="text">Prédiction</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="glass-card">'
        f'<p style="margin-bottom:0.6rem;color:var(--text-muted)!important;font-size:0.82rem;">'
        f'RÉSULTAT DU MODÈLE</p>'
        f'<span class="pred-badge {pred_css}">{pred_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Metric tiles
    st.markdown(
        f"""
        <div class="metric-row" style="margin-top:0.8rem;">
          <div class="metric-tile">
            <div class="metric-value">{confidence_val:.0f}%</div>
            <div class="metric-label">Confiance</div>
          </div>
          <div class="metric-tile">
            <div class="metric-value">{k_user}</div>
            <div class="metric-label">Voisins (k)</div>
          </div>
          <div class="metric-tile">
            <div class="metric-value">{rayon:.2f}</div>
            <div class="metric-label">Rayon</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Vote visualization
    st.markdown(
        '<div class="section-header"><span class="icon">🗳️</span><span class="text">Vote des Voisins</span></div>',
        unsafe_allow_html=True,
    )

    pct_a = (votes_a / k_user * 100) if k_user > 0 else 0
    pct_b = (votes_b / k_user * 100) if k_user > 0 else 0

    st.markdown(
        f"""
        <div class="vote-bar-container">
          <div class="vote-bar-label">
            <span style="color:var(--accent-blue);">● Classe A</span>
            <span style="font-family:'JetBrains Mono',monospace;color:var(--text-muted);">{votes_a}/{k_user}</span>
          </div>
          <div class="vote-bar-track">
            <div class="vote-bar-fill blue" style="width:{pct_a}%;"></div>
          </div>
        </div>
        <div class="vote-bar-container">
          <div class="vote-bar-label">
            <span style="color:var(--accent-orange);">◆ Classe B</span>
            <span style="font-family:'JetBrains Mono',monospace;color:var(--text-muted);">{votes_b}/{k_user}</span>
          </div>
          <div class="vote-bar-track">
            <div class="vote-bar-fill orange" style="width:{pct_b}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Vote donut chart
    vote_fig = go.Figure(
        data=[
            go.Pie(
                labels=["Classe A", "Classe B"],
                values=[votes_a, votes_b],
                hole=0.55,
                marker=dict(colors=["#3b82f6", "#f97316"], line=dict(color="#0f172a", width=2)),
                textinfo="percent+label",
                textfont=dict(size=11, color="#e2e8f0", family="Inter"),
                hovertemplate="%{label}: %{value} vote(s)<extra></extra>",
            )
        ]
    )
    vote_fig.update_layout(
        height=200,
        margin=dict(l=5, r=5, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{k_user}</b>",
                x=0.5, y=0.5, font=dict(size=22, color="#22d3ee", family="JetBrains Mono"),
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(vote_fig, use_container_width=True)

    # Neighbor table
    st.markdown(
        '<div class="section-header"><span class="icon">📋</span><span class="text">Voisins Utilisés</span></div>',
        unsafe_allow_html=True,
    )

    neighbor_rows = []
    for rank, (coord, dist, cls) in enumerate(zip(voisins, k_dist[0], voisins_class), start=1):
        neighbor_rows.append(
            {
                "Rang": rank,
                "Point": f"({coord[0]:.1f}, {coord[1]:.1f})",
                "Distance": round(float(dist), 3),
                "Classe": "A" if int(cls) == 0 else "B",
            }
        )
    st.dataframe(neighbor_rows, use_container_width=True, hide_index=True)

    # Quick info
    st.markdown(
        '<div class="section-header"><span class="icon">💡</span><span class="text">Lecture Rapide</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-list">
          <div class="info-item">
            <span class="info-icon">🔍</span>
            <span>KNN cherche les <b>k voisins les plus proches</b> puis détermine la classe par vote majoritaire.</span>
          </div>
          <div class="info-item">
            <span class="info-icon">🎨</span>
            <span>La frontière colorée représente les <b>zones de décision</b> qui se déplacent selon k.</span>
          </div>
          <div class="info-item">
            <span class="info-icon">📊</span>
            <span>La confiance est la <b>probabilité</b> estimée de la classe prédite par le vote local.</span>
          </div>
          <div class="info-item">
            <span class="info-icon">⭕</span>
            <span>Le cercle pointillé délimite la <b>zone de recherche</b> couverte par les k voisins.</span>
          </div>
          <div class="info-item">
            <span class="info-icon">✨</span>
            <span>Les points entourés en <b style="color:var(--accent-cyan);">cyan</b> participent au vote final.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ───────────────────────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer-text">
      <div>Built with ❤️ by <b>Ouahid Samrani</b></div>
      <div class="tech-stack" style="margin-top:6px;">
        <span class="tech-tag">Streamlit</span>
        <span class="tech-tag">scikit-learn</span>
        <span class="tech-tag">Plotly</span>
        <span class="tech-tag">NumPy</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
