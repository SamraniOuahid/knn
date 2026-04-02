import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="KNN Real Case Web", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
      --bg-1: #f4f8fb;
      --bg-2: #e7f0f7;
            --card: #fffffff2;
      --ink: #12314a;
      --muted: #4c6b85;
      --blue: #1f6fb2;
      --cyan: #1ea7b8;
      --orange: #e56a2d;
      --border: #c9dce9;
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes glow {
      0%, 100% { box-shadow: 0 6px 18px rgba(31, 111, 178, 0.08); }
      50% { box-shadow: 0 10px 24px rgba(31, 111, 178, 0.18); }
    }

    .stApp {
      background:
        radial-gradient(circle at 10% -10%, #d6eaf7 0%, transparent 42%),
        radial-gradient(circle at 100% 10%, #ffe8db 0%, transparent 35%),
        linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
            font-family: 'IBM Plex Sans', 'Segoe UI', Tahoma, Arial, sans-serif;
            color: var(--ink);
    }

        .stApp p, .stApp li, .stApp label, .stApp span, .stApp div {
            color: var(--ink);
        }

    .block-container {
      max-width: 1320px;
      padding-top: 1rem;
    }

    .hero {
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 1rem 1.2rem;
            background: linear-gradient(130deg, #fffffffa 0%, #f2f9fffa 70%, #edf7fffa 100%);
      animation: rise .7s ease-out;
    }

    .hero h1 {
      margin: .2rem 0 .4rem 0;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 2rem;
      color: var(--ink);
      letter-spacing: -.4px;
    }

    .badge {
      display: inline-block;
      padding: .24rem .62rem;
      border-radius: 999px;
      font-size: .75rem;
      border: 1px solid #b8d6ea;
      color: #0f4f7e;
      background: #eaf5ff;
    }

    .kpi {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: .65rem .75rem;
            background: #ffffff;
      animation: rise .8s ease-out;
    }

    .kpi .label {
      font-size: .75rem;
      color: var(--muted);
    }

    .kpi .value {
      font-family: 'Space Grotesk', sans-serif;
      color: var(--ink);
      font-weight: 700;
      font-size: 1.05rem;
      margin-top: .08rem;
    }

    .insight {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: .85rem .95rem;
            background: #ffffff;
      animation: rise .8s ease-out, glow 4s ease-in-out infinite;
    }

    .insight p {
      margin: .24rem 0;
      color: var(--ink);
      font-size: .92rem;
    }

    .section-title {
      font-family: 'Space Grotesk', sans-serif;
      color: var(--ink);
      font-size: 1.02rem;
      margin: .25rem 0 .55rem 0;
    }

    section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #f5fbff 0%, #eaf4fb 100%);
      border-right: 1px solid #d2e3ef;
    }

    .img-card {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: .4rem;
            background: #ffffff;
      animation: rise .9s ease-out;
    }

        [data-testid="stDataFrame"] {
            background: #ffffff;
            border-radius: 10px;
            border: 1px solid #d4e3ee;
            padding: 2px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <span class="badge">NEW WEB VERSION</span>
      <h1>Cas Pratique Réel: Triage Métabolique avec KNN</h1>
      <p style="color:#4c6b85;margin:0;">Le modèle classe un patient en <b>risque faible</b> ou <b>risque élevé</b> selon l'IMC et la glycémie.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Paramètres du cas réel")
    k_user = st.slider("Nombre de voisins (k)", min_value=1, max_value=9, value=5, step=2)
    imc = st.slider("IMC du patient", min_value=18.0, max_value=36.0, value=26.5, step=0.1)
    glycemie = st.slider("Glycémie (mg/dL)", min_value=80.0, max_value=150.0, value=108.0, step=0.5)
    show_boundary = st.toggle("Frontière de décision", value=True)
    show_confidence = st.toggle("Carte de confiance", value=True)

# Donnees reelles simplifiees (demo)
faible_risque = np.array([
    [20.2, 82], [21.5, 88], [22.1, 91], [23.0, 95], [24.2, 99],
    [25.1, 101], [21.0, 86], [22.8, 93], [24.7, 97], [23.5, 90],
])
eleve_risque = np.array([
    [28.4, 118], [29.6, 124], [30.8, 130], [31.5, 135], [33.0, 140],
    [27.9, 116], [32.2, 138], [34.0, 145], [29.1, 121], [31.0, 133],
])

X = np.vstack([faible_risque, eleve_risque])
y = np.array([0] * len(faible_risque) + [1] * len(eleve_risque))
new_point = np.array([[imc, glycemie]])

k_user = int(max(1, min(k_user, len(X))))
knn = KNeighborsClassifier(n_neighbors=k_user)
knn.fit(X, y)
pred = int(knn.predict(new_point)[0])
proba = knn.predict_proba(new_point)[0]

dist, idx = knn.kneighbors(new_point, n_neighbors=k_user)
rayon = float(dist[0][-1])
voisins = X[idx[0]]
voisins_class = y[idx[0]]

votes_faible = int(np.sum(voisins_class == 0))
votes_eleve = int(np.sum(voisins_class == 1))

pred_label = "Risque faible" if pred == 0 else "Risque élevé"
conf_label = float(proba[pred]) * 100

# Comparaison avec une regle simple (baseline)
baseline_pred = 1 if glycemie >= 115 else 0
baseline_label = "Risque élevé" if baseline_pred == 1 else "Risque faible"

# Qualite sur l'echantillon (illustratif)
knn_train_pred = knn.predict(X)
knn_train_acc = accuracy_score(y, knn_train_pred)

baseline_train_pred = (X[:, 1] >= 115).astype(int)
baseline_train_acc = accuracy_score(y, baseline_train_pred)

# Sensibilite de la prediction selon k
k_candidates = [1, 3, 5, 7, 9]
k_pred_risk = []
for k_val in k_candidates:
  k_model = KNeighborsClassifier(n_neighbors=k_val)
  k_model.fit(X, y)
  k_pred_risk.append(int(k_model.predict(new_point)[0]))

k_pred_numeric = [0.15 if p == 0 else 0.85 for p in k_pred_risk]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='kpi'><div class='label'>Prédiction</div><div class='value'>{pred_label}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='label'>Confiance</div><div class='value'>{conf_label:.1f}%</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='label'>Votes Faible</div><div class='value'>{votes_faible}/{k_user}</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi'><div class='label'>Votes Élevé</div><div class='value'>{votes_eleve}/{k_user}</div></div>", unsafe_allow_html=True)

left, right = st.columns([2.1, 1.0], gap="large")

with left:
    fig = go.Figure()

    if show_boundary:
        gx = np.linspace(18.0, 36.0, 140)
        gy = np.linspace(80.0, 150.0, 140)
        xx, yy = np.meshgrid(gx, gy)
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = knn.predict(grid).reshape(xx.shape)

        fig.add_trace(
            go.Contour(
                x=gx,
                y=gy,
                z=zz,
                colorscale=[[0.0, "#dbeafe"], [0.499, "#dbeafe"], [0.5, "#ffedd5"], [1.0, "#ffedd5"]],
                showscale=False,
                opacity=0.35,
                contours=dict(showlines=False),
                hoverinfo="skip",
                name="Frontière",
            )
        )

    if show_confidence:
        gx2 = np.linspace(18.0, 36.0, 120)
        gy2 = np.linspace(80.0, 150.0, 120)
        xx2, yy2 = np.meshgrid(gx2, gy2)
        grid2 = np.c_[xx2.ravel(), yy2.ravel()]
        conf = knn.predict_proba(grid2)[:, 1].reshape(xx2.shape)

        fig.add_trace(
            go.Contour(
                x=gx2,
                y=gy2,
                z=conf,
                colorscale="RdBu_r",
                showscale=False,
                opacity=0.16,
                contours=dict(start=0.0, end=1.0, size=0.1, coloring="heatmap", showlines=False),
                hoverinfo="skip",
                name="Confiance",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=faible_risque[:, 0],
            y=faible_risque[:, 1],
            mode="markers",
            name="Risque faible",
            marker=dict(size=12, color="#1f6fb2", line=dict(color="white", width=1.4)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eleve_risque[:, 0],
            y=eleve_risque[:, 1],
            mode="markers",
            name="Risque élevé",
            marker=dict(size=12, color="#e56a2d", symbol="diamond", line=dict(color="white", width=1.4)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=voisins[:, 0],
            y=voisins[:, 1],
            mode="markers",
            name=f"{k_user} voisins",
            marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(color="#1ea7b8", width=2.4)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[new_point[0, 0]],
            y=[new_point[0, 1]],
            mode="markers+text",
            name="Nouveau patient",
            text=["?"],
            textposition="middle center",
            textfont=dict(size=16, color="white"),
            marker=dict(size=28, color="#1ea7b8", line=dict(color="#104f58", width=2)),
        )
    )

    t = np.linspace(0, 2 * np.pi, 180)
    fig.add_trace(
        go.Scatter(
            x=new_point[0, 0] + rayon * np.cos(t),
            y=new_point[0, 1] + rayon * np.sin(t),
            mode="lines",
            name="Zone de recherche",
            line=dict(color="#1ea7b8", width=1.8, dash="dash"),
        )
    )

    for vc in voisins:
        fig.add_trace(
            go.Scatter(
                x=[new_point[0, 0], vc[0]],
                y=[new_point[0, 1], vc[1]],
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(30,167,184,0.25)", width=1, dash="dot"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"KNN sur cas réel (IMC, Glycémie) - K={k_user}",
        height=690,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(range=[17.5, 36.5], title="IMC", showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(range=[79, 151], title="Glycémie (mg/dL)", showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    st.plotly_chart(fig, width="stretch")

with right:
    st.markdown("<div class='section-title'>Résumé clinique simulé</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="insight">
          <p><b>Décision KNN:</b> {pred_label}</p>
          <p><b>Confiance:</b> {conf_label:.1f}%</p>
          <p><b>IMC patient:</b> {imc:.1f}</p>
          <p><b>Glycémie:</b> {glycemie:.1f} mg/dL</p>
          <p><b>Rayon de recherche:</b> {rayon:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Images réelles (contexte)</div>", unsafe_allow_html=True)
    st.markdown("<div class='img-card'>", unsafe_allow_html=True)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Body_mass_index_chart.svg/640px-Body_mass_index_chart.svg.png",
        caption="Référence IMC (Wikimedia Commons)",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='img-card' style='margin-top:.5rem;'>", unsafe_allow_html=True)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Blood_glucose_testing.JPG/640px-Blood_glucose_testing.JPG",
        caption="Mesure de glycémie (Wikimedia Commons)",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Voisins utilisés</div>", unsafe_allow_html=True)
    rows = []
    for rank, (coord, d, cls) in enumerate(zip(voisins, dist[0], voisins_class), start=1):
        rows.append(
            {
                "Rang": rank,
                "Patient": f"(IMC {coord[0]:.1f}, G {coord[1]:.0f})",
                "Distance": round(float(d), 2),
                "Classe": "Faible" if int(cls) == 0 else "Élevé",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")

st.markdown("### Pourquoi KNN est important ici")

exp_col1, exp_col2 = st.columns([1.05, 1.0], gap="large")

with exp_col1:
    st.markdown(
        f"""
        <div class="insight">
          <p><b>KNN (local):</b> {pred_label}</p>
          <p><b>Règle simple (glycémie seule):</b> {baseline_label}</p>
          <p><b>Accuracy KNN (échantillon):</b> {knn_train_acc * 100:.1f}%</p>
          <p><b>Accuracy règle simple:</b> {baseline_train_acc * 100:.1f}%</p>
          <p style="color:#36566f; margin-top:.45rem;">
            KNN exploite <b>deux variables en même temps</b> (IMC + glycémie) et adapte la décision au voisinage réel du patient,
            alors qu'une règle fixe peut manquer des cas intermédiaires.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pred != baseline_pred:
        st.warning("Ici, KNN et la règle simple ne donnent pas la même décision: ce cas montre l'intérêt concret de KNN.")
    else:
        st.info("Ici, les deux approches donnent la même décision, mais KNN fournit une justification locale via les voisins.")

with exp_col2:
    sens_fig = go.Figure()
    sens_fig.add_trace(
        go.Scatter(
            x=k_candidates,
            y=k_pred_numeric,
            mode="lines+markers",
            line=dict(color="#1f6fb2", width=3),
            marker=dict(size=10, color="#1ea7b8", line=dict(color="white", width=1.3)),
            hovertemplate="k=%{x}<br>Prediction=%{customdata}<extra></extra>",
            customdata=["Faible" if p == 0 else "Élevé" for p in k_pred_risk],
            name="Prédiction",
        )
    )
    sens_fig.update_layout(
        title="Stabilité de la décision selon k",
        height=290,
        margin=dict(l=15, r=10, t=45, b=15),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    sens_fig.update_xaxes(title="k", tickmode="array", tickvals=k_candidates)
    sens_fig.update_yaxes(
        title="Classe prédite",
        range=[0, 1],
        tickmode="array",
        tickvals=[0.15, 0.85],
        ticktext=["Faible", "Élevé"],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    st.plotly_chart(sens_fig, width="stretch")

    stable_count = sum(1 for p in k_pred_risk if p == pred)
    st.caption(f"La décision actuelle est retrouvée pour {stable_count}/{len(k_candidates)} valeurs de k testées.")

st.caption("Version web enrichie avec animations, storytelling visuel et images réelles de contexte.")
