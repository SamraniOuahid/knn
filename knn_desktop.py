"""
KNN Studio — Desktop Edition
Standalone desktop application with CustomTkinter + Matplotlib.
Author: Ouahid Samrani
"""

import customtkinter as ctk
import matplotlib
import numpy as np

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from sklearn.neighbors import KNeighborsClassifier

# ──────────────────────────────────────────────────────────
#  THEME
# ──────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Color palette
BG_DARK = "#0a0e1a"
BG_CARD = "#111827"
BG_SIDEBAR = "#0d1321"
ACCENT_CYAN = "#22d3ee"
ACCENT_BLUE = "#3b82f6"
ACCENT_ORANGE = "#f97316"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
PLOT_BG = "#ffffff"

# ──────────────────────────────────────────────────────────
#  DATA
# ──────────────────────────────────────────────────────────
CLASSE_A = np.array([
    [1.0, 4.2], [1.4, 3.0], [2.0, 3.6], [2.3, 4.8], [2.8, 3.4],
    [3.1, 4.4], [1.8, 4.6], [2.6, 3.8], [3.6, 4.0], [1.2, 3.6],
])
CLASSE_B = np.array([
    [6.2, 2.1], [6.8, 1.6], [7.4, 2.4], [7.8, 3.0], [8.2, 2.6],
    [6.9, 3.3], [7.5, 2.9], [8.3, 3.6], [6.3, 1.4], [7.0, 2.7],
])
X_ALL = np.vstack([CLASSE_A, CLASSE_B])
Y_ALL = np.array([0] * len(CLASSE_A) + [1] * len(CLASSE_B))


class KNNStudioApp(ctk.CTk):
    """Main desktop application window."""

    def __init__(self):
        super().__init__()
        self.title("🧠  KNN Studio — Desktop Edition")
        self.geometry("1420x820")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_DARK)

        # ── Variables ────────────────────────────────────
        self.k_var = ctk.IntVar(value=5)
        self.x_var = ctk.DoubleVar(value=4.8)
        self.y_var = ctk.DoubleVar(value=2.8)
        self.show_boundary = ctk.BooleanVar(value=True)
        self.show_confidence = ctk.BooleanVar(value=True)

        self._build_ui()
        self._update_plot()

    # ──────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ──────────────────────────────────────────────────
    def _build_ui(self):
        # Root grid
        self.grid_columnconfigure(0, weight=0, minsize=280)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──────────────────────────────────
        sidebar = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, corner_radius=0, width=280)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        # Logo / Title
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=20, pady=(24, 6))
        ctk.CTkLabel(
            logo_frame, text="🧠  KNN Studio",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=ACCENT_CYAN,
        ).pack(anchor="w")
        ctk.CTkLabel(
            logo_frame, text="Desktop Edition  v2.0",
            font=ctk.CTkFont(size=11), text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        self._add_separator(sidebar)

        # ── K slider ─────────────────────────────────
        self._section_label(sidebar, "⚙️  Paramètres")
        self.k_label = ctk.CTkLabel(
            sidebar, text=f"k = {self.k_var.get()}", text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=13),
        )
        self.k_label.pack(padx=24, anchor="w")
        k_slider = ctk.CTkSlider(
            sidebar, from_=1, to=9, number_of_steps=4,
            variable=self.k_var, command=self._on_k_change,
            progress_color=ACCENT_CYAN, button_color=ACCENT_CYAN,
            button_hover_color="#06b6d4",
        )
        k_slider.pack(fill="x", padx=24, pady=(2, 12))

        # ── Point sliders ────────────────────────────
        self._section_label(sidebar, "📍  Nouveau Point")

        self.x_label = ctk.CTkLabel(
            sidebar, text=f"x = {self.x_var.get():.1f}", text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=13),
        )
        self.x_label.pack(padx=24, anchor="w")
        ctk.CTkSlider(
            sidebar, from_=0.5, to=9.0, number_of_steps=85,
            variable=self.x_var, command=self._on_x_change,
            progress_color=ACCENT_BLUE, button_color=ACCENT_BLUE,
            button_hover_color="#2563eb",
        ).pack(fill="x", padx=24, pady=(2, 8))

        self.y_label = ctk.CTkLabel(
            sidebar, text=f"y = {self.y_var.get():.1f}", text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=13),
        )
        self.y_label.pack(padx=24, anchor="w")
        ctk.CTkSlider(
            sidebar, from_=1.0, to=5.0, number_of_steps=40,
            variable=self.y_var, command=self._on_y_change,
            progress_color=ACCENT_BLUE, button_color=ACCENT_BLUE,
            button_hover_color="#2563eb",
        ).pack(fill="x", padx=24, pady=(2, 12))

        # ── Toggles ──────────────────────────────────
        self._section_label(sidebar, "🗺️  Affichage")
        ctk.CTkSwitch(
            sidebar, text="Frontière de décision",
            variable=self.show_boundary, command=self._update_plot,
            progress_color=ACCENT_CYAN, button_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY,
        ).pack(padx=24, anchor="w", pady=4)
        ctk.CTkSwitch(
            sidebar, text="Carte de confiance",
            variable=self.show_confidence, command=self._update_plot,
            progress_color=ACCENT_CYAN, button_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY,
        ).pack(padx=24, anchor="w", pady=4)

        self._add_separator(sidebar)

        # ── Results panel ────────────────────────────
        self._section_label(sidebar, "🎯  Résultat")

        self.result_frame = ctk.CTkFrame(
            sidebar, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color="#1e3a5f",
        )
        self.result_frame.pack(fill="x", padx=20, pady=(4, 8))

        self.pred_label_widget = ctk.CTkLabel(
            self.result_frame, text="—", text_color=ACCENT_CYAN,
            font=ctk.CTkFont(family="Consolas", size=20, weight="bold"),
        )
        self.pred_label_widget.pack(pady=(12, 2))

        self.conf_label_widget = ctk.CTkLabel(
            self.result_frame, text="Confiance: —", text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.conf_label_widget.pack(pady=(0, 4))

        self.radius_label_widget = ctk.CTkLabel(
            self.result_frame, text="Rayon: —", text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.radius_label_widget.pack(pady=(0, 12))

        # ── Vote bars ────────────────────────────────
        self._section_label(sidebar, "🗳️  Vote")
        self.vote_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        self.vote_frame.pack(fill="x", padx=20, pady=(4, 8))

        self.vote_a_bar = ctk.CTkProgressBar(
            self.vote_frame, progress_color=ACCENT_BLUE,
            fg_color="#1e293b", height=12, corner_radius=6,
        )
        self.vote_a_label = ctk.CTkLabel(
            self.vote_frame, text="Classe A: 0/0", text_color=ACCENT_BLUE,
            font=ctk.CTkFont(size=11),
        )
        self.vote_a_label.pack(anchor="w")
        self.vote_a_bar.pack(fill="x", pady=(2, 6))

        self.vote_b_bar = ctk.CTkProgressBar(
            self.vote_frame, progress_color=ACCENT_ORANGE,
            fg_color="#1e293b", height=12, corner_radius=6,
        )
        self.vote_b_label = ctk.CTkLabel(
            self.vote_frame, text="Classe B: 0/0", text_color=ACCENT_ORANGE,
            font=ctk.CTkFont(size=11),
        )
        self.vote_b_label.pack(anchor="w")
        self.vote_b_bar.pack(fill="x", pady=(2, 4))

        # ── Footer ───────────────────────────────────
        ctk.CTkLabel(
            sidebar, text="Built with ❤️ by Ouahid Samrani",
            font=ctk.CTkFont(size=10), text_color=TEXT_MUTED,
        ).pack(side="bottom", pady=(0, 16))

        # ── Chart area ───────────────────────────────
        chart_frame = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.fig, self.ax = plt.subplots(figsize=(10, 7), dpi=100)
        self.fig.patch.set_facecolor(BG_DARK)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # ──────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────
    @staticmethod
    def _add_separator(parent):
        sep = ctk.CTkFrame(parent, fg_color="#1e3a5f", height=1)
        sep.pack(fill="x", padx=20, pady=12)

    @staticmethod
    def _section_label(parent, text):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=ACCENT_CYAN,
        ).pack(padx=20, anchor="w", pady=(8, 4))

    # ──────────────────────────────────────────────────
    #  SLIDER CALLBACKS
    # ──────────────────────────────────────────────────
    def _on_k_change(self, val):
        k = int(round(float(val)))
        # force odd
        if k % 2 == 0:
            k = max(1, k - 1)
        self.k_var.set(k)
        self.k_label.configure(text=f"k = {k}")
        self._update_plot()

    def _on_x_change(self, val):
        v = round(float(val), 1)
        self.x_label.configure(text=f"x = {v:.1f}")
        self._update_plot()

    def _on_y_change(self, val):
        v = round(float(val), 1)
        self.y_label.configure(text=f"y = {v:.1f}")
        self._update_plot()

    # ──────────────────────────────────────────────────
    #  MAIN PLOT
    # ──────────────────────────────────────────────────
    def _update_plot(self, *_args):
        ax = self.ax
        ax.clear()

        k = self.k_var.get()
        xp = self.x_var.get()
        yp = self.y_var.get()
        new_pt = np.array([[xp, yp]])

        k = int(max(1, min(k, len(X_ALL))))

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_ALL, Y_ALL)
        pred = int(knn.predict(new_pt)[0])
        proba = knn.predict_proba(new_pt)[0]

        k_dist, k_idx = knn.kneighbors(new_pt, n_neighbors=k)
        rayon = float(k_dist[0][-1])
        voisins = X_ALL[k_idx[0]]
        voisins_class = Y_ALL[k_idx[0]]
        votes_a = int(np.sum(voisins_class == 0))
        votes_b = int(np.sum(voisins_class == 1))

        # ── Background fills ─────────────────────────
        if self.show_boundary.get():
            gx = np.linspace(0.0, 10.0, 120)
            gy = np.linspace(0.5, 5.5, 120)
            xx, yy = np.meshgrid(gx, gy)
            grid = np.c_[xx.ravel(), yy.ravel()]
            zz = knn.predict(grid).reshape(xx.shape)
            ax.contourf(
                xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                colors=["#dbeafe", "#ffedd5"], alpha=0.35,
            )
            ax.contour(
                xx, yy, zz, levels=[0.5],
                colors=["#94a3b8"], linewidths=1.0, linestyles="--",
            )

        if self.show_confidence.get():
            gx2 = np.linspace(0.0, 10.0, 90)
            gy2 = np.linspace(0.5, 5.5, 90)
            xx2, yy2 = np.meshgrid(gx2, gy2)
            grid2 = np.c_[xx2.ravel(), yy2.ravel()]
            conf = knn.predict_proba(grid2)[:, 1].reshape(xx2.shape)
            ax.contourf(
                xx2, yy2, conf, levels=10,
                cmap="RdBu_r", alpha=0.12,
            )

        # ── Data points ──────────────────────────────
        ax.scatter(
            CLASSE_A[:, 0], CLASSE_A[:, 1],
            s=110, c=ACCENT_BLUE, edgecolors="white", linewidths=1.5,
            zorder=5, label="Classe A",
        )
        ax.scatter(
            CLASSE_B[:, 0], CLASSE_B[:, 1],
            s=110, c=ACCENT_ORANGE, edgecolors="white", linewidths=1.5,
            zorder=5, label="Classe B", marker="D",
        )

        # ── Neighbor highlights ──────────────────────
        ax.scatter(
            voisins[:, 0], voisins[:, 1],
            s=260, facecolors="none", edgecolors="#0891b2",
            linewidths=2.2, zorder=6,
        )

        # ── Lines to neighbors ───────────────────────
        for vc in voisins:
            ax.plot(
                [xp, vc[0]], [yp, vc[1]],
                color="#0891b2", alpha=0.25, linewidth=0.8,
                linestyle=":", zorder=4,
            )

        # ── Search radius circle ─────────────────────
        circle = Circle(
            (xp, yp), rayon, fill=False,
            edgecolor="#0891b2", linewidth=1.8, linestyle="--",
            alpha=0.55, zorder=7,
        )
        ax.add_patch(circle)

        # ── New point ────────────────────────────────
        ax.scatter(
            [xp], [yp], s=320, c="#0891b2",
            edgecolors="#164e63", linewidths=2, zorder=8,
        )
        ax.text(
            xp, yp, "?", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white", zorder=9,
        )

        # ── Axis styling ─────────────────────────────
        pad = 0.5
        x_lo = min(0.0, xp - rayon - pad)
        x_hi = max(10.0, xp + rayon + pad)
        y_lo = min(0.5, yp - rayon - pad)
        y_hi = max(5.5, yp + rayon + pad)

        # Equalise spans for 1:1 aspect
        x_span = x_hi - x_lo
        y_span = y_hi - y_lo
        if x_span > y_span:
            mid = (y_lo + y_hi) / 2
            y_lo, y_hi = mid - x_span / 2, mid + x_span / 2
        else:
            mid = (x_lo + x_hi) / 2
            x_lo, x_hi = mid - y_span / 2, mid + y_span / 2

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_facecolor(PLOT_BG)
        ax.set_xlabel("Caractéristique 1", fontsize=11, color="#334155", labelpad=8)
        ax.set_ylabel("Caractéristique 2", fontsize=11, color="#334155", labelpad=8)
        ax.set_title(
            f"Simulation KNN  —  K = {k}",
            fontsize=15, fontweight="bold", color=TEXT_PRIMARY, pad=14,
        )
        ax.tick_params(colors="#64748b", labelsize=9)
        ax.grid(True, alpha=0.08, color="#000000")

        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.5)

        ax.legend(
            loc="upper right", framealpha=0.85,
            fontsize=9, edgecolor="#cbd5e1",
            facecolor="white",
        )

        self.fig.tight_layout()
        self.canvas.draw_idle()

        # ── Update sidebar widgets ───────────────────
        pred_label = "Classe A" if pred == 0 else "Classe B"
        pred_color = ACCENT_BLUE if pred == 0 else ACCENT_ORANGE
        confidence = float(proba[pred]) * 100

        self.pred_label_widget.configure(text=pred_label, text_color=pred_color)
        self.conf_label_widget.configure(text=f"Confiance: {confidence:.0f}%")
        self.radius_label_widget.configure(text=f"Rayon: {rayon:.3f}")

        self.vote_a_label.configure(text=f"Classe A: {votes_a}/{k}")
        self.vote_b_label.configure(text=f"Classe B: {votes_b}/{k}")
        self.vote_a_bar.set(votes_a / k if k > 0 else 0)
        self.vote_b_bar.set(votes_b / k if k > 0 else 0)


if __name__ == "__main__":
    app = KNNStudioApp()
    app.mainloop()
