# 🧠 KNN Studio

**Plateforme interactive de démonstration et d'exploration de l'algorithme K-Nearest Neighbors**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](#)

[![Demo Live](https://img.shields.io/badge/🚀_Demo_Live-knn--yo.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://knn-yo.streamlit.app)

> Projet réalisé dans le cadre du module **Intelligence Artificielle** — GL S6

---

## 📋 Table des matières

- [Aperçu](#-aperçu)
- [Fonctionnalités](#-fonctionnalités)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Technologies](#-technologies)
- [Auteurs](#-auteurs)

---

## 🔍 Aperçu

**KNN Studio** est une suite complète d'outils pédagogiques pour comprendre, visualiser et expérimenter l'algorithme KNN (K-Nearest Neighbors). Le projet propose **quatre interfaces** complémentaires :

- 📓 **Notebook Jupyter** — Exposé théorique avec simulations visuelles (Matplotlib)
- 🌐 **Application Web** — Simulation interactive avec UI glassmorphism (Streamlit + Plotly)
- 🩺 **Web — Cas Réels** — Classification sur datasets réels : médical, Iris, immobilier (scikit-learn)
- 🖥️ **Application Desktop** — Interface native standalone (CustomTkinter + Matplotlib)

---

## ✨ Fonctionnalités

### 🎨 Interface Premium

- Design **glassmorphism** sombre avec animations fluides
- Typographie moderne (Inter, JetBrains Mono)
- Micro-animations et effets de survol
- Interface 100% responsive

### 📊 Visualisation Avancée

- **Frontière de décision** en temps réel
- **Carte de confiance** avec dégradé de probabilités
- **Zone de recherche** circulaire (rayon adaptatif)
- Lignes de connexion vers les k plus proches voisins
- Graphique donut pour la répartition des votes

### 🧪 Analyse Statistique (Cas Réels)

- **3 datasets réels** : Triage Métabolique, Iris, Immobilier
- **Matrice de confusion** interactive
- **Sensibilité au paramètre K** (confiance & accuracy)
- **Comparaison KNN vs Règle simple** (baseline)
- **Cross-validation** (5-fold accuracy)
- Choix de la **métrique de distance** (Euclidienne, Manhattan, Minkowski)
- **Pondération** uniforme ou par distance

### 🖥️ Mode Desktop

- Application standalone sans navigateur
- Interface CustomTkinter avec thème sombre
- Sliders interactifs en temps réel
- Cas pratique médical : IMC + Glycémie

---

## 📁 Structure du projet

```
knn/
├── knn_expose.ipynb        # 📓 Notebook — exposé théorique et simulations
├── app.py                  # 🌐 App web interactive (données synthétiques)
├── app_web_real_case.py    # 🩺 App web — cas réels avec statistiques
├── knn_desktop.py          # 🖥️ App desktop (CustomTkinter)
├── requirements.txt        # 📦 Dépendances Python
└── README.md               # 📄 Documentation du projet
```

---

## 🚀 Installation

### Prérequis

- **Python 3.9+** installé sur votre machine
- **pip** (gestionnaire de paquets Python)

### Étapes

**1. Cloner le dépôt :**

```bash
git clone https://github.com/SamraniOuahid/knn.git
cd knn
```

**2. Installer les dépendances :**

```bash
pip install -r requirements.txt
```

---

## 🎮 Utilisation

### 📓 Notebook Jupyter

1. Ouvrir le dossier dans **VS Code** ou **Jupyter Lab**
2. Ouvrir `knn_expose.ipynb`
3. Exécuter les cellules séquentiellement
4. Modifier le paramètre `k_user` (ex : 1, 3, 5, 7, 9) et ré-exécuter pour observer l'effet

### 🌐 Application Web — Simulation Interactive

```bash
streamlit run app.py
```

Ouvrir le lien local affiché dans le terminal (par défaut : http://localhost:8501)

### 🩺 Application Web — Cas Réels

🔗 **Demo en ligne :** [https://knn-yo.streamlit.app](https://knn-yo.streamlit.app)

Ou en local :

```bash
streamlit run app_web_real_case.py
```

> Sélectionner un dataset dans la barre latérale, ajuster les paramètres KNN, et explorer les statistiques avancées.

### 🖥️ Application Desktop

```bash
python knn_desktop.py
```

> ⚠️ Nécessite un environnement graphique (pas compatible avec les serveurs headless).

---

## 🛠️ Technologies

| Catégorie | Technologies |
|---|---|
| **Langage** | Python 3.9+ |
| **Machine Learning** | scikit-learn (KNeighborsClassifier, cross-validation, StandardScaler) |
| **Web Framework** | Streamlit |
| **Graphiques interactifs** | Plotly, Matplotlib |
| **Interface Desktop** | CustomTkinter |
| **Calcul numérique** | NumPy, Pandas |

---

## 👥 Auteurs

- **Ouahid Samrani**
- **Yassir Mrabti**

---

> Construit avec ❤️ pour le module Intelligence Artificielle — GL S6
