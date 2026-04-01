# KNN Expose

Projet simple de demonstration de l'algorithme KNN (K-Nearest Neighbors) avec un notebook Jupyter.

Version web interactive professionnelle incluse via Streamlit.

## Contenu

- `knn_expose.ipynb` : notebook principal
  - simulation visuelle KNN
  - parametre `k_user` modifiable
  - affichage des voisins utilises pour la prediction
- `app.py` : application web interactive (UI moderne)
- `requirements.txt` : dependances Python du projet

## Lancer le notebook

1. Ouvrir le dossier dans VS Code.
2. Ouvrir `knn_expose.ipynb`.
3. Executer la cellule de code.
4. Changer `k_user` (ex: 1, 3, 5, 7, 9) puis re-executer pour voir l'effet.

## Lancer la version web interactive

1. Installer les dependances:

  ```bash
  pip install -r requirements.txt
  ```

2. Demarrer l'application:

  ```bash
  streamlit run app.py
  ```

3. Ouvrir le lien local affiche dans le terminal (souvent http://localhost:8501).

## Technologies

- Python
- NumPy
- Matplotlib
- scikit-learn
- Streamlit
- Plotly

## Auteur

Ouahid Samrani
