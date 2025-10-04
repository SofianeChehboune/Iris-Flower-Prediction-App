# Tutoriel : Votre Première Application de Machine Learning avec Streamlit

Bienvenue dans ce guide pour débutants ! Ensemble, nous allons construire pas à pas une application web interactive. L'objectif est simple mais puissant : créer une application capable de prédire l'espèce d'une fleur d'Iris simplement en ajustant quelques curseurs.

Pensez-y comme à un jeu : vous donnez les dimensions de la fleur, et l'ordinateur, grâce au *machine learning*, devine de quelle espèce il s'agit.

**Les Outils du Développeur :**

*   **Python** : Notre langage de programmation.
*   **Streamlit** : La baguette magique pour transformer un script Python en une application web, sans toucher au HTML/CSS complexe.
*   **Pandas** : L'expert de la manipulation de données. Il nous aidera à lire et organiser nos informations.
*   **Scikit-learn** : La boîte à outils du machine learning. Elle contient le "cerveau" de notre application.
*   **Plotly & Bokeh** : Nos artistes pour créer de superbes graphiques interactifs.

---

## Étape 1 : Les Fondations (Les Imports)

**Le Concept :** Chaque projet Python commence par appeler ses outils. Au lieu de les importer au fur et à mesure, nous allons tous les déclarer au début. C'est une bonne pratique qui rend le code plus lisible.

**Le Code :**
```python
# --- Les Imports Principaux ---
import streamlit as st
import pandas as pd
from PIL import Image
import base64

# --- Les Outils de Machine Learning ---
from sklearn.ensemble import RandomForestClassifier

# --- Les Bibliothèques de Visualisation ---
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull # Un outil mathématique pour Plotly
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10_3 # La palette de couleurs pour Bokeh
from bokeh.transform import factor_cmap, factor_mark
from streamlit_bokeh import streamlit_bokeh
```

**Explication Simple :**
On dit simplement à Python : "Hey, pour ce projet, je vais avoir besoin de tous ces outils. Prépare-les pour moi !". Chaque ligne `import` charge une fonctionnalité spécifique dont nous aurons besoin plus tard.

---

## Étape 2 : La Page d'Accueil

**Le Concept :** Donnons un titre et une belle image de couverture à notre application pour la rendre accueillante.

**Le Code :**
```python
st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type! 
""")

image = Image.open('dataset-cover.jpg')
st.image(image, caption='Iris Flower Dataset-**Kaggle**-', use_container_width=True)
```
**Explication Simple :**
*   `st.write()` est la commande la plus simple de Streamlit pour écrire du texte. Le `#` crée un grand titre.
*   On utilise la bibliothèque `PIL` (via `Image.open`) pour ouvrir notre fichier image, puis `st.image()` pour l'afficher en pleine largeur.

---

## Étape 3 : Le Panneau de Contrôle (La Sidebar)

**Le Concept :** Une bonne application est intuitive. Nous allons placer tous les contrôles dans une barre latérale pour ne pas encombrer la page principale. Cette barre contiendra les curseurs pour que l'utilisateur entre les dimensions de la fleur, ainsi qu'une section "Auteur" pour la touche personnelle.

**Le Code :**
```python
# --- Sidebar ---
with st.sidebar:
    # 1. Afficher la bannière en premier
    st.image("banniere.png", use_container_width=True)
    
    # 2. Ajouter les entrées utilisateur
    st.header('User Input Parameters')
    
    def user_input_features():
        sepal_length = st.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'SepalLengthCm': sepal_length,
                'SepalWidthCm': sepal_width,
                'PetalLengthCm': petal_length,
                'PetalWidthCm': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features

    df_input = user_input_features()

    # 3. Ajouter la section Auteur
    st.markdown("---")
    # ... (Le code pour la section Auteur, que nous avons déjà écrit)
```

**Explication Simple :**
*   `with st.sidebar:` : Tout ce qui est indenté sous cette ligne apparaîtra dans la barre latérale.
*   `st.slider()`: Crée un curseur interactif. C'est beaucoup plus amusant pour l'utilisateur que de taper des chiffres !
*   `pd.DataFrame()`: On organise les 4 valeurs des curseurs dans un petit tableau (un DataFrame). C'est le format que notre "cerveau" de machine learning comprend.

---

## Étape 4 : Le Cerveau (Le Machine Learning)

**Le Concept :** C'est le cœur de l'application.
1.  **Apprentissage :** On donne à notre modèle (`RandomForestClassifier`) 150 exemples de fleurs du fichier `Iris.csv` pour qu'il apprenne à les reconnaître.
2.  **Prédiction :** On lui montre ensuite les mesures que l'utilisateur a choisies avec les curseurs, et on lui demande "À quelle fleur cela te fait-il penser ?".

**Le Code :**
```python
# Charger le dataset Iris pour l'entraînement
iris = pd.read_csv('Iris.csv')
X = iris.drop(['Id', 'Species'], axis=1) # Les mesures
Y = iris['Species'] # L'espèce (la réponse)

# Entraîner le modèle
clf = RandomForestClassifier()
clf.fit(X, Y) # Le moment de l'apprentissage !

# Faire la prédiction sur les données de l'utilisateur
prediction = clf.predict(df_input)
prediction_proba = clf.predict_proba(df_input)
```

**Explication Simple :**
*   `clf.fit(X, Y)` : C'est la ligne magique. `fit` veut dire "ajuster". Le modèle s'ajuste aux données, il apprend les relations entre les mesures (`X`) et l'espèce (`Y`).
*   `clf.predict(df_input)` : Le modèle regarde les données de l'utilisateur et donne son verdict final.
*   `clf.predict_proba(df_input)` : Le modèle nous dit à quel point il est confiant. "Je suis sûr à 95% que c'est un Iris-setosa".

---

## Étape 5 : Le Verdict (Afficher les Résultats)

**Le Concept :** Montrons à l'utilisateur ce que le modèle a deviné, de manière claire et encourageante.

**Le Code :**
```python
st.subheader('Prediction')
st.success(f"The predicted flower is: **{prediction[0]}**")

st.subheader('Prediction Probability')
proba_df = pd.DataFrame(prediction_proba, columns=clf.classes_, index=['Probability'])
st.write(proba_df)
```
**Explication Simple :**
*   `st.success()`: Affiche la prédiction dans une boîte verte, pour un effet positif.
*   On affiche aussi le tableau des probabilités pour que l'utilisateur curieux puisse voir le "raisonnement" du modèle.

---

## Étape 6 : La Magie Visuelle (Les Graphiques)

**Le Concept :** Les chiffres, c'est bien, mais les images, c'est mieux ! Les graphiques nous aident à "voir" les données et à comprendre pourquoi le modèle prend ses décisions. Nous allons créer plusieurs graphiques interactifs.

### Partie A : Le Nuage de Points avec Plotly

**Le But :** Visualiser si les espèces de fleurs sont faciles à séparer en regardant simplement la longueur et la largeur de leurs pétales. Nous allons même dessiner une "enveloppe" autour de chaque groupe d'espèces.

**Le Code :**
```python
# (Code pour le fig_scatter de Plotly, qui est déjà dans votre fichier)
st.plotly_chart(fig_scatter)
```

### Partie B : La Comparaison avec Bokeh

**Le But :** Montrer qu'il existe plusieurs "artistes" pour dessiner des graphiques. On place ce graphique juste en dessous de celui de Plotly pour comparer les styles. On va utiliser une palette de couleurs différente (`Category10_3`) et des formes différentes (carré, cercle, triangle) pour chaque espèce.

**Le Code :**
```python
st.subheader("Interactive Scatter Plot with Bokeh")

# Préparer les données pour Bokeh
source = ColumnDataSource(iris)

# ... (Code pour configurer le hover, les marqueurs, etc.)

# Utiliser une palette de couleurs prédéfinie et attractive
color_map = factor_cmap('Species', palette=Category10_3, factors=species_list)
marker_map = factor_mark('Species', markers=markers, factors=species_list)

# ... (Code pour créer la figure et le scatter plot Bokeh)

# Afficher le graphique dans Streamlit
streamlit_bokeh(p, use_container_width=True)
```
**Explication Simple :**
*   `factor_cmap`: Une fonction magique de Bokeh qui associe une couleur de la palette `Category10_3` à chaque espèce de fleur.
*   `factor_mark`: Fait la même chose, mais avec des formes.
*   `streamlit_bokeh(p)`: La commande spéciale pour afficher notre graphique Bokeh.

### Partie C : Les Autres Graphiques (Histogramme & Heatmap)

**Le But :** L'histogramme nous montre la distribution des données (par exemple, y a-t-il beaucoup de petites ou de grandes pétales ?). La carte de chaleur (heatmap) est un outil d'expert qui montre quelles caractéristiques sont les plus liées entre elles.

**Le Code :**
```python
# (Code pour l'histogramme et la heatmap de Plotly)
```

---

## Étape 7 : Personnalisation Avancée (Couleurs)

Notre application est fonctionnelle, mais rendons-la plus jolie ! Nous allons personnaliser les couleurs de nos graphiques et du bouton de téléchargement.

### Partie A : Changer les Couleurs des Graphiques Plotly

**Le But :** Remplacer les couleurs par défaut de l'histogramme et de la carte de chaleur pour mieux correspondre à notre style.

**Le Code :**

```python
# Histogramme interactif avec une nouvelle palette
st.write("Interactive Histogram of Petal Length")
fig_hist = px.histogram(iris, x="PetalLengthCm", color="Species",
                        title="Distribution of Petal Length",
                        color_discrete_sequence=px.colors.qualitative.Vivid) # <-- On ajoute cette ligne
st.plotly_chart(fig_hist)

# Heatmap de corrélation avec une nouvelle échelle de couleurs
st.write("Correlation Heatmap of Iris Features")
numeric_iris = iris.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_iris.corr()
fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Heatmap of Iris Features",
                        color_continuous_scale='Viridis') # <-- On modifie cette ligne
st.plotly_chart(fig_heatmap)
```

**Explication Simple :**
*   `color_discrete_sequence` : Pour l'histogramme, on passe une liste de couleurs prédéfinies de Plotly (`px.colors.qualitative.Vivid`).
*   `color_continuous_scale` : Pour la carte de chaleur, on utilise une échelle de couleurs nommée (`'Viridis'`) pour un dégradé plus agréable.

### Partie B : Donner du Style au Bouton de Téléchargement

**Le But :** Le bouton de téléchargement par défaut est un peu simple. Donnons-lui une couleur de fond verte pour le faire ressortir.

**Le Code :**
On utilise une astuce en injectant un peu de code CSS juste avant de créer le bouton.

```python
# --- Dans la Sidebar ---

# ... (code précédent de la sidebar) ...

# 4. Ajouter le bouton de téléchargement du tutoriel
st.markdown("---")
st.markdown("### 📄 Tutoriel de l'Application")

# On injecte le style CSS pour le bouton
st.markdown("""
<style>
    div.stDownloadButton button {
        background-color: green;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

with open("Tutoriel-Iris-App.md", "r", encoding="utf-8") as f:
    tutorial_content = f.read()
st.download_button(
    label="📥 Télécharger le Tutoriel (MD)",
    data=tutorial_content,
    file_name="Tutoriel-Iris-App.md",
    mime="text/markdown",
)
```

**Explication Simple :**
*   `st.markdown("...")`: Cette commande nous permet d'écrire du HTML/CSS directement dans notre application.
*   On cible spécifiquement le bouton créé par `st.download_button` (`div.stDownloadButton button`) et on lui applique une couleur de fond (`background-color`) verte et un texte blanc (`color`).

---

## Étape 8 : Le Code Complet (Mis à Jour)

Hermès le code complet de `app.py` avec toutes les améliorations que nous avons apportées.

```python
# --- IMPORTS ---
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import base64
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10_3
from bokeh.transform import factor_cmap, factor_mark
from streamlit_bokeh import streamlit_bokeh

# --- PAGE PRINCIPALE ---
st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type! 
""")

image = Image.open('dataset-cover.jpg')
st.image(image, caption='Iris Flower Dataset-**Kaggle**-', use_container_width=True)

# --- FONCTION HELPER (pour la section auteur) ---
@st.cache_data
def get_image_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("banniere.png", use_container_width=True)
    st.header('User Input Parameters')
    
    def user_input_features():
        sepal_length = st.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'SepalLengthCm': sepal_length,
                'SepalWidthCm': sepal_width,
                'PetalLengthCm': petal_length,
                'PetalWidthCm': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features

    df_input = user_input_features()

    st.markdown("---")
    id_photo_b64 = get_image_as_base64("ID.jpg")
    linkedin_icon_b64 = get_image_as_base64("linkedin.png")

    if id_photo_b64 and linkedin_icon_b64:
        st.markdown("### 💻 Développé par")
        st.markdown(f'''
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <img src="data:image/jpg;base64,{id_photo_b64}" alt="ID Photo" style="width: 70px; height: 70px; border-radius: 50%; margin-right: 15px; object-fit: cover;">
            <a href="https://www.linkedin.com/in/sofiane-chehboune-5b243766/" target="_blank" style="display: inline-block; text-decoration: none; background-color: #0077B5; color: white; padding: 8px 12px; border-radius: 4px; font-weight: bold;">
                <img src="data:image/png;base64,{linkedin_icon_b64}" alt="LinkedIn Logo" style="height: 20px; vertical-align: middle; margin-right: 8px;">
                Sofiane Chehboune
            </a>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Tutoriel de l'Application")
    st.markdown("""
<style>
    div.stDownloadButton button {
        background-color: green;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
    with open("Tutoriel-Iris-App.md", "r", encoding="utf-8") as f:
        tutorial_content = f.read()
    st.download_button(
        label="📥 Télécharger le Tutoriel (MD)",
        data=tutorial_content,
        file_name="Tutoriel-Iris-App.md",
        mime="text/markdown",
    )

# --- MACHINE LEARNING ---
st.subheader('User Input parameters')
st.write(df_input)

iris = pd.read_csv('Iris.csv')
X = iris.drop(['Id', 'Species'], axis=1)
Y = iris['Species']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df_input)
prediction_proba = clf.predict_proba(df_input)

st.subheader('Possible Flower Species (Classes)')
st.write(Y.unique())

st.subheader('Prediction')
st.success(f"The predicted flower is: **{prediction[0]}**")

st.subheader('Prediction Probability')
proba_df = pd.DataFrame(prediction_proba, columns=clf.classes_, index=['Probability'])
st.write(proba_df)

# --- VISUALISATIONS ---
st.subheader('Interactive Data Visualization with Plotly')

# Scatter Plot (Plotly)
st.write("Interactive Scatter plot of Petal Length vs Petal Width")
fig_scatter = go.Figure()
species_list = iris['Species'].unique()
colors = px.colors.qualitative.Plotly
color_map = {species: colors[i] for i, species in enumerate(species_list)}
for i, species in enumerate(species_list):
    color = color_map[species]
    points = iris[iris['Species'] == species][['PetalLengthCm', 'PetalWidthCm']].values
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    fig_scatter.add_trace(go.Scatter(x=hull_points[:,0], y=hull_points[:,1], fill="toself", mode='lines',
                                     line_color=color, fillcolor=color, opacity=0.2,
                                     name=species, showlegend=False, hoverinfo='none'))
for i, species in enumerate(species_list):
    color = color_map[species]
    species_df = iris[iris['Species'] == species]
    fig_scatter.add_trace(go.Scatter(x=species_df['PetalLengthCm'], y=species_df['PetalWidthCm'], mode='markers',
                                     marker=dict(color=color), name=species))
fig_scatter.update_layout(title="Petal Length vs. Petal Width with Convex Hulls")
st.plotly_chart(fig_scatter)

# Scatter Plot (Bokeh)
st.subheader("Interactive Scatter Plot with Bokeh")
source = ColumnDataSource(iris)
hover = HoverTool(tooltips=[("Espèce", "@Species"), ("Longueur Pétale", "@PetalLengthCm cm"), ("Largeur Pétale", "@PetalWidthCm cm")])
species_list = iris['Species'].unique()
markers = ['circle', 'square', 'triangle']
color_map = factor_cmap('Species', palette=Category10_3, factors=species_list)
marker_map = factor_mark('Species', markers=markers, factors=species_list)
p = figure(title="Iris Dataset - Petal Length vs. Width (Bokeh)",
           x_axis_label='Petal Length (cm)', y_axis_label='Petal Width (cm)',
           tools=[hover, "pan,wheel_zoom,box_zoom,reset,save"])
p.scatter(x='PetalLengthCm', y='PetalWidthCm', source=source, legend_field='Species',
          fill_color=color_map, marker=marker_map, size=10, alpha=0.8, line_color="white", line_width=0.5)
p.legend.location = "top_left"
p.legend.title = "Species"
streamlit_bokeh(p, use_container_width=True)

# Histogramme (Plotly)
st.write("Interactive Histogram of Petal Length")
fig_hist = px.histogram(iris, x="PetalLengthCm", color="Species",
                        title="Distribution of Petal Length",
                        color_discrete_sequence=px.colors.qualitative.Vivid)
st.plotly_chart(fig_hist)

# Heatmap (Plotly)
st.write("Correlation Heatmap of Iris Features")
numeric_iris = iris.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_iris.corr()
fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Heatmap of Iris Features",
                        color_continuous_scale='Viridis')
st.plotly_chart(fig_heatmap)

```

---

## Étape 9 : Comment Lancer l'Application

1.  Assurez-vous que tous vos fichiers (`app.py`, `Iris.csv`, `dataset-cover.jpg`, `banniere.png`, `ID.jpg`, `linkedin.png`) sont dans le même dossier.
2.  Ouvrez un terminal (comme `cmd` ou `PowerShell`).
3.  Naviguez jusqu'au dossier de votre projet avec la commande `cd`.
4.  Lancez la commande magique :
    ```bash
    streamlit run app.py
    ```

Et voilà ! Votre navigateur devrait s'ouvrir et afficher votre magnifique application. Félicitations !
