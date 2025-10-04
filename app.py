import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import base64
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10_3
from bokeh.transform import factor_cmap, factor_mark
from streamlit_bokeh import streamlit_bokeh

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type! 
""")

image = Image.open('dataset-cover.jpg')
st.image(image, caption='Iris Flower Dataset-**Kaggle**-', use_container_width=True)

@st.cache_data
def get_image_as_base64(file_path):
    """Lit un fichier image et le retourne en tant que chaîne base64."""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

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

    # 4. Ajouter le bouton de téléchargement du tutoriel
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


# Afficher les paramètres d'entrée pour confirmation
st.subheader('User Input parameters')
st.write(df_input)

# Charger le dataset Iris
iris = pd.read_csv('Iris.csv')
X = iris.drop(['Id', 'Species'], axis=1) # Features
Y = iris['Species'] # Target

# Entraîner le modèle
clf = RandomForestClassifier()
clf.fit(X, Y)

# Faire la prédiction sur les entrées de l'utilisateur
prediction = clf.predict(df_input)
prediction_proba = clf.predict_proba(df_input)

# Afficher les classes possibles
st.subheader('Possible Flower Species (Classes)')
st.write(Y.unique())

# Afficher les résultats de la prédiction
st.subheader('Prediction')
st.success(f"The predicted flower is: **{prediction[0]}**")

st.subheader('Prediction Probability')
proba_df = pd.DataFrame(prediction_proba, columns=clf.classes_, index=['Probability'])
st.write(proba_df)

# Visualisation des données interactive avec Plotly
st.subheader('Interactive Data Visualization with Plotly')

# Scatter Plot interactif
st.write("Interactive Scatter plot of Petal Length vs Petal Width")

# Créer une figure vide
fig_scatter = go.Figure()

# Définir une palette de couleurs
colors = px.colors.qualitative.Plotly
species_list = iris['Species'].unique()
color_map = {species: colors[i] for i, species in enumerate(species_list)}

# Pour chaque espèce, calculer et dessiner l'enveloppe convexe
for i, species in enumerate(species_list):
    color = color_map[species]
    points = iris[iris['Species'] == species][['PetalLengthCm', 'PetalWidthCm']].values
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    fig_scatter.add_trace(go.Scatter(x=hull_points[:,0], y=hull_points[:,1], fill="toself", mode='lines',
                                     line_color=color, fillcolor=color, opacity=0.2,
                                     name=species, showlegend=False, hoverinfo='none'))

# Ensuite, ajouter les points du nuage de points PAR-DESSUS
for i, species in enumerate(species_list):
    color = color_map[species]
    species_df = iris[iris['Species'] == species]
    fig_scatter.add_trace(go.Scatter(x=species_df['PetalLengthCm'], y=species_df['PetalWidthCm'], mode='markers',
                                     marker=dict(color=color), name=species))

fig_scatter.update_layout(title="Petal Length vs. Petal Width with Convex Hulls")
st.plotly_chart(fig_scatter)

# Visualisation avec Bokeh
st.subheader("Interactive Scatter Plot with Bokeh")

# Préparer les données pour Bokeh
source = ColumnDataSource(iris)

# Définir les outils interactifs, y compris l'infobulle
hover = HoverTool(
    tooltips=[
        ("Espèce", "@Species"),
        ("Longueur Pétale", "@PetalLengthCm cm"),
        ("Largeur Pétale", "@PetalWidthCm cm"),
    ]
)

# Créer la palette de couleurs et les marqueurs basés sur les espèces
species_list = iris['Species'].unique()
markers = ['circle', 'square', 'triangle']
# Utiliser une palette adaptée aux thèmes sombres
color_map = factor_cmap('Species', palette=Category10_3, factors=species_list)
marker_map = factor_mark('Species', markers=markers, factors=species_list)

# Créer la figure Bokeh
p = figure(title="Iris Dataset - Petal Length vs. Width (Bokeh)",
           x_axis_label='Petal Length (cm)', y_axis_label='Petal Width (cm)',
           tools=[hover, "pan,wheel_zoom,box_zoom,reset,save"])

p.scatter(x='PetalLengthCm', y='PetalWidthCm', source=source, legend_field='Species',
          fill_color=color_map, marker=marker_map, size=10, alpha=0.8, line_color="white", line_width=0.5)

p.legend.location = "top_left"
p.legend.title = "Species"

# Afficher le graphique dans Streamlit
streamlit_bokeh(p, use_container_width=True)

# Histogramme interactif
st.write("Interactive Histogram of Petal Length")
fig_hist = px.histogram(iris, x="PetalLengthCm", color="Species",
                        title="Distribution of Petal Length",
                        color_discrete_sequence=px.colors.qualitative.Vivid)
st.plotly_chart(fig_hist)

# Heatmap de corrélation
st.write("Correlation Heatmap of Iris Features")
# Nous devons sélectionner uniquement les colonnes numériques pour la corrélation
numeric_iris = iris.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_iris.corr()
fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Heatmap of Iris Features",
                        color_continuous_scale='Viridis')
st.plotly_chart(fig_heatmap)
