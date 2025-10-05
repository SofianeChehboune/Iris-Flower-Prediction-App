# Iris Flower Prediction App 🌺

**👉 Live Demo: [Iris Flower Prediction App](https://iris-flower-prediction-app-jbheuyketpszewtwnoed4o.streamlit.app/)**

![Dataset Cover](dataset-cover.jpg)

An interactive web application built with Streamlit to predict the species of an Iris flower based on its sepal and petal measurements. This project demonstrates a complete machine learning workflow, from data exploration to model deployment in a user-friendly interface.

---

## ✨ Features

-   **Interactive Prediction:** Use sliders to input flower dimensions and get an instant species prediction.
-   **Prediction Probabilities:** Understand the model's confidence in its prediction.
-   **Advanced Data Visualization:**
    -   Interactive scatter plots of the Iris dataset using **Plotly**, with convex hulls to show species clusters.
    -   An alternative scatter plot built with **Bokeh** to showcase different styling and interactivity options.
    -   An interactive histogram to view the distribution of features.
    -   A correlation heatmap to understand relationships between features.
-   **Personalized UI:** A clean and customized interface with a developer profile section.

---

## 🛠️ Tech Stack

-   **Language:** Python
-   **Web Framework:** Streamlit
-   **Data Manipulation:** Pandas & NumPy
-   **Machine Learning:** Scikit-learn (RandomForestClassifier)
-   **Data Visualization:** Plotly, Bokeh, Matplotlib
-   **Image Handling:** Pillow

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   `pip` and `venv`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To launch the application, run the following command in your terminal:

```bash
streamlit run app.py
```

Your web browser will automatically open with the running application.

---

## 📖 Documentation

For a detailed, step-by-step guide on how this application was built, please see the [**Full Tutorial (in French)**](./Tutoriel-Iris-App.md).

---

## 👤 Author

-   **Sofiane Chehboune**
-   [LinkedIn Profile](https://www.linkedin.com/in/sofiane-chehboune-5b243766/)
