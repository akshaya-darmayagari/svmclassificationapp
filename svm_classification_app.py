import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Page config
st.set_page_config(
    page_title="SVM Classification App",
    page_icon="🌸",
    layout="wide"
)

# Custom Theme CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #111827, #065f46, #022c22);
}
* {
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 {
    color: white !important;
}
[data-testid="stSidebar"]{
    background: #022c22;
}
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}
.stDataFrame {
    background-color: white !important;
    color: black !important;
}
.stDataFrame table {
    color: black !important;
}
.stButton > button {
    background: #059669;
    color: white !important;
    border-radius: 10px;
    padding: 8px 18px;
    border: none;
}
.stButton > button:hover {
    background: #047857;
}
.main-title{
    background: linear-gradient(90deg, #10b981, #059669);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
<h1>SVM Classification</h1>
<h3>Iris Species Identification System</h3>
</div>
""", unsafe_allow_html=True)

# Load Model & Scaler
model = pickle.load(open("model_svc.pkl", "rb"))
scaler = pickle.load(open("scaler_svc.pkl", "rb"))

# Load Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "EDA", "Model Evaluation", "Prediction"]
)

if page == "Dashboard":
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Features", len(df.columns) - 1)
    c4.metric("Classes", len(iris.target_names))

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    tab1, tab2 = st.tabs(["Distribution", "Scatter Visualizations"])

    with tab1:
        feature = st.selectbox("Select Feature", df.columns[:-1])
        fig, ax = plt.subplots()
        for i, name in enumerate(iris.target_names):
            ax.hist(df[df["Species"] == i][feature], label=name, alpha=0.6, bins=15)
        ax.set_title(feature)
        ax.legend()
        st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        x_axis = c1.selectbox("X Axis", df.columns[:-1], index=0)
        y_axis = c2.selectbox("Y Axis", df.columns[:-1], index=1)

        fig, ax = plt.subplots()
        for i, name in enumerate(iris.target_names):
            subset = df[df["Species"] == i]
            ax.scatter(subset[x_axis], subset[y_axis], label=name, edgecolors='k')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.legend()
        st.pyplot(fig)

elif page == "Model Evaluation":
    X = df.drop("Species", axis=1)
    y = df["Species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)
    pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="weighted")
    rec = recall_score(y_test, pred, average="weighted")
    f1 = f1_score(y_test, pred, average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", round(acc, 3))
    c2.metric("Precision (Weighted)", round(prec, 3))
    c3.metric("Recall (Weighted)", round(rec, 3))
    c4.metric("F1-Score (Weighted)", round(f1, 3))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(ax=ax, cmap=plt.cm.Greens)
    st.pyplot(fig)

elif page == "Prediction":
    st.subheader("Enter Flower Morphology Measurements")
    c1, c2 = st.columns(2)

    with c1:
        sepal_len = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
        sepal_wid = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    with c2:
        petal_len = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
        petal_wid = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

    if st.button("Predict Species"):
        data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0]

        species_name = iris.target_names[prediction].title()
        confidence = prob[prediction] * 100

        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #10b981, #059669);
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: white;">
            Predicted Class: {species_name}<br>
            Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()