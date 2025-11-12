import streamlit as st
import pandas as pd
import numpy as np

from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# ---------------------------------------------------------------
# 🔹 PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="📰",
    layout="wide",
)

# ---------------------------------------------------------------
# 🔹 HEADER / SIDEBAR
# ---------------------------------------------------------------
st.sidebar.title("🧠 Model Comparison Dashboard")
st.sidebar.markdown("Compare predictions & accuracy of multiple models on fake news data.")

# ---------------------------------------------------------------
# 🔹 LOAD MODELS & VECTORIZER (placeholder logic)
# ---------------------------------------------------------------
# Example: load pre-trained ML models (replace with your saved models)
# with open("vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)
# with open("logistic_model.pkl", "rb") as f:
#     logistic_model = pickle.load(f)
# with open("svm_model.pkl", "rb") as f:
#     svm_model = pickle.load(f)
# with open("dt_model.pkl", "rb") as f:
#     dt_model = pickle.load(f)

# For demonstration
vectorizer = TfidfVectorizer(max_features=5000)
logistic_model = LogisticRegression()
svm_model = LinearSVC()
dt_model = DecisionTreeClassifier()

# Placeholder deep learning model
# deep_model = load_model("fake_news_nn.h5")

# ---------------------------------------------------------------
# 🔹 LAYOUT: TWO TABS
# ---------------------------------------------------------------
tab1, tab2 = st.tabs(["📰 Classify News", "📊 Compare Model Performance"])

# ---------------------------------------------------------------
# TAB 1: CLASSIFICATION
# ---------------------------------------------------------------
with tab1:
    st.markdown("### 🧾 Enter a news headline or article")
    user_input = st.text_area("Enter your news text here:", height=200)

    if st.button("Classify"):
        if user_input.strip():
            # Vectorize user input
            sample = vectorizer.fit_transform([user_input])

            # Example predictions (replace with real ones)
            pred_log = np.random.choice(["Real", "Fake"])
            pred_svm = np.random.choice(["Real", "Fake"])
            pred_dt = np.random.choice(["Real", "Fake"])
            pred_dl = np.random.choice(["Real", "Fake"])

            # Display predictions
            st.subheader("🔍 Model Predictions:")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Logistic Regression", pred_log)
            col2.metric("SVM", pred_svm)
            col3.metric("Decision Tree", pred_dt)
            col4.metric("Deep Learning", pred_dl)

            # Final consensus (simple majority vote)
            results = [pred_log, pred_svm, pred_dt, pred_dl]
            final_output = max(set(results), key=results.count)
            st.success(f"✅ Final Verdict: **{final_output.upper()} News**")
        else:
            st.warning("Please enter some text to classify.")

# ---------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE COMPARISON
# ---------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Model Accuracy Comparison")
    # Placeholder accuracies
    data = {
        "Model": ["Logistic Regression", "SVM", "Decision Tree", "Deep Learning"],
        "Accuracy": [0.92, 0.89, 0.84, 0.95],
    }
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("Model"))

    st.markdown("### 🧾 Performance Summary")
    st.dataframe(df)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>Built with ❤️ using Streamlit & TensorFlow</p>",
    unsafe_allow_html=True,
)
