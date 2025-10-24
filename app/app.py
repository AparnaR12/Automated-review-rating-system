import streamlit as st
import joblib

# 🎯 Load models and vectorizers
model_a = joblib.load("Models/ModelA_balanced.pkl")
model_b = joblib.load("Models/ModelB_imbalanced.pkl")

vectorizer_a = joblib.load("Models/vectorizer_bal.pkl")
vectorizer_b = joblib.load("Models/vectorizer_unbal.pkl")

# 🌟 Page setup
st.set_page_config(page_title="Rating Predictor", layout="wide")

# 🧭 Custom CSS to center content
st.markdown("""
    <style>
    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    h1, h2, h3, h4, h5, h6 {
        text-align: center;
    }
    .stTextArea, .stButton {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# 🌟 Title
st.title("⭐ Review Sentiment Testing")
st.markdown("<h4> Enter</b> the product review : </h4>", unsafe_allow_html=True)

# ✍️ Centered Input
review_text = st.text_area("Enter a Review:", height=100, key="review", label_visibility="collapsed")

# Center the button using Streamlit columns
col_center = st.columns([2, 1, 2])
with col_center[1]:
    predict_button = st.button("🔍 Predict")

# ⚙️ Prediction Logic
if predict_button:
    if review_text.strip() == "":
        st.warning("⚠️ Please enter a review before predicting.")
    else:
        X_a = vectorizer_a.transform([review_text])
        X_b = vectorizer_b.transform([review_text])

        pred_a = model_a.predict(X_a)[0]
        pred_b = model_b.predict(X_b)[0]

        # 🧮 Confidence Scores
        try:
            conf_a = model_a.predict_proba(X_a).max()
            conf_b = model_b.predict_proba(X_b).max()
        except:
            conf_a = conf_b = None

        # 📊 Centered Output Columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📘 Model A (Balanced)")
            st.metric(label="Predicted Rating", value=f"{pred_a}/5")
            if conf_a:
                st.caption(f"Confidence: {conf_a:.2f}")

        with col2:
            st.subheader("📗 Model B (Imbalanced)")
            st.metric(label="Predicted Rating", value=f"{pred_b}/5")
            if conf_b:
                st.caption(f"Confidence: {conf_b:.2f}")

        st.success("✅ Predictions generated successfully!")
