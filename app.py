import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load model & vectorizer
with open("model_lr.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Label mapping
label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}

# Page config
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("📊 Analisis Sentimen Ulasan Gojek")
st.caption("Model: Logistic Regression (Class Weight: Balanced)")

# ==== METRICS ====
col1, col2, col3 = st.columns(3)
col1.metric("Model", "Logistic Regression")
col2.metric("Vectorizer", "TF-IDF")
col3.metric("Class Weight", "Balanced")

st.divider()

# ==== INPUT TEXT ====
st.subheader("📝 Uji Kalimat")
text = st.text_area("Masukkan ulasan:")

if st.button("Hasil Sentimen"):
    if text.strip() == "":
        st.warning("Teks tidak boleh kosong")
    else:
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0]
        st.success(f"Hasil Sentimen: **{pred}**")

st.divider()

# ==== OPTIONAL: EVALUATION RESULT ====
st.subheader("📈 Evaluasi Model")

y_test = [0, 2, 2, 1, 0, 2, 1]
y_pred = [0, 2, 2, 0, 0, 2, 1]

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_map.values(),
            yticklabels=label_map.values())
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
