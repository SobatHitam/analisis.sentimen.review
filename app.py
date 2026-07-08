import re
import pandas as pd
import nltk
import joblib
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Analisis Sentimen ZZZ", layout="centered")


@st.cache_resource
def load_model_resources():
    """Memuat model dan vectorizer yang sudah disimpan ke file joblib."""
    ensure_nltk_resources()

    model = joblib.load("naive_bayes_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


def ensure_nltk_resources():
    """Mengunduh resource NLTK yang diperlukan jika belum tersedia."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


ensure_nltk_resources()
stop_words = set(stopwords.words("english"))
model, vectorizer = load_model_resources()


def clean_text(text):
    """Melakukan preprocessing teks agar identik dengan notebook."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def predict_sentiment(text):
    """Melakukan preprocessing, transform TF-IDF, lalu prediksi sentimen."""
    cleaned_text = clean_text(text)
    tfidf_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_vector)[0]

    class_names = list(model.classes_)
    class_index = class_names.index(prediction)
    confidence = model.predict_proba(tfidf_vector)[0][class_index] * 100

    return prediction, round(confidence, 2)


def main():
    st.title("Analisis Sentimen Review Game Zenless Zone Zero")
    st.write(
        "Aplikasi ini memprediksi apakah review game Zenless Zone Zero bersifat positif atau negatif "
        "berdasarkan model Naive Bayes yang sudah dilatih."
    )

    with st.sidebar:
        st.header("Informasi Model")
        st.write(f"- Model: {type(model).__name__}")
        st.write("- Feature Extraction: TF-IDF")
        st.write("- Dataset: Review Game Zenless Zone Zero")

    review_text = st.text_area(
        "Masukkan review",
        height=180,
        placeholder="Contoh: Game ini sangat seru dan grafisnya keren!",
    )

    if st.button("Prediksi Sentimen"):
        if not review_text.strip():
            st.warning("Silakan masukkan teks review sebelum melakukan prediksi.")
            return

        label, confidence = predict_sentiment(review_text)

        if label == "positive":
            st.success(f"Prediksi: Positif 😄")
        else:
            st.error(f"Prediksi: Negatif 😠")

        st.metric("Confidence Score", f"{confidence:.2f}%")


if __name__ == "__main__":
    main()
