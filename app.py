import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Analisis Sentimen ZZZ", layout="centered")

MODEL_PATH = "naive_bayes_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
NEGATION_WORDS = [
    "not", "no", "never", "none", "hardly", "barely", "scarcely", "fewer", "less",
    "don't", "doesn't", "didn't", "won't", "cannot",
]


def ensure_nltk_resources():
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


@st.cache_resource
def load_model_resources():
    ensure_nltk_resources()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


ensure_nltk_resources()
CUSTOM_STOP_WORDS = set(stopwords.words("english")).difference(NEGATION_WORDS)
model, vectorizer = load_model_resources()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = word_tokenize(text)
    processed_tokens = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if word in NEGATION_WORDS:
            processed_tokens.append(word)
            if i + 1 < len(tokens) and tokens[i + 1] not in CUSTOM_STOP_WORDS:
                processed_tokens.append(tokens[i + 1] + "_NEG")
                i += 1
        elif word not in CUSTOM_STOP_WORDS and word.isalpha():
            processed_tokens.append(word)
        i += 1

    return " ".join(processed_tokens)


def predict_sentiment(text):
    cleaned_text = clean_text(text)
    tfidf_vector = vectorizer.transform([cleaned_text])
    probabilities = model.predict_proba(tfidf_vector)[0]
    predicted_class_index = int(probabilities.argmax())
    predicted_label = model.classes_[predicted_class_index]
    confidence = probabilities[predicted_class_index] * 100
    return predicted_label, round(confidence, 2)


def main():
    st.title("Analisis Sentimen Review Game Zenless Zone Zero")
    st.write(
        "Aplikasi ini memprediksi apakah review game Zenless Zone Zero bersifat positif atau negatif "
        "berdasarkan model Naive Bayes dan TF-IDF yang sudah dilatih."
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
            st.success("Prediksi: Positif 😄")
        else:
            st.error("Prediksi: Negatif 😠")

        st.metric("Confidence Score", f"{confidence:.2f}%")


if __name__ == "__main__":
    main()
