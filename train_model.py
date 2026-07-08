import re
import os
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATASET_PATH = "review.csv"
MODEL_PATH = "naive_bayes_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"


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


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def assign_sentiment(score):
    return "positive" if score >= 4 else "negative"


def load_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
    else:
        df = pd.DataFrame(
            {
                "content": [
                    "This game is amazing and the graphics are beautiful",
                    "I love this game so much it is fun and exciting",
                    "The gameplay is smooth and the character designs are great",
                    "Really enjoyable experience with excellent sound design",
                    "This game is very fun and the events are amazing",
                    "The controls are responsive and the story is engaging",
                    "I had a great time playing this and the visuals are stunning",
                    "This game is awesome and the missions are entertaining",
                    "The game has many bugs and the performance is poor",
                    "I dislike this game because the controls are terrible",
                    "The update ruined the experience and the gameplay is frustrating",
                    "This game is boring and the content feels repetitive",
                    "Poor graphics and terrible optimization make it unenjoyable",
                    "I regret downloading this because it is full of issues",
                    "The story is weak and the game feels unbalanced",
                    "This game is not worth the money and the support is bad",
                    "The monetization is annoying and the lobby is laggy",
                ],
                "score": [5, 5, 4, 5, 4, 5, 5, 4, 1, 1, 2, 1, 2, 1, 2, 1, 2],
            }
        )

    required_columns = {"content", "score"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset harus memiliki kolom 'content' dan 'score'.")

    df = df[["content", "score"]].dropna().copy()
    df["sentiment"] = df["score"].apply(assign_sentiment)
    return df


ensure_nltk_resources()
stop_words = set(stopwords.words("english"))

if __name__ == "__main__":
    df = load_dataset()
    df["clean_content"] = df["content"].apply(clean_text)

    X = df["clean_content"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Model Naive Bayes berhasil dilatih. Akurasi: {accuracy:.2%}")
    print(f"Model disimpan ke {MODEL_PATH}")
    print(f"Vectorizer disimpan ke {VECTORIZER_PATH}")
