import app

sample_review = "this game is amazing and fun"
label, confidence = app.predict_sentiment(sample_review)
print(f"Review: {sample_review}")
print(f"Prediksi: {label} ({confidence:.2f}%)")
