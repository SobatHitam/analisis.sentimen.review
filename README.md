# Analisis Sentimen Review Game Zenless Zone Zero

Aplikasi web Streamlit untuk memprediksi sentimen review game Zenless Zone Zero menggunakan model Naive Bayes dan TF-IDF, mengikuti alur yang ada di notebook Colab.

## Fitur

- Input review melalui text area
- Preprocessing teks yang meniru notebook, termasuk handling negasi
- Prediksi sentimen positif/negatif
- Menampilkan confidence score dalam persen
- Sidebar berisi informasi model

## Instalasi

1. Pastikan Python 3.10+ sudah terinstal.
2. Buka terminal di folder project.
3. Install dependency:

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

Jalankan perintah berikut:

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser secara otomatis.

## Melatih Ulang Model

Jika Anda ingin melatih ulang model sesuai dengan notebook:

```bash
python train_model.py
```

## Deploy ke Streamlit Community Cloud

1. Upload project ini ke GitHub.
2. Buka https://streamlit.io/cloud.
3. Pilih "New app".
4. Pilih repository dan branch yang sesuai.
5. Tentukan file utama sebagai `app.py`.
6. Streamlit akan otomatis menginstall dependency dari `requirements.txt`.

Pastikan file berikut tersedia di repository:

- `app.py`
- `naive_bayes_model.joblib`
- `tfidf_vectorizer.joblib`
- `requirements.txt`
