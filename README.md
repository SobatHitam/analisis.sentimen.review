# Analisis Sentimen Review Game Zenless Zone Zero

## 1. Pendahuluan

- Latar belakang: Review pengguna menjadi salah satu sumber informasi penting dalam menilai kualitas sebuah game. Namun, jumlah review yang sangat banyak membuat proses analisis secara manual menjadi sulit dan memakan waktu.
- Dataset yang digunakan berasal dari review game Zenless Zone Zero yang tersedia pada Google Play Store. Sumber dataset: https://www.kaggle.com/datasets/bagush/zenless-zone-zero-mobile-game-reviews-google-play
- Masalah yang dihadapi pada dataset adalah banyaknya teks ulasan yang beragam, mengandung kata-kata informal, singkatan, serta sering kali tidak terstruktur. Hal ini membuat klasifikasi sentimen menjadi tantangan tersendiri.
- Knowledge/information/insight yang akan disajikan: proyek ini akan menunjukkan bagaimana teks review dapat diklasifikasikan menjadi sentimen positif atau negatif menggunakan pendekatan machine learning, khususnya Naive Bayes dan TF-IDF.

## 2. Tujuan

- Mengembangkan model machine learning yang mampu mengklasifikasikan review game menjadi sentimen positif atau negatif.
- Membantu memahami opini pengguna secara lebih cepat dan otomatis tanpa perlu membaca seluruh review satu per satu.
- Menilai performa model berdasarkan data yang tersedia dan mengevaluasi hasil prediksi yang dihasilkan.

## 3. Batasan

- Ruang lingkup dataset terbatas pada review game Zenless Zone Zero yang tersedia dalam dataset yang dipakai.
- Informasi yang digunakan adalah teks review dan rating atau skor review.
- Informasi yang tidak digunakan dalam model ini adalah data pengguna lain seperti username, timestamp, atau metadata tambahan yang tidak relevan untuk klasifikasi sentimen.
- Batasan tugas model adalah mengklasifikasikan review ke dalam dua kelas saja, yaitu positif dan negatif, sehingga ulasan netral tidak diproses secara eksplisit.

## 4. Metode/Teknik yang Digunakan

- Tahap pra-pemrosesan data:
  - Mengubah teks menjadi huruf kecil.
  - Menghapus URL, angka, dan tanda baca.
  - Melakukan tokenisasi kata.
  - Menghapus stopword umum.
  - Menangani negasi seperti "not good" agar makna sentimen lebih terjaga.
- Representasi fitur:
  - Menggunakan TF-IDF untuk mengubah teks menjadi vektor numerik.
- Pemodelan:
  - Menggunakan algoritma Multinomial Naive Bayes untuk klasifikasi sentimen.
- Alur kerja yang dilakukan:
  1. Mengambil data review.
  2. Membuat label sentimen berdasarkan skor review.
  3. Melakukan preprocessing teks.
  4. Membagi data menjadi data training dan testing.
  5. Melakukan vectorisasi dengan TF-IDF.
  6. Melatih model Naive Bayes.
  7. Menguji model dan mengevaluasi hasilnya.

## 5. Hasil dan Analisis

- Tahap 1: Loading dataset
  - Source code: lihat file train_model.py dan notebook Analisis Sentimen naive bayes.ipynb.
  - Hasil: dataset berhasil dibaca dan dipersiapkan untuk proses selanjutnya.
  - Analisis: data review yang cukup banyak memberikan cukup banyak pola untuk dipelajari model.

- Tahap 2: Labeling sentimen
  - Source code: bagian assign_sentiment pada train_model.py.
  - Hasil: review dengan skor 4–5 diberi label positive, sedangkan skor 1–3 diberi label negative.
  - Analisis: pendekatan ini sederhana namun efektif karena rating sering mencerminkan sentimen pengguna.

- Tahap 3: Preprocessing teks
  - Source code: fungsi clean_text pada train_model.py dan app.py.
  - Hasil: teks menjadi lebih bersih dan konsisten.
  - Analisis: preprocessing membantu mengurangi noise dan meningkatkan kualitas fitur yang dimasukkan ke model.

- Tahap 4: TF-IDF Vectorization
  - Source code: bagian TfidfVectorizer pada train_model.py.
  - Hasil: teks berhasil diubah menjadi representasi numerik.
  - Analisis: TF-IDF membantu menonjolkan kata-kata yang lebih penting dalam menentukan sentimen.

- Tahap 5: Pelatihan model
  - Source code: bagian model = MultinomialNB(alpha=0.5) pada train_model.py.
  - Hasil: model berhasil dilatih menggunakan data training.
  - Analisis: Naive Bayes bekerja dengan baik untuk data teks karena sifatnya yang cepat dan sederhana.

- Tahap 6: Evaluasi model
  - Source code: bagian evaluasi pada notebook dan train_model.py.
  - Hasil: model menghasilkan akurasi tertentu pada data testing.
  - Analisis: hasil evaluasi menunjukkan bahwa model mampu mengenali pola sentimen dari review yang belum pernah dilihat sebelumnya.

## 6. Cara Menjalankan Proyek

### Instalasi

```bash
pip install -r requirements.txt
```

### Menjalankan aplikasi Streamlit

```bash
streamlit run app.py
```

### Melatih ulang model

```bash
python train_model.py
```

## 7. Kesimpulan

- Model ini berhasil mengubah review teks menjadi informasi sentimen yang lebih terstruktur.
- Pendekatan preprocessing, TF-IDF, dan Naive Bayes terbukti cukup efektif untuk tugas klasifikasi sentimen review game.
- Aplikasi yang dibuat memungkinkan prediksi sentimen dilakukan secara cepat melalui antarmuka sederhana.
