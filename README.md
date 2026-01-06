# Analisis Sentimen Ulasan Gojek

Web sederhana berbasis **Streamlit** untuk melakukan analisis sentimen terhadap ulasan aplikasi Gojek. Model ini menggunakan **Logistic Regression** dan **TF-IDF** untuk mengklasifikasikan ulasan menjadi **Negatif**, **Netral**, atau **Positif**.

## Fitur
- **Uji Kalimat**: Masukkan teks ulasan secara manual untuk melihat prediksi sentimennya.
- **Visualisasi Evaluasi**: Menampilkan Confusion Matrix.

## Struktur Project
- `app.py`: Main script untuk aplikasi Streamlit.
- `model_lr.pkl`: Model Logistic Regression yang sudah dilatih.
- `tfidf.pkl`: Vectorizer TF-IDF yang sudah dilatih.
- `gojek_reviews.csv`: Dataset ulasan Gojek (untuk referensi).
- `requirements.txt`: Daftar library Python yang dibutuhkan.

## Instalasi

1. **Pastikan Python terinstall** di komputer Anda.
2. **Install Dependencies**:
   Buka terminal/command prompt di folder project, lalu jalankan:
   ```bash
   pip install -r requirements.txt
   ```

## Access
Dapat diakses di link berikut : https://sentiment-analysis-fw.streamlit.app/

