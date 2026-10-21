# ForestGuard AI

Dashboard prototipe untuk mendemonstrasikan pemantauan indikasi deforestasi dan klasifikasi kemungkinan penyebab menggunakan Machine Learning.

## Studi kasus
Pemantauan deforestasi manual dapat memakan waktu dan biaya. Citra satelit dan AI berpotensi membantu menyaring alert, memetakan perubahan vegetasi, dan mengelompokkan kemungkinan direct drivers untuk pemeriksaan lanjutan.

## Fitur
- Dashboard ringkasan alert dan luas indikasi.
- Filter kategori penyebab dan rentang tanggal.
- Grafik distribusi kategori dan luas indikasi.
- Visualisasi pola NDVI sebelum/sesudah.
- Klasifikasi demonstrasi menggunakan Random Forest.
- Ringkasan evaluasi model dan feature importance.
- Tabel hasil serta ekspor CSV.

## Teknologi
- Python
- Streamlit
- Pandas dan NumPy
- Plotly
- scikit-learn

## Menjalankan secara lokal
Pastikan Python sudah terpasang. Buka terminal pada folder proyek dan jalankan:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Lalu buka URL lokal yang ditampilkan di terminal, biasanya `http://localhost:8501`.

## Struktur proyek

```text
forestguard-ai/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Prinsip dan alur kerja
1. **Pengumpulan data:** versi demo menghasilkan data sintetis.
2. **Preprocessing:** menyusun fitur NDVI sebelum/sesudah, perubahan NDVI, jarak jalan, luas indikasi, dan indeks kebakaran.
3. **Pelatihan:** Random Forest dilatih dengan data sintetis dan pembagian train/test.
4. **Evaluasi:** menampilkan akurasi dan classification report pada data uji sintetis.
5. **Visualisasi:** grafik dan tabel interaktif membantu peninjauan alert.

## Batasan penting
**Data, koordinat, dan label pada proyek ini sepenuhnya sintetis.** Akurasi model yang tampil hanya mengukur perilaku model pada data simulasi dan tidak menunjukkan akurasi deteksi deforestasi dunia nyata. Aplikasi ini tidak mengunduh citra satelit, tidak menghasilkan alert operasional, dan tidak boleh digunakan sebagai bukti untuk penegakan hukum atau keputusan lingkungan.

Untuk pengembangan nyata, gunakan data satelit yang sesuai, masking awan dan preprocessing yang tervalidasi, label direct-driver dari sumber tepercaya, pembagian data berbasis lokasi/waktu untuk mencegah kebocoran data, serta validasi ahli atau lapangan.

## Lisensi
MIT (untuk contoh akademik; sesuaikan bila diperlukan).
