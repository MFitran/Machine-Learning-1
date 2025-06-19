# Bab 9: Teknik Pembelajaran Tanpa Pengawasan (Unsupervised Learning Techniques)

Bab 9 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Edisi ke-2) membahas konsep dan aplikasi pembelajaran tanpa pengawasan. Pembelajaran tanpa pengawasan adalah paradigma Machine Learning di mana model belajar dari data yang tidak berlabel, mencari pola dan struktur tersembunyi di dalamnya. Ini sangat kontras dengan pembelajaran terawasi (supervised learning) yang memerlukan data berlabel, dan pembelajaran penguatan (reinforcement learning) yang melibatkan agen yang belajar melalui interaksi lingkungan.

**Pentingnya Pembelajaran Tanpa Pengawasan:**
* Mayoritas data di dunia nyata tidak berlabel, dan proses pelabelan seringkali mahal atau tidak mungkin.
* Dapat digunakan untuk analisis data awal, membantu memahami struktur dataset baru.
* Mendeteksi anomali atau outlier dalam data.
* Sebagai langkah pra-pemrosesan (preprocessing) untuk reduksi dimensi atau transformasi fitur, yang dapat meningkatkan kinerja algoritma pembelajaran terawasi selanjutnya.

Bab ini secara khusus membahas tiga kategori utama teknik tanpa pengawasan:

## 1. Clustering (Pengelompokan)

Clustering adalah tugas untuk mengidentifikasi instance yang mirip dan mengelompokkannya ke dalam klaster.
* **Aplikasi Umum:** Segmentasi pelanggan, analisis data, reduksi dimensi, deteksi anomali, pembelajaran semiterawasi, sistem rekomendasi, mesin pencari, dan segmentasi gambar.
* **K-Means:** Algoritma clustering yang sederhana dan efisien yang mengidentifikasi $k$ centroid klaster dan menugaskan setiap instance ke centroid terdekatnya.
    * **Proses:** Inisialisasi centroid secara acak (atau menggunakan K-Means++), penugasan instance ke klaster terdekat, pembaruan centroid sebagai rata-rata klaster, dan iterasi hingga konvergensi.
    * **Evaluasi:** Inersia (jumlah kuadrat jarak instance ke centroid terdekatnya) dan skor siluet (mengukur seberapa baik instance cocok dengan klasternya dibandingkan dengan klaster lain) digunakan untuk mengevaluasi model dan menentukan $k$ yang optimal.
    * **Keterbatasan:** Sensitif terhadap inisialisasi centroid (diatasi dengan K-Means++), perlu menentukan $k$ di awal, dan kinerja buruk pada klaster dengan ukuran, kepadatan, atau bentuk non-sferis yang bervariasi. Penting untuk melakukan *feature scaling*.
    * **Aplikasi Praktis:** Segmentasi gambar (mengelompokkan piksel berdasarkan warna), pra-pemrosesan untuk meningkatkan akurasi model klasifikasi, dan pembelajaran semiterawasi (propagasi label).
* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Mendefinisikan klaster sebagai wilayah kontinu dengan kepadatan tinggi.
    * **Proses:** Mengidentifikasi instance inti (memiliki cukup tetangga dalam radius $\epsilon$), menghubungkan instance inti yang berdekatan untuk membentuk klaster, dan menandai instance lain sebagai anomali.
    * **Keuntungan:** Dapat menemukan klaster berbentuk arbitrer dan tahan terhadap outlier, tidak memerlukan $k$ sebagai input.
    * **Keterbatasan:** Dapat kesulitan jika kepadatan klaster sangat bervariasi dan kompleksitas memori bisa tinggi ($O(m^2)$) untuk `eps` besar.

## 2. Gaussian Mixtures (Campuran Gaussian)

Model Campuran Gaussian (GMM) adalah model probabilistik yang mengasumsikan bahwa instance dihasilkan dari campuran beberapa distribusi Gaussian yang parameternya tidak diketahui.
* **Karakteristik:** Setiap klaster dapat memiliki bentuk, ukuran, kepadatan, dan orientasi elipsoidal yang berbeda.
* **Algoritma Pelatihan:** Menggunakan algoritma Ekspektasi-Maksimisasi (EM) yang mengestimasi probabilitas instance milik setiap klaster (langkah ekspektasi) dan memperbarui parameter klaster (langkah maksimisasi).
* **Aplikasi:**
    * **Estimasi Kepadatan:** Mengestimasi fungsi kerapatan probabilitas (PDF) data.
    * **Clustering:** Mengelompokkan data berdasarkan distribusi Gaussian yang mendasarinya.
    * **Deteksi Anomali:** Menganggap instance di wilayah kepadatan rendah sebagai anomali.
* **Pemilihan Jumlah Klaster:** Tidak seperti K-Means, inersia atau skor siluet tidak reliable untuk GMM. Sebaliknya, kriteria informasi teoritis seperti BIC (Bayesian Information Criterion) dan AIC (Akaike Information Criterion) digunakan; model dengan nilai BIC/AIC terendah seringkali merupakan pilihan terbaik.
* **`BayesianGaussianMixture`:** Variasi GMM yang dapat secara otomatis menghilangkan klaster yang tidak perlu dengan memberikan bobot nol kepada mereka.
