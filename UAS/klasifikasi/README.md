
# Analisis Klasifikasi dengan Deep Learning (TensorFlow & PyTorch)

Dokumen ini merangkum proses *end-to-end* untuk membangun, melatih, dan mengevaluasi model *Deep Learning* untuk tugas klasifikasi, dengan fokus pada deteksi penipuan. Dua *framework* utama, TensorFlow (dengan Keras) dan PyTorch, digunakan untuk demonstrasi.

## 1. Pipeline End-to-End untuk Model Deep Learning

### a. Pengumpulan & Pembersihan Data (Pandas)

**Tujuan:** Memuat data mentah dan mengubahnya menjadi format yang bersih serta siap untuk diproses lebih lanjut oleh model *Machine Learning*.

**Langkah-langkah Pembersihan Data:**
1.  **Memuat Dataset:** Dataset `KlasifikasiUTS.csv` dimuat. Baris pertama dari file CSV diidentifikasi dan digunakan sebagai header kolom untuk DataFrame.
2.  **Identifikasi Kolom Numerik:** Kolom 'Time', 'Amount', dan semua kolom berlabel 'V' (V1-V28) diidentifikasi sebagai fitur numerik yang relevan untuk model.
3.  **Konversi Tipe Data:** Semua kolom yang diidentifikasi sebagai numerik dikonversi ke tipe data numerik (float). Proses ini juga menangani pemisah desimal yang mungkin menggunakan koma (`,`) dengan menggantinya menjadi titik (`.`). Nilai yang tidak dapat dikonversi akan menjadi `NaN`.
4.  **Penanganan Missing Value:** Setiap nilai `NaN` yang dihasilkan dari proses konversi diisi menggunakan nilai median dari kolom yang bersangkutan. Penggunaan median lebih disukai karena lebih *robust* terhadap *outlier* yang umum dalam data keuangan.
5.  **Konversi Kolom Target:** Kolom 'Class' (kolom terakhir) yang merupakan target klasifikasi, dikonversi secara eksplisit ke tipe data integer.

### b. Feature Engineering

**Tujuan:** Menyiapkan fitur input agar lebih sesuai untuk model *Deep Learning*, terutama melalui penskalaan.

**Langkah-langkah Feature Engineering:**
1.  **Pemisahan Fitur dan Target:** DataFrame dibagi menjadi fitur (`X`) dan variabel target (`y`).
2.  **Pembagian Dataset:** Data dibagi menjadi tiga subset menggunakan `train_test_split` dari Scikit-Learn:
    * **Training Set (60%):** Digunakan untuk melatih model.
    * **Validation Set (20%):** Digunakan untuk menyetel *hyperparameter* dan memantau kinerja model selama pelatihan.
    * **Test Set (20%):** Digunakan untuk evaluasi akhir model yang sudah terlatih sepenuhnya, memberikan perkiraan kinerja model pada data yang belum pernah dilihat sebelumnya.
    * `stratify=y` digunakan untuk memastikan distribusi kelas target yang sama di ketiga subset, krusial untuk dataset yang tidak seimbang seperti deteksi penipuan.
3.  **Penskalaan Fitur:** `StandardScaler` dari Scikit-Learn digunakan untuk menstandarisasi fitur numerik pada semua set data (training, validasi, dan pengujian). Ini mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1. Penskalaan dilakukan *setelah* pembagian data untuk mencegah *data leakage*.

### c. Mengembangkan Arsitektur MLP (Multi-Layer Perceptron)

**Tujuan:** Membangun model *Deep Learning* menggunakan TensorFlow (Keras) dan PyTorch dengan menerapkan berbagai teknik *Deep Learning* modern.

#### Model TensorFlow (dengan Keras)

**Arsitektur:**
* Model `Sequential` dengan 3 *hidden layers* (256, 128, dan 64 neuron).
* **Fungsi Aktivasi:** `relu` untuk *hidden layers*, `sigmoid` untuk *output layer*.
* **Inisialisasi Bobot:** `he_normal` digunakan untuk inisialisasi bobot *hidden layers*, cocok untuk `relu`.
* **Batch Normalization:** Diterapkan setelah setiap *dense layer* untuk menormalkan input, mempercepat pelatihan, dan bertindak sebagai regularisasi.
* **Dropout:** Ditempatkan setelah setiap `BatchNormalization` dengan tingkat 0.3 untuk dua layer pertama dan 0.2 untuk layer ketiga, untuk mencegah *overfitting*.
* **Regularisasi L2 (Weight Decay):** Diterapkan pada *output layer* (`kernel_regularizer=keras.regularizers.l2(0.001)`) untuk mendorong bobot yang lebih kecil dan mengurangi *overfitting*.
* **Output Layer:** Satu neuron dengan `sigmoid` untuk klasifikasi biner.

**Konfigurasi Pelatihan:**
* **Loss Function:** `binary_crossentropy`.
* **Optimizer:** `Adam` dengan *learning rate* yang diatur oleh `ExponentialDecay` untuk penyesuaian *learning rate* selama pelatihan.
* **Early Stopping:** `EarlyStopping` dengan `patience=10` dan `restore_best_weights=True` untuk memantau `val_loss` dan menghentikan pelatihan jika tidak ada peningkatan, serta mengembalikan bobot terbaik.

#### Model PyTorch (Template & Loop Pelatihan Manual)

**Arsitektur:**
* Model `nn.Module` kustom dengan 3 *hidden layers* (256, 128, dan 64 neuron).
* **Layer:** `nn.Linear`, `nn.BatchNorm1d`, `nn.Dropout`.
* **Fungsi Aktivasi:** `torch.relu` untuk *hidden layers*, `torch.sigmoid` untuk *output layer*.
* **Batch Normalization:** Diterapkan setelah layer linier dan sebelum fungsi aktivasi.
* **Dropout:** Ditempatkan setelah `Batch Normalization`.

**Konfigurasi Pelatihan (Loop Manual):**
* **Loss Function:** `nn.BCELoss()` (Binary Cross-Entropy).
* **Optimizer:** `optim.Adam` dengan `weight_decay=1e-4` (L2 regularization).
* **Learning Rate Scheduler:** `optim.lr_scheduler.ReduceLROnPlateau` yang mengurangi *learning rate* ketika *validasi loss* berhenti menurun.
* **Early Stopping:** Logika *early stopping* diimplementasikan secara manual dengan memantau *validasi loss* dan menyimpan bobot model terbaik.

## 2. Matriks Evaluasi dan Visualisasi

**Tujuan:** Mengukur kinerja model menggunakan berbagai metrik yang relevan untuk klasifikasi, terutama untuk dataset tidak seimbang, dan memvisualisasikan hasilnya.

**Metrik Evaluasi yang Digunakan:**
* **Accuracy (Akurasi):** Proporsi prediksi yang benar dari total prediksi. $Accuracy = (TP + TN) / (TP + TN + FP + FN)$.
* **Precision (Presisi):** Proporsi *true positive* dari semua prediksi positif. $Precision = TP / (TP + FP)$.
* **Recall (Sensitivitas / True Positive Rate):** Proporsi *true positive* dari semua kasus positif yang sebenarnya. $Recall = TP / (TP + FN)$.
* **F1-Score:** Rata-rata harmonik dari Presisi dan Recall, berguna untuk menyeimbangkan keduanya, terutama pada dataset tidak seimbang. $F1 = 2 * (Precision * Recall) / (Precision + Recall)$.
* **Confusion Matrix:** Tabel visual yang menampilkan jumlah True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN).
* **ROC Curve (Receiver Operating Characteristic Curve):** Memplot True Positive Rate (Recall) terhadap False Positive Rate (FPR) pada berbagai *threshold* klasifikasi.
* **AUC (Area Under the ROC Curve):** Area di bawah kurva ROC. Nilai mendekati 1 menunjukkan model yang sangat baik dalam membedakan kelas.

**Visualisasi:**
* Heatmap untuk Confusion Matrix.
* Plot ROC Curve dengan nilai AUC.

## 3. Penjelasan Model dan Analisis Perbandingan

**Model TensorFlow:**
* **Kelebihan:**
    * Integrasi erat dengan ekosistem TensorFlow (TensorBoard, TF-Serving, TF-Lite).
    * Menyediakan abstraksi tinggi (Sequential dan Functional API) yang memudahkan pembangunan model.
    * Dirancang untuk skala produksi.
    * Optimasi otomatis model ke TF Functions untuk kinerja.
* **Kekurangan:**
    * Fleksibilitas lebih rendah untuk kustomisasi ekstrem pada *training loop* dibandingkan PyTorch.
    * *Debugging* dalam mode grafik bisa lebih menantang.

**Model PyTorch:**
* **Kelebihan:**
    * Fleksibilitas tinggi dan *eager execution* secara *default* memudahkan eksperimen dan *debugging*.
    * Memberikan kontrol granular atas proses pelatihan.
    * Komunitas yang kuat di kalangan peneliti.
* **Kekurangan:**
    * Abstraksi lebih rendah, membutuhkan lebih banyak kode *boilerplate* untuk *training loop*.
    * Deployment model ke lingkungan yang beragam (mobile, embedded) bisa lebih kompleks dibandingkan TensorFlow.

**Perbandingan Kinerja Berdasarkan Hasil Evaluasi:**

| Metrik      | Model TensorFlow | Model PyTorch |
|-------------|------------------|---------------|
| Accuracy    | 0.9993           | 0.9989        |
| Precision   | 0.7767           | 0.8113        |
| Recall      | 0.8163           | 0.4388        |
| F1-Score    | 0.7960           | 0.5695        |

**Analisis:**
Kedua model menunjukkan akurasi yang sangat tinggi (>99%), yang merupakan hal yang umum pada dataset yang sangat tidak seimbang seperti deteksi penipuan. Namun, untuk metrik yang lebih relevan dalam konteks deteksi penipuan (yaitu, metrik untuk kelas minoritas/penipuan):

* **Model PyTorch** memiliki **Precision** yang sedikit lebih tinggi (0.8113) dibandingkan TensorFlow (0.7767). Ini berarti ketika PyTorch mengidentifikasi penipuan, kemungkinannya lebih besar prediksi tersebut akurat.
* **Model TensorFlow** menunjukkan **Recall** yang jauh lebih baik (0.8163) dibandingkan PyTorch (0.4388). Ini mengindikasikan bahwa TensorFlow berhasil mendeteksi sebagian besar transaksi penipuan yang sebenarnya.
* **F1-Score** pada **Model TensorFlow** (0.7960) secara signifikan lebih tinggi daripada PyTorch (0.5695). F1-Score yang lebih tinggi pada TensorFlow menunjukkan keseimbangan yang lebih baik antara Presisi dan Recall, dan secara keseluruhan lebih efektif dalam menyeimbangkan identifikasi penipuan dengan meminimalkan kesalahan.

**Kesimpulan:**
Dalam konteks deteksi penipuan, di mana seringkali prioritas adalah meminimalkan *false negative* (penipuan yang terlewat), **model TensorFlow dianggap lebih baik** karena nilai Recall dan F1-Score-nya yang jauh lebih tinggi. Meskipun Presisinya sedikit lebih rendah dari PyTorch, kemampuan TensorFlow untuk mendeteksi sebagian besar penipuan yang ada membuatnya lebih berharga dalam skenario ini.

Penting untuk dicatat bahwa kinerja model pada dataset yang tidak seimbang juga sangat dipengaruhi oleh strategi pembagian data (`stratified split`) dan teknik regularisasi yang diterapkan (seperti Batch Normalization, Dropout, dan L2 Regularization). Perbedaan kecil dalam kinerja antar *framework* dapat terjadi karena inisialisasi acak atau detail implementasi internal.
