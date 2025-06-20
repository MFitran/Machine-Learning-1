# Ringkasan dan Penjelasan Notebook Regresi Deep Learning

Notebook `regresi.ipynb` berisi pipeline *end-to-end* untuk membangun dan mengevaluasi model regresi *deep learning* menggunakan dua *framework* populer: TensorFlow dan PyTorch.

## 1. Pipeline End-to-End untuk Regresi Model Deep Learning

Bagian ini mencakup langkah-langkah dasar hingga menengah dalam membangun model regresi.

### a. Pengumpulan & Pembersihan Data (Pandas)
* **Sumber Data**: Data dimuat dari sebuah *string* yang merepresentasikan data tabular. Kolom pertama (`0`) diidentifikasi sebagai kolom target (dependen), sementara kolom sisanya adalah fitur (independen).
* **Pembersihan Data**:
    * Notebook memeriksa keberadaan nilai-nilai nol (`null`) di setiap kolom.
    * Jika ditemukan nilai nol pada kolom numerik, nilai-nilai tersebut diisi menggunakan median dari kolom tersebut. Ini adalah strategi umum untuk menghindari bias yang mungkin disebabkan oleh *outlier* jika menggunakan mean.
    * Kolom target (`0`) secara spesifik diperiksa untuk memastikan bahwa ia adalah tipe data numerik. Jika tidak, akan dicoba dikonversi, dan baris yang gagal dikonversi akan dihapus.

### b. Feature Engineering (Feature Scaling)
* **Standardisasi**: Data fitur (`X`) distandardisasi menggunakan `StandardScaler` dari Scikit-Learn. Standardisasi mengubah distribusi fitur sehingga memiliki rata-rata 0 dan standar deviasi 1. Ini penting untuk model *deep learning* karena dapat mempercepat konvergensi dan mencegah fitur dengan skala lebih besar mendominasi pelatihan.
* **Pembagian Data**: Data dibagi menjadi tiga set:
    * **Training Set**: Digunakan untuk melatih model.
    * **Validation Set**: Digunakan untuk menyetel *hyperparameter* dan memantau kinerja model selama pelatihan, membantu mencegah *overfitting*.
    * **Test Set**: Digunakan untuk evaluasi akhir model yang telah dilatih sepenuhnya, memberikan estimasi kinerja model pada data yang belum pernah dilihat sebelumnya.

### c. Pengembangan Arsitektur MLP (TensorFlow & PyTorch)

Kedua model mengimplementasikan arsitektur *Multilayer Perceptron* (MLP) dengan serangkaian teknik *deep learning* canggih.

#### Model TensorFlow
* **Arsitektur**:
    * Tiga lapisan `Dense` (fully connected) tersembunyi dengan 256, 128, dan 64 neuron. Ukuran neuron yang menurun adalah pola umum untuk mengekstrak fitur berjenjang.
    * Fungsi aktivasi `elu` (Exponential Linear Unit) digunakan di lapisan tersembunyi. ELU seringkali mengungguli ReLU karena menghindari masalah "dying ReLUs" dan membantu normalisasi diri jaringan.
    * Inisialisasi bobot `he_normal` digunakan, yang direkomendasikan untuk fungsi aktivasi seperti ELU.
    * Lapisan output tunggal tanpa aktivasi, cocok untuk regresi.
* **Normalisasi Batch (`BatchNormalization`)**: Diterapkan setelah setiap lapisan `Dense`. Membantu menstabilkan pelatihan dan memungkinkan *learning rate* yang lebih besar.
* **Dropout**: Lapisan `Dropout(0.3)` ditambahkan setelah setiap `BatchNormalization` untuk mengurangi *overfitting* dengan secara acak menonaktifkan neuron selama pelatihan.
* **Regularisasi L2 (`kernel_regularizer`)**: Menerapkan penalti L2 pada bobot lapisan untuk mencegah bobot menjadi terlalu besar, sehingga mengurangi *overfitting*.
* **Optimizer**: Menggunakan `keras.optimizers.Adam` dengan jadwal *learning rate* `ExponentialDecay`. Adam adalah optimizer adaptif yang populer karena konvergensi yang cepat. `ExponentialDecay` mengurangi *learning rate* seiring waktu pelatihan untuk konvergensi yang lebih stabil.
* **Early Stopping**: Digunakan sebagai *callback* untuk menghentikan pelatihan jika kinerja validasi tidak meningkat selama sejumlah epoch tertentu (`patience=10`), dan mengembalikan bobot terbaik.

#### Model PyTorch
* **Arsitektur**:
    * Mirip dengan TensorFlow, dengan tiga lapisan `Linear` (fully connected) yang diikuti oleh `BatchNorm1d` dan `Dropout`.
    * Fungsi aktivasi `relu` digunakan (sebagai pengganti ELU yang bukan *built-in* di PyTorch standar).
    * Lapisan output tunggal untuk regresi.
* **Normalisasi Batch (`nn.BatchNorm1d`)**: Diterapkan setelah setiap lapisan linear.
* **Dropout**: `nn.Dropout(0.3)` juga digunakan untuk regularisasi.
* **Optimizer**: Menggunakan `optim.AdamW` dengan `weight_decay` (regularisasi L2) yang terpisah. `AdamW` seringkali memberikan generalisasi yang lebih baik.
* **Learning Rate Scheduling**: Menggunakan `optim.lr_scheduler.StepLR` untuk mengurangi *learning rate* setiap beberapa epoch.
* **Early Stopping**: Diimplementasikan secara manual dalam *loop* pelatihan PyTorch, memantau *loss* validasi dan menyimpan model terbaik.

## 2. Matriks Evaluasi & Visualisasi Predicted vs Actual Value

Bagian ini fokus pada evaluasi kinerja kedua model yang telah dilatih.

### Metrik Evaluasi
* **RMSE (Root Mean Squared Error)**: Mengukur rata-rata magnitudo kesalahan. Semakin rendah nilai RMSE, semakin baik. Metrik ini sangat sensitif terhadap kesalahan besar (*outlier*).
* **R-squared (Koefisien Determinasi)**: Mengukur proporsi variabilitas dalam variabel target yang dapat dijelaskan oleh model. Nilai berkisar antara 0 dan 1. Nilai yang mendekati 1 menunjukkan model yang sangat baik, sementara nilai 0 berarti model tidak lebih baik dari rata-rata. Nilai negatif menunjukkan kinerja yang lebih buruk dari sekadar prediksi rata-rata.
* **MSE (Mean Squared Error)**: Rata-rata kuadrat kesalahan. Mirip dengan RMSE tetapi dalam satuan kuadrat, sehingga kurang intuitif untuk interpretasi langsung.

### Visualisasi Prediksi vs. Aktual
* Kedua model menampilkan *scatterplot* dari nilai aktual (`y_true`) terhadap nilai prediksi (`y_pred`).
* Garis diagonal merah menunjukkan prediksi yang sempurna (nilai aktual = nilai prediksi). Titik-titik yang berkumpul di sekitar garis ini menunjukkan kinerja model yang baik.
* Judul plot menyertakan nilai RMSE dan R-squared untuk referensi cepat.

## 3. Analisis Kinerja Model

### Ringkasan Metrik
Berdasarkan hasil yang Anda berikan:

| Metrik      | TensorFlow Model | PyTorch Model |
| :---------- | :--------------- | :------------ |
| **RMSE** | 13.0459          | 14.3199       |
| **R-squared** | -0.4154          | -0.7230       |
| **MSE** | 170.1955         | 205.0587      |

### Analisis Hasil

Dari metrik di atas, **model TensorFlow menunjukkan kinerja yang lebih unggul** dibandingkan dengan model PyTorch dalam tugas regresi ini.

1.  **RMSE**: Model TensorFlow memiliki RMSE yang lebih rendah (13.0459) dibandingkan PyTorch (14.3199). Ini berarti model TensorFlow menghasilkan kesalahan prediksi rata-rata yang lebih kecil dan prediksi yang lebih akurat secara keseluruhan. Dalam konteks regresi, RMSE adalah metrik kunci yang mengindikasikan seberapa dekat prediksi model dengan nilai sebenarnya.

2.  **R-squared**: Kedua model menunjukkan nilai R-squared negatif (TensorFlow: -0.4154, PyTorch: -0.7230). Nilai R-squared negatif adalah indikator bahwa model berkinerja *lebih buruk* daripada hanya memprediksi rata-rata nilai target dari dataset pelatihan. Dengan kata lain, model yang dibangun tidak mampu menangkap tren atau pola yang berarti dalam data. Meskipun demikian, R-squared TensorFlow yang "kurang negatif" menunjukkan bahwa ia setidaknya berkinerja sedikit lebih baik dalam menjelaskan variasi data dibandingkan PyTorch.

3.  **MSE**: Konsisten dengan RMSE, MSE model TensorFlow (170.1955) juga lebih rendah dari MSE model PyTorch (205.0587), menegaskan keunggulan model TensorFlow dalam meminimalkan kesalahan kuadrat.

**Kesimpulan dan Saran Perbaikan:**

Meskipun model TensorFlow lebih baik daripada PyTorch dalam perbandingan ini, fakta bahwa kedua model menunjukkan nilai R-squared negatif yang signifikan mengindikasikan bahwa **model-model tersebut masih *underfit* terhadap data**. Ini berarti model terlalu sederhana untuk menangkap kompleksitas pola dalam data, atau ada masalah mendasar pada data itu sendiri.

Beberapa langkah perbaikan yang bisa dipertimbangkan:

* **Peningkatan Arsitektur Model**:
    * **Lapisan Lebih Dalam/Neuron Lebih Banyak**: Coba tambahkan lebih banyak lapisan tersembunyi atau lebih banyak neuron per lapisan untuk meningkatkan kapasitas model.
    * **Fungsi Aktivasi**: Meskipun ELU dan ReLU adalah pilihan yang baik, eksplorasi fungsi aktivasi lain (misalnya SELU jika arsitektur memungkinkan *self-normalization*) mungkin bermanfaat.
* **Hyperparameter Tuning yang Lebih Agresif**:
    * **Learning Rate**: Lakukan pencarian *learning rate* yang lebih cermat. Jadwal *learning rate* yang lebih kompleks (misalnya *warm restarts* atau *cosine annealing*) dapat membantu.
    * **Regularisasi**: Eksperimen dengan tingkat *dropout* atau nilai regularisasi L1/L2 yang berbeda. Terkadang, regularisasi yang terlalu kuat dapat menyebabkan *underfitting*.
    * **Ukuran Batch**: Coba ukuran *batch* yang berbeda.
* **Kualitas Data dan Feature Engineering Lanjutan**:
    * **Analisis Data Eksploratif (EDA) Mendalam**: Pahami lebih lanjut distribusi, korelasi, dan *outlier* dalam fitur-fitur.
    * **Feature Engineering Kustom**: Pertimbangkan untuk membuat fitur-fitur baru dari fitur yang sudah ada yang mungkin lebih relevan atau informatif untuk target regresi.
    * **Penanganan Outlier**: Jika ada *outlier* ekstrem, strategi yang lebih canggih untuk menanganinya (misalnya *winsorization* atau transformasi non-linear) mungkin diperlukan.
    * **Normalisasi Alternatif**: Selain `StandardScaler`, coba `MinMaxScaler` atau normalisasi lain.

Dengan melakukan iterasi pada langkah-langkah ini, kinerja model regresi diharapkan dapat ditingkatkan secara substansial, menuju nilai R-squared positif yang mengindikasikan model yang lebih mampu.
