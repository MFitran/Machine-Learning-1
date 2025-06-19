Tentu, berikut adalah rangkuman Bab 10 yang lebih terstruktur dan rapi dalam format Markdown, lengkap dengan penjelasan umum dan poin-poin penting:

# Bab 10: Pengantar Jaringan Neural Buatan dengan Keras

Bab 10 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Edisi ke-2) menyajikan pengantar komprehensif mengenai Jaringan Neural Buatan (ANN) dan Deep Learning, dengan fokus pada implementasi praktis menggunakan *library* Keras. Bab ini menguraikan dasar-dasar teoritis, evolusi arsitektur jaringan, serta panduan langkah demi langkah untuk membangun dan melatih model.

## I. Konsep Dasar Jaringan Neural Buatan (ANN)

* **Inspirasi Biologis**: ANN terinspirasi oleh struktur dan fungsi neuron biologis di otak, meskipun tidak selalu meniru persis cara kerja biologisnya.
    * **Neuron Biologis**: Dijelaskan sebagai sel dengan badan sel, dendrit (ekstensi bercabang), dan akson yang berakhir di sinapsis, di mana sinyal listrik (potensial aksi) dan sinyal kimia (neurotransmiter) berperan dalam komunikasi antar neuron.
    * **Komputasi Logis**: McCulloch dan Pitts (1943) mengusulkan model neuron buatan sederhana (input/output biner) yang dapat melakukan operasi logika dasar seperti AND, OR, dan NOT.
* **Perceptron**:
    * **Threshold Logic Unit (TLU)**: Model neuron buatan ini menerima input numerik berbobot, menjumlahkannya, dan menerapkan fungsi langkah (seperti fungsi langkah Heaviside) untuk menghasilkan output.
    * **Arsitektur Perceptron**: Terdiri dari satu lapisan TLU yang *fully connected* (dense layer), dengan neuron input dan neuron bias tambahan.
    * **Algoritma Pelatihan Perceptron**: Diusulkan oleh Frank Rosenblatt (1957), terinspirasi dari aturan Hebb, di mana bobot koneksi diperkuat berdasarkan prediksi yang benar.
    * **Keterbatasan**: Perceptron klasik hanya dapat menyelesaikan masalah yang *linearly separable* (seperti yang ditunjukkan oleh Minsky dan Papert dengan masalah XOR) dan tidak mengeluarkan probabilitas kelas.
* **Multilayer Perceptron (MLP)**:
    * **Arsitektur**: MLP mengatasi keterbatasan Perceptron dengan menumpuk satu atau lebih *hidden layers* (lapisan tersembunyi) di antara lapisan input dan output. Sinyal mengalir searah (disebut *feedforward neural network* atau FNN). Jaringan dengan tumpukan lapisan tersembunyi yang dalam disebut *Deep Neural Network* (DNN).
    * **Algoritma Backpropagation**: Diperkenalkan oleh Rumelhart, Hinton, dan Williams (1986), algoritma ini merupakan bentuk Gradient Descent yang secara efisien menghitung gradien error (menggunakan *reverse-mode autodiff*) di seluruh parameter jaringan dalam dua *pass* (forward dan backward).
    * **Inisialisasi Bobot**: Penting untuk menginisialisasi bobot lapisan tersembunyi secara acak untuk memecah simetri dan memungkinkan pembelajaran yang efektif.
    * **Fungsi Aktivasi**: Fungsi aktivasi non-linear seperti fungsi logistik, tangen hiperbolik (tanh), dan Rectified Linear Unit (ReLU) sangat penting untuk memperkenalkan non-linearitas, memungkinkan MLP memecahkan masalah kompleks.

## II. Jenis-jenis MLP Berdasarkan Tugas

* **MLP untuk Regresi**:
    * **Neuron Output**: Satu neuron output untuk prediksi nilai tunggal; beberapa neuron output untuk regresi multivariat.
    * **Fungsi Aktivasi Output**: Umumnya tidak ada fungsi aktivasi (output dapat berkisar bebas). ReLU atau softplus untuk output positif, atau logistik/tanh untuk output terbatas.
    * **Fungsi Loss**: Umumnya *Mean Squared Error* (MSE). Alternatifnya, *Mean Absolute Error* (MAE) atau *Huber loss* untuk data dengan *outlier*.
* **MLP untuk Klasifikasi**:
    * **Klasifikasi Biner**: Satu neuron output dengan fungsi aktivasi logistik (sigmoid) untuk memprediksi probabilitas kelas positif.
    * **Klasifikasi Biner Multilabel**: Satu neuron output per label biner positif, masing-masing dengan fungsi aktivasi logistik.
    * **Klasifikasi Multikelas**: Satu neuron output per kelas (untuk tiga atau lebih kelas eksklusif), menggunakan fungsi aktivasi *softmax* pada seluruh lapisan output, memastikan probabilitas berjumlah 1.
    * **Fungsi Loss**: Umumnya *cross-entropy loss* (juga dikenal sebagai *log loss*).

## III. Implementasi MLP dengan Keras

Keras adalah API Deep Learning tingkat tinggi yang mempermudah pembangunan, pelatihan, evaluasi, dan eksekusi jaringan neural.

* **Instalasi TensorFlow 2**: Keras terintegrasi dengan TensorFlow, sehingga instalasi TensorFlow 2 diperlukan.
* **Membangun Pengklasifikasi Gambar (Sequential API)**:
    * **Memuat Dataset**: Menggunakan `keras.datasets` untuk memuat data (misalnya, Fashion MNIST). Data biasanya perlu diskalakan (misalnya, intensitas piksel menjadi 0-1) dan dibagi menjadi set pelatihan, validasi, dan pengujian.
    * **Membuat Model**: Sequential API (`keras.models.Sequential`) memungkinkan pembangunan model dengan menumpuk lapisan secara berurutan. Lapisan `Flatten` untuk mengubah input multidimensi menjadi 1D, diikuti oleh lapisan `Dense` (tersembunyi dan output) dengan fungsi aktivasi yang sesuai.
    * **Ringkasan Model**: Metode `model.summary()` menampilkan detail lapisan, bentuk output, dan jumlah parameter.
    * **Mengompilasi Model**: Metode `model.compile()` menentukan fungsi *loss* (`sparse_categorical_crossentropy` untuk label sparse multikelas), *optimizer* (`sgd`), dan metrik (`accuracy`) untuk evaluasi.
    * **Melatih dan Mengevaluasi**: Metode `model.fit()` melatih model menggunakan data pelatihan dan data validasi (opsional). Objek `History` yang dikembalikan berisi riwayat *loss* dan metrik per *epoch*. Kurva pembelajaran dapat diplot dari `history.history`. Metode `model.evaluate()` digunakan untuk mengestimasi *generalization error* pada set pengujian.
    * **Membuat Prediksi**: Metode `model.predict()` menghasilkan probabilitas kelas, dan `model.predict_classes()` memberikan kelas dengan probabilitas tertinggi.
* **Membangun MLP Regresi (Sequential API)**: Mirip dengan klasifikasi, namun dengan satu neuron output tanpa aktivasi, dan fungsi *loss* MSE.
* **Membangun Model Kompleks (Functional API)**:
    * Digunakan untuk arsitektur non-sequential atau model dengan banyak input/output (misalnya, *Wide & Deep neural network*).
    * Lapisan dibuat dan dihubungkan dengan memanggilnya seperti fungsi (`hidden1 = keras.layers.Dense(...)(input_)`).
    * Model didefinisikan dengan menentukan input dan outputnya (`keras.Model(inputs=[...], outputs=[...])`).
    * Mendukung model dengan *multiple inputs* (misalnya, memisahkan fitur ke jalur lebar dan dalam) dan *multiple outputs* (misalnya, untuk tugas ganda atau regularisasi).
* **Membangun Model Dinamis (Subclassing API)**:
    * Digunakan untuk model dengan perilaku dinamis (loop, percabangan kondisional) atau preferensi gaya pemrograman imperatif.
    * Membuat subclass `keras.Model`, mendefinisikan lapisan di konstruktor, dan mengimplementasikan logika komputasi di metode `call()`.
    * Kelebihan: Fleksibilitas maksimal. Kekurangan: Arsitektur tersembunyi, kurang mudah disimpan/diinspeksi secara statis.

## IV. Manajemen Model dan Visualisasi

* **Menyimpan dan Memulihkan Model**: Menggunakan `model.save("my_keras_model.h5")` untuk menyimpan arsitektur, bobot, dan *optimizer*. Model dapat dimuat kembali dengan `keras.models.load_model()`.
* **Menggunakan Callbacks**:
    * `ModelCheckpoint`: Menyimpan checkpoint model selama pelatihan, dapat menyimpan model terbaik berdasarkan performa validasi (`save_best_only=True`).
    * `EarlyStopping`: Menginterupsi pelatihan ketika tidak ada peningkatan performa validasi selama periode tertentu (`patience`), dan secara opsional mengembalikan bobot model terbaik (`restore_best_weights=True`).
    * *Custom Callbacks*: Dapat dibuat dengan *subclassing* `keras.callbacks.Callback` untuk kontrol lebih lanjut.
* **Visualisasi dengan TensorBoard**:
    * TensorBoard adalah alat visualisasi interaktif untuk memantau kurva pembelajaran, membandingkan *run* yang berbeda, dan menganalisis statistik pelatihan.
    * Callback `keras.callbacks.TensorBoard` digunakan untuk menulis *event files* yang kemudian dapat dibaca oleh server TensorBoard.

## V. Fine-Tuning Hyperparameter Jaringan Neural

Menyesuaikan banyak *hyperparameter* adalah tantangan dalam jaringan neural.

* **Pencarian Hyperparameter Otomatis**:
    * Menggunakan `keras.wrappers.scikit_learn.KerasRegressor` atau `KerasClassifier` untuk membungkus model Keras menjadi objek yang kompatibel dengan Scikit-Learn.
    * Kemudian menggunakan `RandomizedSearchCV` (atau `GridSearchCV`) untuk menjelajahi ruang *hyperparameter*.
* **Panduan Memilih Hyperparameter**:
    * **Jumlah Hidden Layers**: Untuk banyak masalah, satu atau dua lapisan tersembunyi sudah cukup. Untuk masalah yang lebih kompleks, DNN yang lebih dalam dapat belajar fitur hierarkis dan mencapai performa yang lebih baik.
    * **Jumlah Neuron per Hidden Layer**: Ditentukan oleh input dan output. Praktiknya, menggunakan jumlah neuron yang sama di semua lapisan tersembunyi atau lapisan pertama yang lebih besar sering kali berfungsi dengan baik. Hindari lapisan 'bottleneck' dengan terlalu sedikit neuron.
    * **Learning Rate, Batch Size, dan Lainnya**: `Learning rate` sangat penting; `optimizer` yang lebih cepat dapat meningkatkan kecepatan pelatihan. `Batch size` memengaruhi performa dan waktu pelatihan. Fungsi aktivasi ReLU adalah default yang baik untuk lapisan tersembunyi.
