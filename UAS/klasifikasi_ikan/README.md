# Klasifikasi pada Data Gambar (Ikan)

Notebook `klasifikasi_ikan.ipynb` ini bertujuan untuk melakukan klasifikasi jenis ikan menggunakan Convolutional Neural Network (CNN). Prosesnya mengikuti alur kerja *Machine Learning* yang umum, mulai dari persiapan data hingga evaluasi model.

**Beberapa hal yang ingin saya sampaikan sebelum dimulai:**

  - **Download dataset** pada tautan: [https://drive.google.com/drive/folders/1UKpVcmjXUXvmRTEU7vWJOo1-jwPFoQzB](https://drive.google.com/drive/folders/1UKpVcmjXUXvmRTEU7vWJOo1-jwPFoQzB)
  - **Dataset didownload berbentuk zip**, langsung *upload* ke MyDrive Anda, lalu masuk ke Google Colab.
  - **Sebelum *connect***, ubah *runtime type* menjadi T4 GPU untuk kinerja yang optimal.

## 1\. Hubungkan ke Google Drive

  * **Isi Kode:** Bagian ini menggunakan library `google.colab` untuk mengautentikasi dan menghubungkan sesi Google Colab ke akun Google Drive pengguna.
  * **Hasil *Output*:** Setelah eksekusi, akan muncul pesan di konsol yang mengindikasikan bahwa Google Drive telah berhasil terhubung, seperti "Google Drive berhasil terhubung." dan "Mounted at /content/drive". Ini penting karena dataset yang dibutuhkan disimpan di Google Drive.

## 2\. Ekstraksi Data

  * **Isi Kode:** Kode ini berfungsi untuk mengekstrak file ZIP yang berisi dataset ikan dari Google Drive ke direktori lokal di lingkungan Colab (`/content/dataset`). Ini melibatkan penggunaan modul `zipfile` untuk operasi dekompresi.
  * **Hasil *Output*:** Konsol akan menampilkan pesan konfirmasi bahwa dataset telah berhasil diekstrak dan menunjukkan lokasi direktori tujuan, contohnya: "Dataset berhasil diekstrak ke: /content/dataset". Ini memastikan bahwa gambar-gambar ikan siap diakses untuk langkah selanjutnya.

## 3\. Persiapan Data

  * **Isi Kode:** Tahap ini adalah inti dari pra-pemrosesan data gambar.
      * **Penjelajahan Struktur Direktori Data:** Meskipun tidak ada kode eksplisit yang dijalankan di ringkasan Anda untuk penjelajahan direktori, bagian ini secara implisit memeriksa dan memastikan struktur folder yang benar di dalam dataset (misalnya, setiap subdirektori mewakili satu kelas ikan).
      * **Data Augmentasi dan Pra-pemrosesan:** Kode ini menginisialisasi `ImageDataGenerator` dari Keras. Generator ini melakukan dua tugas utama:
          * **Normalisasi Piksel:** Mengubah nilai piksel gambar dari rentang 0-255 menjadi 0-1 (`rescale=1./255`), yang merupakan praktik standar untuk input jaringan saraf.
          * **Augmentasi Data:** Menerapkan transformasi acak pada gambar (seperti `shear_range`, `zoom_range`, `horizontal_flip`) untuk membuat variasi data pelatihan. Ini membantu model belajar lebih baik dan mengurangi *overfitting*, seperti yang dijelaskan dalam Bab 13 buku "Hands-On Machine Learning" tentang *Data Augmentation*.
          * **Pembagian Data:** Membagi dataset menjadi set pelatihan (80%) dan set validasi (20%) menggunakan `validation_split`.
          * `flow_from_directory` kemudian digunakan untuk memuat gambar dari direktori yang ditentukan, secara otomatis mengidentifikasi kelas dari nama subdirektori, dan mengubah ukuran gambar menjadi target yang ditentukan (128x128 piksel).
  * **Hasil *Output*:**
      * Konsol akan menunjukkan proses `Found NNNN images belonging to M classes` untuk data pelatihan dan validasi, mengonfirmasi jumlah gambar dan kelas yang terdeteksi.
      * Akan ada pesan seperti "Data berhasil dimuat dan diproses."
      * Juga akan menampilkan jumlah kelas dan indeks kelas yang dipetakan (misalnya, `{'Ikan A': 0, 'Ikan B': 1, ...}`).

## 4\. Membuat Model CNN

  * **Isi Kode:** Bagian ini membangun arsitektur Convolutional Neural Network (CNN) menggunakan API Sequential Keras, sesuai dengan prinsip-prinsip desain CNN yang dibahas dalam Bab 14 buku "Hands-On Machine Learning".
      * Model terdiri dari beberapa lapisan:
          * **`Conv2D`:** Lapisan konvolusi untuk mengekstrak fitur dari gambar. Parameter seperti `filters` (jumlah filter), `kernel_size` (ukuran filter), dan `activation` (fungsi aktivasi ReLU) ditentukan. `input_shape` hanya ditentukan pada lapisan pertama.
          * **`MaxPooling2D`:** Lapisan *pooling* untuk mengurangi dimensi spasial gambar, membantu mengurangi komputasi dan *overfitting*.
          * **`Flatten`:** Mengubah output dari lapisan konvolusi/pooling menjadi vektor 1D untuk dapat diumpankan ke lapisan *dense*.
          * **`Dense`:** Lapisan *fully connected* (padat) untuk melakukan klasifikasi. Lapisan `Dense` pertama memiliki fungsi aktivasi ReLU, dan lapisan output menggunakan `softmax` karena ini adalah masalah klasifikasi multi-kelas.
          * **`Dropout`:** Lapisan *dropout* digunakan sebagai teknik regularisasi untuk mencegah *overfitting* dengan secara acak menonaktifkan sebagian neuron selama pelatihan.
      * **Kompilasi Model:** Model dikompilasi dengan:
          * `optimizer='adam'`: Algoritma optimasi yang populer dan efektif.
          * `loss='categorical_crossentropy'`: Fungsi *loss* yang cocok untuk klasifikasi multi-kelas dengan label *one-hot encoded* (yang dihasilkan oleh `ImageDataGenerator` dengan `class_mode='categorical'`).
          * `metrics=['accuracy']`: Metrik yang akan dipantau selama pelatihan.
  * **Hasil *Output*:** `model.summary()` akan menampilkan tabel ringkasan arsitektur model, termasuk jenis lapisan, bentuk output setiap lapisan, dan jumlah parameter yang dapat dilatih. Ini memberikan gambaran visual tentang struktur CNN yang dibuat.

## 5\. Pelatihan Model

  * **Isi Kode:** Model dilatih menggunakan metode `model.fit()`.
      * `train_generator` dan `validation_generator` menyediakan data secara *batch* selama pelatihan.
      * `epochs` menentukan berapa kali seluruh dataset pelatihan akan diulang.
      * `steps_per_epoch` dan `validation_steps` dihitung berdasarkan jumlah sampel dan ukuran *batch* untuk memastikan semua data diproses dalam setiap *epoch*.
  * **Hasil *Output*:**
      * Selama pelatihan, konsol akan menampilkan *progress bar* untuk setiap *epoch*, menunjukkan `loss` dan `accuracy` pada data pelatihan, serta `val_loss` dan `val_accuracy` pada data validasi.
      * Setelah pelatihan selesai, tidak ada *output* lain selain *log* pelatihan yang lengkap.

## 6\. Evaluasi Model

  * **Isi Kode:** Bagian terakhir ini mengevaluasi kinerja model dan memvisualisasikan *history* pelatihan.
      * **Plot *History*:** Menggunakan `matplotlib.pyplot` untuk memplot grafik `accuracy` dan `loss` selama pelatihan, baik untuk data pelatihan maupun data validasi. Ini membantu menganalisis *overfitting* atau *underfitting*.
      * **Evaluasi Akhir:** Model dievaluasi pada `validation_generator` menggunakan `model.evaluate()` untuk mendapatkan nilai *loss* dan *accuracy* akhir pada data yang tidak terlihat selama pelatihan.
      * **Penyimpanan Model (Opsional):** Ada baris kode yang dikomentari untuk menyimpan model terlatih dalam format `.h5`, yang memungkinkan model digunakan kembali tanpa perlu dilatih ulang.
  * **Hasil *Output*:**
      * Akan ditampilkan dua plot grafik:
          * "Training and Validation Accuracy" menunjukkan bagaimana akurasi berkembang seiring *epoch*. Jika akurasi validasi mulai menurun sementara akurasi pelatihan terus meningkat, ini adalah tanda *overfitting*.
          * "Training and Validation Loss" menunjukkan bagaimana *loss* berkembang seiring *epoch*. Pola serupa dengan akurasi berlaku untuk *loss*.
      * Di bawah plot, akan dicetak nilai *loss* dan *accuracy* terakhir pada data validasi, misalnya:
        ```
        Loss pada data validasi: 0.X
        Akurasi pada data validasi: 0.Y
        ```
        Nilai `0.Y` ini adalah metrik kinerja utama yang menunjukkan seberapa baik model dapat mengklasifikasikan jenis ikan pada data baru.
