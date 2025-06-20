
# Rangkuman Bab 1: The Machine Learning Landscape

Bab 1 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" memberikan gambaran umum tentang Machine Learning (ML), menjelaskan konsep-konsep dasar, berbagai jenis sistem ML, serta tantangan umum yang dihadapi dalam proyek ML. Tujuannya adalah untuk membangun fondasi pemahaman sebelum masuk ke implementasi praktis di bab-bab selanjutnya.

## 1. Apa Itu Machine Learning?

Machine Learning didefinisikan sebagai ilmu (dan seni) memprogram komputer agar dapat belajar dari data. Konsep intinya adalah bahwa program komputer dapat meningkatkan kinerjanya pada suatu tugas (T) dengan pengalaman (E), yang diukur oleh ukuran kinerja (P). Sebagai contoh, filter spam adalah program ML yang belajar menandai email spam dari contoh-contoh data pelatihan (pengalaman), dan kinerjanya diukur dari akurasi klasifikasi email.

**Mengapa Menggunakan ML?**
ML sangat berguna untuk:
* **Masalah kompleks tanpa solusi algoritmik yang jelas:** ML dapat menemukan solusi di mana pemrograman tradisional kesulitan. 
* **Mengganti aturan yang kompleks dan panjang:** ML dapat menyederhanakan kode dan berkinerja lebih baik daripada pendekatan tradisional yang mengandalkan daftar aturan yang panjang. 
* **Lingkungan yang berfluktuasi:** Sistem ML dapat beradaptasi secara otomatis dengan data baru dan perubahan lingkungan. 
* **Mendapatkan wawasan dari data besar:** ML dapat membantu manusia menemukan pola tersembunyi (*data mining*) dalam jumlah data yang besar. 

## 2. Contoh Aplikasi ML

ML diterapkan dalam berbagai bidang, seperti:
* **Klasifikasi Gambar:** Mengidentifikasi produk atau mendeteksi tumor pada pindaian otak (menggunakan CNNs). 
* **Pemrosesan Bahasa Alami (NLP):** Mengklasifikasikan artikel berita, menandai komentar ofensif, meringkas dokumen, atau membangun chatbot (menggunakan RNNs, CNNs, atau Transformers). 
* **Regresi:** Memprediksi harga rumah atau pendapatan perusahaan (menggunakan model Regresi Linear, Random Forest, atau jaringan saraf tiruan). 
* **Pengenalan Suara:** Mengubah perintah suara menjadi teks. 
* **Deteksi Anomali:** Mengidentifikasi transaksi kartu kredit yang mencurigakan atau cacat produksi. 
* **Segmentasi Pelanggan:** Mengelompokkan pelanggan berdasarkan perilaku pembelian mereka (*clustering*). 
* **Sistem Rekomendasi:** Menyarankan produk kepada pengguna. 
* **Pembelajaran Penguatan (RL):** Melatih agen cerdas untuk bermain game (misalnya, AlphaGo). 

## 3. Jenis-Jenis Sistem ML

Sistem ML dapat diklasifikasikan berdasarkan:

### a. Tingkat Pengawasan

* **Supervised Learning:** Data pelatihan dilengkapi dengan *label* (solusi yang diinginkan). Tugas umum adalah **klasifikasi** (misalnya, filter spam) dan **regresi** (memprediksi nilai numerik). 
    * *Algoritma contoh:* k-Nearest Neighbors, Regresi Linear, Regresi Logistik, SVM, Decision Trees, Random Forests, Neural Networks. 
* **Unsupervised Learning:** Data pelatihan tidak memiliki *label*. Sistem mencoba menemukan pola tersembunyi dalam data. 
    * *Algoritma contoh:*
        * **Clustering:** Mengelompokkan instans serupa (misalnya, K-Means, DBSCAN). 
        * **Deteksi Anomali & Kebaruan:** Mengidentifikasi instans abnormal. 
        * **Visualisasi & Reduksi Dimensi:** Memproyeksikan data ke dimensi lebih rendah (misalnya, PCA, LLE, t-SNE). 
        * **Pembelajaran Aturan Asosiasi:** Menemukan hubungan antar atribut (misalnya, Apriori). 
* **Semisupervised Learning:** Kombinasi dari supervised dan unsupervised learning, menggunakan data yang sebagian berlabel. Contoh: layanan foto yang mengelompokkan wajah, lalu meminta sedikit *label* untuk mengidentifikasi semua orang. 
* **Reinforcement Learning:** Agen belajar melalui interaksi dengan lingkungan, melakukan tindakan, dan menerima *reward* atau *penalty* untuk mencapai tujuan. Contoh: melatih robot berjalan. 

### b. Cara Belajar

* **Batch Learning (Offline Learning):** Sistem dilatih menggunakan semua data yang tersedia secara *offline* dan tidak dapat belajar secara bertahap. Jika ada data baru, seluruh sistem harus dilatih ulang dari awal. Cocok untuk data statis atau yang berubah lambat. 
* **Online Learning (Incremental Learning):** Sistem dilatih secara bertahap dengan data yang masuk secara berurutan (*mini-batches*). Sistem dapat beradaptasi dengan cepat terhadap perubahan data dan cocok untuk *dataset* besar yang tidak muat di memori (*out-of-core learning*). 

### c. Metode Generalisasi

* **Instance-Based Learning:** Sistem belajar dengan "menghafal" contoh data pelatihan dan menggeneralisasi ke kasus baru dengan mengukur kemiripan dengan contoh yang diketahui. Contoh: k-Nearest Neighbors. 
* **Model-Based Learning:** Sistem membangun model dari data pelatihan, kemudian menggunakan model tersebut untuk membuat prediksi. Proses ini melibatkan pemilihan model, definisi fungsi biaya, pelatihan model (menemukan parameter yang meminimalkan fungsi biaya), dan inferensi (membuat prediksi pada data baru). 

## 4. Tantangan Utama Machine Learning

Proyek ML dapat menghadapi berbagai masalah, terutama terkait "data buruk" dan "algoritma buruk". 

### a. Data Buruk

* **Kuantitas Data Pelatihan yang Tidak Cukup:** Sebagian besar algoritma ML memerlukan ribuan hingga jutaan contoh data. 
* **Data Pelatihan yang Tidak Representatif:** Jika data pelatihan tidak mencerminkan data yang akan dihadapi sistem di produksi, model akan bias (*sampling bias*) dan menghasilkan prediksi yang tidak akurat. 
* **Data Berkualitas Buruk:** Kesalahan, *outliers*, dan *noise* dalam data pelatihan membuat sistem sulit mendeteksi pola yang mendasari. 
* **Fitur yang Tidak Relevan (*Irrelevant Features*):** Sistem hanya dapat belajar jika data pelatihan mengandung fitur yang relevan. Proses *feature engineering* (pemilihan fitur, ekstraksi fitur, pembuatan fitur baru) sangat penting. 

### b. Algoritma Buruk

* **Overfitting Data Pelatihan:** Model berkinerja sangat baik pada data pelatihan, tetapi gagal menggeneralisasi ke data baru. Ini terjadi ketika model terlalu kompleks relatif terhadap jumlah atau *noise* data pelatihan. 
    * *Solusi:* Sederhanakan model (kurangi parameter, atribut, atau batasi model/gunakan *regularization*), kumpulkan lebih banyak data, atau kurangi *noise* data. 
    * *Regularization* adalah tindakan membatasi model agar lebih sederhana untuk mengurangi risiko *overfitting*. Tingkat *regularization* dikontrol oleh *hyperparameter*. 
* **Underfitting Data Pelatihan:** Model terlalu sederhana untuk mempelajari struktur data yang mendasari, menghasilkan prediksi yang tidak akurat bahkan pada data pelatihan. 
    * *Solusi:* Pilih model yang lebih kuat, berikan fitur yang lebih baik (*feature engineering*), atau kurangi batasan model (misalnya, kurangi *hyperparameter regularization*). 

## 5. Pengujian dan Validasi

Setelah model dilatih, penting untuk mengevaluasinya.
* **Test Set:** Pisahkan data menjadi *training set* dan *test set*. Model dilatih pada *training set* dan diuji pada *test set* untuk mengestimasi *generalization error*. 
    * **Penting:** Jangan pernah melihat atau menyetel *hyperparameter* berdasarkan *test set* untuk menghindari *data snooping bias* (estimasi terlalu optimis). 
* **Hyperparameter Tuning & Model Selection:** Gunakan *validation set* (dipisahkan dari *training set*) untuk mengevaluasi beberapa model kandidat dan memilih yang terbaik, serta menyetel *hyperparameter*. Setelah itu, model terbaik dilatih pada seluruh *training set* dan dievaluasi sekali pada *test set*. 
    * *Cross-validation* (misalnya, K-fold cross-validation) adalah teknik yang lebih kuat untuk evaluasi model dan *hyperparameter tuning* karena memberikan estimasi kinerja yang lebih akurat. 
* **Data Mismatch:** Jika data pelatihan dan data produksi (yang diwakili oleh *validation set* dan *test set*) berbeda signifikan, model mungkin tidak menggeneralisasi dengan baik. Penggunaan *train-dev set* dapat membantu mendiagnosis masalah ini. 
* **No Free Lunch Theorem:** Menyatakan bahwa tidak ada model yang secara *a priori* dijamin bekerja lebih baik daripada yang lain untuk semua data. Pilihan model bergantung pada asumsi yang dibuat tentang data. 

Bab ini menekankan pentingnya pemahaman konsep dasar ML dan langkah-langkah dalam proyek ML untuk membangun sistem cerdas yang efektif.
