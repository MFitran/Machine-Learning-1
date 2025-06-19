# Rangkuman Bab 5: Support Vector Machines (SVM)

Bab 5 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" membahas secara mendalam tentang **Support Vector Machines (SVM)**, sebuah model Machine Learning yang sangat **kuat dan serbaguna**. SVM tidak hanya mampu melakukan **klasifikasi linier dan nonlinier**, tetapi juga dapat digunakan untuk **regresi** dan bahkan **deteksi outlier**. Model ini sangat populer dan cocok untuk **klasifikasi dataset berukuran kecil hingga menengah yang kompleks**.

Bab ini diorganisir untuk memberikan pemahaman menyeluruh tentang SVM, mulai dari konsep dasar hingga detail implementasi dan cara kerjanya di balik layar.

## 1. Klasifikasi SVM Linier

Bagian ini memperkenalkan ide fundamental SVM melalui **klasifikasi margin besar (large margin classification)**. Konsep utamanya adalah menemukan batas keputusan (garis atau hyperplane) yang tidak hanya memisahkan kelas-kelas data, tetapi juga **berada sejauh mungkin dari instansi pelatihan terdekat**. Instansi-instansi yang berada di "tepi jalan" ini disebut **vektor dukungan (support vectors)**, dan mereka adalah satu-satunya titik data yang memengaruhi batas keputusan.

Penting untuk dicatat bahwa SVM **sensitif terhadap skala fitur**; oleh karena itu, penskalaan fitur (seperti menggunakan `StandardScaler`) sangat krusial sebelum melatih model.

Untuk mengatasi keterbatasan klasifikasi margin keras (yang hanya berfungsi pada data yang terpisah secara linier sempurna dan sensitif terhadap outlier), diperkenalkan **klasifikasi margin lunak (soft margin classification)**. Ini mencari keseimbangan antara memiliki margin yang lebar dan membatasi pelanggaran margin. Hyperparameter **C** mengontrol trade-off ini: nilai `C` yang rendah mengizinkan lebih banyak pelanggaran margin untuk margin yang lebih lebar dan model yang lebih umum, sedangkan nilai `C` yang tinggi membatasi pelanggaran margin untuk model yang lebih kompleks.

## 2. Klasifikasi SVM Nonlinier

SVM juga dapat menangani data yang tidak terpisah secara linier. Ada dua pendekatan utama yang dibahas:

* **Menambahkan Fitur Polinomial**: Salah satu cara adalah dengan **menambahkan fitur-fitur baru berupa pangkat dari fitur asli** (fitur polinomial). Ini terkadang dapat mengubah dataset nonlinier menjadi terpisah secara linier di ruang fitur yang lebih tinggi.
* **Trik Kernel (Kernel Trick)**: Ini adalah teknik matematika yang **memungkinkan SVM bekerja seolah-olah fitur polinomial berderajat tinggi telah ditambahkan, tanpa benar-benar menambahkannya**. Ini menghindari "ledakan kombinatorial" fitur. Kelas `SVC` di Scikit-Learn mengimplementasikan trik kernel, termasuk **kernel polinomial** dan **kernel RBF Gaussian**. Hyperparameter seperti `degree` (untuk kernel polinomial), `coef0`, dan `gamma` (untuk kernel RBF) digunakan untuk menyetel kompleksitas model.

Bagian ini juga memberikan panduan untuk **memilih kernel**, menyarankan untuk **selalu mencoba kernel linier terlebih dahulu**, diikuti oleh kernel RBF Gaussian jika dataset tidak terlalu besar.

## 3. Regresi SVM

SVM tidak terbatas pada klasifikasi; ia juga mendukung tugas **regresi linier dan nonlinier**. Untuk regresi, tujuannya dibalik: alih-alih memaksimalkan margin tanpa pelanggaran, Regresi SVM mencoba **memasukkan sebanyak mungkin instansi ke dalam "jalan" (margin) sambil membatasi instansi yang berada di luar jalan**. Lebar "jalan" ini dikontrol oleh hyperparameter **`epsilon` ($\epsilon$)**. Model ini disebut **$\epsilon$-insentif** karena penambahan instansi di dalam margin tidak memengaruhi prediksi model. Kelas `LinearSVR` digunakan untuk regresi SVM linier, dan `SVR` (dengan trik kernel) untuk regresi SVM nonlinier.

## 4. Di Balik Layar

Bagian ini menyelami detail matematis SVM, menjelaskan:

* **Fungsi Keputusan dan Prediksi**: Bagaimana SVM linier menghitung output berdasarkan kombinasi bobot fitur dan bias.
* **Tujuan Pelatihan**: Formulasi masalah optimasi margin keras dan margin lunak, termasuk pengenalan **variabel kendur ($\zeta$)** untuk margin lunak dan peran hyperparameter `C`.
* **Pemrograman Kuadratik (QP)**: Mengidentifikasi masalah SVM sebagai masalah optimasi kuadratik cembung dengan kendala linier.
* **Masalah Ganda (Dual Problem)**: Menjelaskan bentuk dual dari masalah optimasi SVM, yang **lebih cepat dipecahkan ketika jumlah instansi pelatihan lebih kecil dari jumlah fitur**, dan yang **memungkinkan trik kernel**.
* **SVM Berkernel**: Menunjukkan secara matematis bagaimana trik kernel memungkinkan perhitungan produk dot di ruang fitur berdimensi tinggi tanpa transformasi eksplisit, menggunakan contoh kernel polinomial.
* **SVM Online**: Singkatnya membahas bagaimana SVM dapat diimplementasikan untuk pembelajaran inkremental, seperti menggunakan `SGDClassifier` untuk meminimalkan fungsi biaya.
