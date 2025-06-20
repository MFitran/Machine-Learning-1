# Machine-Learning-1
## Pendahuluan

Dokumen ini merangkum pemahaman mendalam dan keterampilan praktis dalam mengimplementasikan konsep inti Machine Learning. Rangkuman ini didasarkan pada buku *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems* (O’Reilly) edisi ke-2, yang diperbarui untuk TensorFlow 2.

## Tujuan Pembelajaran

Tujuan utama dari pembelajaran ini adalah untuk:
* Memperdalam pemahaman teoretis tentang berbagai algoritma Machine Learning.
* Mengembangkan keterampilan praktis dalam mengimplementasikan algoritma-algoritma tersebut menggunakan *framework* Python siap produksi seperti Scikit-Learn, Keras, dan TensorFlow.
* Mampu mereproduksi kode dan memahami penjelasan konseptual secara terstruktur.
* Mampu mengidentifikasi dan mengatasi tantangan umum dalam proyek Machine Learning.

## Alat dan Bahan (Prasyarat)

Untuk mengikuti pembelajaran ini, prasyarat yang dibutuhkan adalah:
* **Pengalaman Pemrograman Python:** Familiar dengan dasar-dasar Python.
* **Pustaka Ilmiah Python:** Familiar dengan NumPy, Pandas, dan Matplotlib.
* **Matematika Tingkat Perguruan Tinggi:** Pemahaman dasar kalkulus, aljabar linier, probabilitas, dan statistik (terutama untuk memahami detail di balik layar).
* **Lingkungan Kerja:** Jupyter Notebook, TensorFlow 2, Scikit-Learn, Keras.

## Struktur Pembelajaran (Roadmap)

Pembelajaran ini terbagi menjadi dua bagian utama:

### Bagian I: Dasar-dasar Machine Learning

1.  **Landscape Machine Learning (Bab 1)**
    * **Definisi Machine Learning:** Ilmu dan seni memprogram komputer agar dapat belajar dari data.
    * **Mengapa Menggunakan Machine Learning:** Otomatisasi, penanganan masalah kompleks, adaptasi lingkungan yang fluktuatif, dan penemuan *insight* dari data besar.
    * **Jenis Sistem Machine Learning:**
        * **Pembelajaran Terawasi (Supervised Learning):** Data pelatihan memiliki label (solusi yang diinginkan). Contoh: Klasifikasi (memprediksi kelas diskrit, seperti *spam*/*bukan spam*) dan Regresi (memprediksi nilai numerik, seperti harga rumah). Algoritma populer: K-Nearest Neighbors, Linear Regression, Logistic Regression, Support Vector Machines (SVMs), Decision Trees, Random Forests, Neural networks. 
        * **Pembelajaran Tanpa Terawasi (Unsupervised Learning):** Data pelatihan tidak memiliki label. Contoh: Clustering (mengelompokkan *instance* serupa), Anomaly/Novelty Detection (mendeteksi *instance* tidak biasa), Visualisasi & Dimensionality Reduction (menyederhanakan data tanpa kehilangan informasi signifikan), Association Rule Learning (menemukan hubungan antar *attribute*). 
        * **Pembelajaran Semi-terawasi (Semisupervised Learning):** Kombinasi pembelajaran terawasi dan tanpa terawasi (sebagian data berlabel, sebagian tidak). 
        * **Pembelajaran Penguatan (Reinforcement Learning):** Sistem (agen) belajar dari interaksi dengan lingkungan, mengambil tindakan dan menerima hadiah atau penalti, untuk memaksimalkan hadiah dari waktu ke waktu. 
    * **Pembelajaran *Batch* vs. *Online*:**
        * ***Batch Learning* (Offline):** Sistem dilatih menggunakan semua data yang tersedia secara *offline* dan tidak belajar secara inkremental. Membutuhkan banyak waktu dan sumber daya komputasi. 
        * ***Online Learning* (Incremental):** Sistem dilatih secara inkremental dengan menyalurkan *instance* data secara berurutan. Cocok untuk data berkelanjutan atau sumber daya terbatas. 
    * **Pembelajaran Berbasis *Instance* vs. Berbasis Model:**
        * **Berbasis *Instance*:** Sistem belajar dengan mengingat contoh dan menggeneralisasi kasus baru dengan membandingkannya menggunakan ukuran kesamaan. 
        * **Berbasis Model:** Sistem membangun model dari contoh dan menggunakannya untuk membuat prediksi. 
    * **Tantangan Utama Machine Learning:** Kualitas data buruk (*insufficient quantity, nonrepresentative, poor-quality, irrelevant features*), model terlalu sederhana (*underfitting*), model terlalu kompleks (*overfitting*). 
    * **Pengujian dan Validasi:** Membagi data menjadi *training set*, *validation set*, dan *test set* untuk evaluasi model dan *fine-tuning* *hyperparameter*. 

2.  **Proyek Machine Learning *End-to-End* (Bab 2)**
    * **Kerangka Masalah:** Memahami tujuan bisnis, mengidentifikasi jenis masalah (regresi, klasifikasi, dll.), dan memilih strategi pembelajaran. 
    * **Pengukuran Kinerja:** Memilih metrik yang tepat (contoh: *Root Mean Square Error - RMSE* untuk regresi). 
    * **Mendapatkan Data:** Mengunduh dan memuat data, serta memahami strukturnya (*quick look at data structure*). 
    * **Membuat *Test Set*:** Penting untuk memisahkan *test set* di awal untuk menghindari *data snooping bias*. 
    * **Eksplorasi dan Visualisasi Data:** Memvisualisasikan data (misal: dengan *scatterplot*, histogram) untuk mendapatkan *insight* dan mencari korelasi antar *attribute*. 
    * **Persiapan Data:** Fungsi untuk pembersihan data (*data cleaning*, penanganan nilai yang hilang), penanganan *attribute* tekstual dan kategorikal (*one-hot encoding*), *feature scaling* (min-max scaling, standardisasi), dan membangun *pipeline* transformasi. 
    * **Pemilihan dan Pelatihan Model:** Melatih berbagai model (misal: Linear Regression, Decision Tree, Random Forest) dan mengevaluasi kinerjanya menggunakan *cross-validation*. 
    * **Penyempurnaan Model (*Fine-tuning*):** Mengoptimalkan *hyperparameter* menggunakan *Grid Search* atau *Randomized Search*. 
    * **Analisis Model Terbaik dan Kesalahannya:** Memeriksa pentingnya *feature* (*feature importance*) dan menganalisis jenis kesalahan yang dibuat model (misal: melalui *confusion matrix*). 
    * **Evaluasi Sistem pada *Test Set*:** Menguji kinerja akhir model pada *test set* yang belum pernah dilihat sebelumnya. 
    * **Peluncuran, Pemantauan, dan Pemeliharaan Sistem:** Mengotomasikan proses *deployment*, memantau kinerja secara berkala (*model rot*), dan mengelola pembaruan data serta pelatihan ulang model. 

3.  **Klasifikasi (Bab 3)**
    * **MNIST Dataset:** Pengenalan dataset standar untuk klasifikasi digit tulisan tangan. 
    * **Pelatihan Pengklasifikasi Biner (*Binary Classifier*):** Membangun model yang membedakan dua kelas (misal: digit 5 vs. bukan 5). 
    * **Pengukuran Kinerja:**
        * **Akurasi:** Rasio prediksi yang benar (seringkali tidak memadai untuk dataset yang miring). 
        * ***Confusion Matrix*:** Matriks yang menunjukkan jumlah *instance* dari kelas A yang diklasifikasikan sebagai kelas B (True Negatives, False Positives, False Negatives, True Positives). 
        * **Presisi (Precision) & Rekall (Recall):**
            * Presisi: Akurasi prediksi positif (TP / (TP + FP)). 
            * Rekall: Rasio *instance* positif yang terdeteksi dengan benar (TP / (TP + FN)). 
            * *Precision/Recall Trade-off*: Meningkatkan presisi cenderung menurunkan *recall*, dan sebaliknya. 
        * **F1 Score:** Rata-rata harmonis presisi dan *recall*. 
        * **Kurva ROC (*Receiver Operating Characteristic*) & AUC (*Area Under the Curve*):** Memplot *True Positive Rate (recall)* terhadap *False Positive Rate*. AUC mengukur seberapa baik pengklasifikasi membedakan antar kelas. 
    * **Klasifikasi *Multiclass*:** Mengidentifikasi lebih dari dua kelas. Strategi: *One-versus-the-Rest (OvR)* atau *One-versus-One (OvO)*. 
    * **Analisis Kesalahan:** Memeriksa *confusion matrix* untuk memahami jenis kesalahan model. 
    * **Klasifikasi *Multilabel*:** Mengklasifikasikan *instance* ke dalam beberapa kelas sekaligus. 
    * **Klasifikasi *Multioutput*:** Generalisasi dari klasifikasi *multilabel* di mana setiap label dapat bersifat *multiclass*. 

4.  **Pelatihan Model (Bab 4)**
    * **Regresi Linier (*Linear Regression*):**
        * **Model:** $\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$. 
        * **Fungsi Biaya MSE:** Mengukur jarak antara prediksi model linier dan contoh pelatihan. 
        * **Persamaan Normal (*Normal Equation*):** Solusi *closed-form* untuk menemukan parameter optimal ($\theta = (X^T X)^{-1} X^T y$). 
        * **Kompleksitas Komputasi:** Persamaan Normal dan SVD menjadi lambat seiring bertambahnya jumlah *feature*. 
    * **Penurunan Gradien (*Gradient Descent*):** Algoritma optimasi iteratif untuk meminimalkan fungsi biaya. 
        * **Laju Pembelajaran (*Learning Rate*):** Ukuran langkah dalam *Gradient Descent*. Penting untuk memilih nilai yang tepat. 
        * **Penurunan Gradien *Batch* (*Batch Gradient Descent*):** Menggunakan seluruh *training set* untuk menghitung gradien di setiap langkah. Lambat untuk *training set* yang sangat besar. 
        * **Penurunan Gradien *Stochastic* (*Stochastic Gradient Descent - SGD*):** Memilih satu *instance* acak di *training set* di setiap langkah untuk menghitung gradien. Jauh lebih cepat, tetapi kurang stabil. 
        * **Penurunan Gradien *Mini-batch* (*Mini-batch Gradient Descent*):** Menghitung gradien pada kelompok kecil *instance* acak (*mini-batch*). Keseimbangan antara kecepatan SGD dan stabilitas *Batch Gradient Descent*. 
    * **Regresi Polinomial (*Polynomial Regression*):** Menambahkan pangkat setiap *feature* sebagai *feature* baru untuk model linier agar dapat menyesuaikan data nonlinier. 
    * **Kurva Pembelajaran (*Learning Curves*):** Memplot kinerja model pada *training set* dan *validation set* sebagai fungsi dari ukuran *training set* (atau iterasi pelatihan) untuk mendeteksi *overfitting* atau *underfitting*. 
    * ***Bias/Variance Trade-off*:** Keseimbangan antara kesalahan karena asumsi yang salah (*bias*) dan kesalahan karena sensitivitas berlebihan terhadap variasi data pelatihan (*variance*). 
    * **Model Linier Terderegulerisasi (*Regularized Linear Models*):** Mengendalikan kompleksitas model untuk mengurangi *overfitting*. 
        * ***Ridge Regression* (L2 Regularization):** Menambahkan $\alpha \sum \theta_i^2$ ke fungsi biaya. 
        * ***Lasso Regression* (L1 Regularization):** Menambahkan $\alpha \sum |\theta_i|$ ke fungsi biaya. Cenderung mengeliminasi bobot *feature* yang kurang penting (mengaturnya ke nol). 
        * ***Elastic Net*:** Campuran *Ridge* dan *Lasso*. Umumnya lebih disukai daripada *Lasso*. 
        * ***Early Stopping*:** Menghentikan pelatihan segera setelah kesalahan validasi mencapai minimum. 
    * **Regresi Logistik (*Logistic Regression*):** Digunakan untuk mengestimasi probabilitas suatu *instance* termasuk dalam kelas tertentu. 
        * **Fungsi Logistik (*Sigmoid Function*):** Fungsi S-shaped yang menghasilkan nilai antara 0 dan 1. 
        * **Fungsi Biaya (*Log Loss*):** Meminimalkan MSE antara probabilitas yang diestimasi dan kelas target. 
        * **Batas Keputusan (*Decision Boundaries*):** Model memprediksi kelas berdasarkan probabilitas yang diestimasi. 
    * **Regresi *Softmax* (*Softmax Regression*):** Generalisasi Regresi Logistik untuk mendukung banyak kelas secara langsung (menggunakan fungsi *softmax* di lapisan keluaran). 

5.  **Mesin Vektor Dukungan (*Support Vector Machines - SVMs*) (Bab 5)**
    * **Klasifikasi SVM Linier:** Menyesuaikan "jalur" terluas antara kelas-kelas (klasifikasi *large margin*). 
        * **Klasifikasi *Hard Margin*:** Secara ketat memaksakan semua *instance* harus berada di luar "jalur" dan di sisi yang benar. Rentan terhadap *outlier*. 
        * **Klasifikasi *Soft Margin*:** Mengizinkan pelanggaran *margin* (beberapa *instance* berada di dalam atau di sisi yang salah) untuk fleksibilitas yang lebih besar. Dikontrol oleh *hyperparameter* C. 
    * **Klasifikasi SVM Nonlinier:**
        * **Kernel Polinomial:** Menambahkan *feature* polinomial secara implisit menggunakan *kernel trick*. 
        * ***Similarity Features* (Kernel RBF Gaussian):** Menambahkan *feature* berdasarkan kesamaan dengan *landmark*. 
        * **Memilih *Kernel*:** Umumnya mencoba *kernel* linier terlebih dahulu, lalu *Gaussian RBF*, kemudian yang lain. 
    * **Kompleksitas Komputasi:** Berbeda antar kelas SVM (LinearSVC, SGDClassifier, SVC). 
    * **Regresi SVM:** Menggunakan SVM untuk tugas regresi, mencoba menyesuaikan *instance* sebanyak mungkin di dalam "jalur". 
    * **Di Balik Layar:** Menjelaskan fungsi keputusan, tujuan pelatihan (*training objective*), *Quadratic Programming*, dan *Dual Problem*. 

6.  **Pohon Keputusan (*Decision Trees*) (Bab 6)**
    * **Pelatihan dan Visualisasi:** Membangun dan memvisualisasikan pohon keputusan. 
    * **Membuat Prediksi:** Cara pohon keputusan membuat prediksi (mulai dari *root node* hingga *leaf node*). 
    * **Interpretasi Model (White Box vs. Black Box):** Pohon Keputusan adalah model "kotak putih" yang mudah diinterpretasikan. 
    * **Mengestimasi Probabilitas Kelas:** Pohon keputusan dapat mengestimasi probabilitas suatu *instance* termasuk dalam kelas tertentu. 
    * **Algoritma Pelatihan CART:** Algoritma yang digunakan Scikit-Learn untuk melatih pohon keputusan, membagi *training set* untuk meminimalkan *impurity*. 
    * **Kompleksitas Komputasi:** Prediksi sangat cepat (O(log(m))), tetapi pelatihan lebih lambat (O(n x m log(m))). 
    * ***Gini Impurity* atau *Entropy*?** Kedua ukuran kemurnian umumnya menghasilkan pohon serupa, tetapi *Gini impurity* sedikit lebih cepat. 
    * ***Hyperparameter* Regularisasi:** Mengendalikan kompleksitas pohon untuk menghindari *overfitting* (misal: `max_depth`, `min_samples_leaf`). 
    * **Regresi:** Pohon keputusan juga dapat melakukan tugas regresi dengan meminimalkan MSE. 
    * **Instabilitas:** Pohon Keputusan sangat sensitif terhadap variasi kecil dalam data pelatihan dan orientasi data. 

7.  **Pembelajaran *Ensemble* dan *Random Forests* (Bab 7)**
    * ***Voting Classifiers*:** Menggabungkan prediksi beberapa pengklasifikasi (misal: Logistic Regression, SVM, Random Forest) untuk mendapatkan kinerja yang lebih baik. 
        * ***Hard Voting*:** Memilih kelas dengan suara terbanyak. 
        * ***Soft Voting*:** Menggunakan probabilitas kelas yang diestimasi dan memilih kelas dengan probabilitas rata-rata tertinggi. 
    * ***Bagging* dan *Pasting*:** Melatih beberapa prediktor menggunakan algoritma yang sama pada subset acak yang berbeda dari *training set*. 
        * ***Bagging* (dengan penggantian):** Mengambil sampel dengan penggantian (*bootstrap aggregating*). 
        * ***Pasting* (tanpa penggantian):** Mengambil sampel tanpa penggantian. 
        * **Implementasi di Scikit-Learn:** Menggunakan kelas `BaggingClassifier` (atau `BaggingRegressor`). 
        * ***Out-of-Bag Evaluation*:** Mengevaluasi prediktor menggunakan *instance* yang tidak digunakan dalam pelatihannya. 
        * ***Random Patches* & *Random Subspaces*:** Mengambil sampel *feature* di samping *instance*. 
    * ***Random Forests*:** *Ensemble* dari Pohon Keputusan, biasanya dilatih melalui metode *bagging*. 
        * ***Extra-Trees*:** Pohon yang lebih acak (menggunakan *threshold* acak untuk setiap *feature*) yang lebih cepat dilatih. 
        * ***Feature Importance*:** *Random Forests* dapat mengukur kepentingan relatif setiap *feature*. 
    * ***Boosting*:** Metode *ensemble* yang menggabungkan beberapa pelajar lemah menjadi pelajar yang kuat, dengan melatih prediktor secara berurutan, setiap prediktor mencoba mengoreksi pendahulunya. 
        * ***AdaBoost* (Adaptive Boosting):** Memberi perhatian lebih pada *instance* pelatihan yang salah diklasifikasikan oleh prediktor sebelumnya. 
        * ***Gradient Boosting*:** Melatih prediktor baru untuk menyesuaikan kesalahan residual yang dibuat oleh prediktor sebelumnya. 
        * ***XGBoost*:** Implementasi *Gradient Boosting* yang sangat dioptimalkan. 
    * ***Stacking* (Stacked Generalization):** Melatih model untuk melakukan agregasi prediksi dari semua prediktor dalam suatu *ensemble*. 

### Bagian II: Jaringan Saraf dan Pembelajaran Mendalam (*Neural Networks & Deep Learning*)

1.  **Pengantar Jaringan Saraf Tiruan dengan Keras (Bab 10)**
    * **Dari Neuron Biologis ke Artifisial:** Inspirasi dari arsitektur otak. 
        * **Neuron Biologis:** Sel-sel otak yang berkomunikasi melalui impuls listrik. 
        * **Komputasi Logis dengan Neuron:** Model sederhana neuron artifisial dapat melakukan komputasi logis. 
    * ***Perceptron*:** Arsitektur ANN paling sederhana, berdasarkan *threshold logic unit (TLU)*. 
        * **Pelatihan *Perceptron*:** Menggunakan varian dari aturan Hebb untuk memperkuat koneksi. 
        * **Kelemahan *Perceptron*:** Tidak mampu menyelesaikan masalah nonlinier sederhana (misal: XOR). 
    * ***Multilayer Perceptron (MLP)* dan *Backpropagation*:**
        * **Arsitektur MLP:** Lapisan masukan, satu atau lebih lapisan tersembunyi (*hidden layers*), dan lapisan keluaran (*output layer*). 
        * ***Deep Neural Networks (DNNs)*:** MLP dengan tumpukan lapisan tersembunyi yang dalam. 
        * **Algoritma *Backpropagation*:** Penurunan Gradien yang menggunakan teknik efisien untuk menghitung gradien secara otomatis (*autodiff*). 
        * **Fungsi Aktivasi Nonlinier:** Penting untuk dapat menyelesaikan masalah kompleks (misal: *logistic/sigmoid*, *tanh*, *ReLU*). 
    * **MLP untuk Regresi:** Arsitektur umum untuk tugas regresi. 
    * **MLP untuk Klasifikasi:** Arsitektur umum untuk tugas klasifikasi (biner, *multilabel*, *multiclass*). 
    * **Implementasi MLP dengan Keras:**
        * **Menginstal TensorFlow 2:** Pustaka komputasi numerik yang kuat. 
        * **Membangun Pengklasifikasi Gambar Menggunakan API Sekuensial:** Langkah-langkah membuat, mengkompilasi, melatih, mengevaluasi, dan menggunakan model. 
        * **Membangun Model Kompleks Menggunakan API Fungsional:** Untuk arsitektur non-sekuensial, banyak masukan/keluaran (misal: Wide & Deep Network). 
        * **Menggunakan API *Subclassing* untuk Membangun Model Dinamis:** Memberikan fleksibilitas lebih untuk perilaku model yang dinamis. 
        * **Menyimpan dan Memulihkan Model:** Cara menyimpan model terlatih. 
        * **Menggunakan *Callbacks*:** Objek yang dipanggil Keras selama pelatihan (misal: *ModelCheckpoint*, *EarlyStopping*, *TensorBoard*). 
        * **Menggunakan *TensorBoard* untuk Visualisasi:** Alat visualisasi interaktif untuk memantau kurva pembelajaran, grafik komputasi, dll. 
    * **Penyempurnaan *Hyperparameter* Jaringan Saraf:** Mencari kombinasi *hyperparameter* terbaik (misal: jumlah lapisan tersembunyi, jumlah neuron per lapisan, laju pembelajaran, ukuran *batch*, fungsi aktivasi). 

2.  **Pelatihan Jaringan Saraf Dalam (*Deep Neural Networks*) (Bab 11)**
    * ***Vanishing/Exploding Gradients Problems*:** Gradien menjadi terlalu kecil atau terlalu besar saat mengalir mundur melalui DNN, membuat lapisan bawah sulit dilatih. 
        * ***Glorot* dan *He Initialization*:** Strategi inisialisasi bobot yang mengurangi masalah gradien tidak stabil. 
        * **Fungsi Aktivasi Non-saturasi:** Fungsi seperti *ReLU* dan variannya (misal: *Leaky ReLU*, *PReLU*, *ELU*, *SELU*) membantu mengurangi masalah gradien. 
        * ***Batch Normalization (BN)*:** Menambahkan operasi yang menstandardisasi input lapisan, lalu menskalakan dan menggeser hasilnya. Mempercepat pelatihan dan bertindak sebagai regularisasi. 
        * ***Gradient Clipping*:** Memotong gradien agar tidak melebihi *threshold* tertentu. 
    * **Menggunakan Kembali Lapisan *Pretrained* (*Transfer Learning*):** Menggunakan kembali lapisan bawah dari jaringan saraf yang sudah dilatih pada tugas serupa untuk mempercepat pelatihan dan mengurangi kebutuhan data berlabel. 
    * **Pelatihan Tanpa Pengawasan (*Unsupervised Pretraining*):** Melatih model tanpa pengawasan (misal: *autoencoder* atau GAN) pada data tidak berlabel, lalu menggunakan lapisan bawahnya untuk tugas sebenarnya. 
    * **Pelatihan pada Tugas Tambahan (*Auxiliary Task*):** Melatih jaringan saraf pada tugas yang mudah mendapatkan data berlabel, lalu menggunakan lapisan bawahnya untuk tugas utama. 
    * **Optimizer yang Lebih Cepat:** 
        * ***Momentum Optimization*:** Menggunakan momentum untuk mempercepat penurunan gradien. 
        * ***Nesterov Accelerated Gradient (NAG)*:** Varian *momentum optimization* yang lebih cepat. 
        * ***AdaGrad*:** Menyesuaikan laju pembelajaran untuk setiap parameter, menurunkan skala gradien sepanjang dimensi yang curam. 
        * ***RMSProp*:** Memperbaiki *AdaGrad* dengan hanya mengakumulasi gradien dari iterasi terbaru. 
        * ***Adam* dan *Nadam* Optimization:** Menggabungkan ide *momentum optimization* dan *RMSProp*. 
    * **Penjadwalan Laju Pembelajaran (*Learning Rate Scheduling*):** Mengubah laju pembelajaran selama pelatihan (misal: *Power scheduling*, *Exponential scheduling*, *Piecewise constant scheduling*, *Performance scheduling*, *1cycle scheduling*). 
    * **Menghindari *Overfitting* Melalui Regularisasi:** 
        * **Regularisasi L1 dan L2:** Menambahkan *term* penalti pada fungsi biaya untuk membatasi bobot model. 
        * ***Dropout*:** Secara acak "menjatuhkan" neuron selama pelatihan untuk mencegah *co-adaptation* dan meningkatkan robustnes model. 
        * ***Monte Carlo (MC) Dropout*:** Meningkatkan kinerja model *dropout* yang terlatih dan memberikan ukuran ketidakpastian model yang lebih baik. 
        * ***Max-Norm Regularization*:** Membatasi bobot koneksi masuk setiap neuron. 

3.  **Model Kustom dan Pelatihan dengan TensorFlow (Bab 12)**
    * **Pengantar Singkat TensorFlow:** Pustaka komputasi numerik yang kuat dengan dukungan GPU, komputasi terdistribusi, *just-in-time compiler*, *autodiff*, dan *optimizers* yang canggih. 
    * **Menggunakan TensorFlow Seperti NumPy:** Bekerja dengan *tensor* dan operasi. 
        * ***Tensors* dan Operasi:** Mirip dengan *NumPy ndarray*, tetapi *immutable*. 
        * ***Tensors* dan NumPy:** Integrasi yang baik antara *tensor* TensorFlow dan *NumPy array*. 
        * **Konversi Tipe:** TensorFlow tidak melakukan konversi tipe secara otomatis untuk menghindari penurunan kinerja. 
        * ***Variables*:** *Tensor* yang dapat dimodifikasi (*mutable*) untuk bobot jaringan saraf. 
        * **Struktur Data Lainnya:** *Sparse tensors*, *tensor arrays*, *ragged tensors*, *string tensors*, *sets*, *queues*. 
    * **Menyesuaikan Model dan Algoritma Pelatihan:** 
        * **Fungsi *Loss* Kustom:** Membuat fungsi *loss* khusus (misal: *Huber loss*). 
        * **Menyimpan dan Memuat Model yang Mengandung Komponen Kustom:** Memastikan komponen kustom dapat disimpan dan dimuat bersama model. 
        * **Fungsi Aktivasi, Inisialisasi, Regularisasi, dan Batasan Kustom:** Membuat fungsi kustom untuk berbagai aspek model. 
        * **Metrik Kustom:** Membuat metrik khusus (misal: metrik *streaming*). 
        * **Lapisan Kustom (*Custom Layers*):** Membangun lapisan neural network sendiri. 
        * **Model Kustom (*Custom Models*):** Membangun arsitektur model yang kompleks. 
        * **Fungsi *Loss* dan Metrik Berbasis Internal Model:** Mendefinisikan *loss* atau metrik berdasarkan bobot atau aktivasi lapisan tersembunyi. 
    * **Menghitung Gradien Menggunakan *Autodiff*:** Memanfaatkan `tf.GradientTape` untuk menghitung gradien secara otomatis. 
    * **Loop Pelatihan Kustom (*Custom Training Loops*):** Menulis loop pelatihan sendiri untuk fleksibilitas maksimal. 
    * **Fungsi dan Grafik TensorFlow:** Mengubah fungsi Python menjadi *TF Function* (`@tf.function`) untuk meningkatkan kinerja dan portabilitas. 
        * ***AutoGraph* dan *Tracing*:** Bagaimana TensorFlow menganalisis kode Python untuk membangun grafik komputasi. 
        * **Aturan *TF Function*:** Panduan untuk menulis fungsi yang dapat dikonversi dengan baik ke *TF Function*. 

4.  **Memuat dan Memproses Data dengan TensorFlow (Bab 13)**
    * **API Data (`tf.data`):** Fokus pada konsep *dataset* sebagai urutan item data. 
        * **Rantai Transformasi:** Menerapkan berbagai transformasi (*map, batch, repeat, filter*) secara berantai pada *dataset*. 
        * **Mengacak Data (*Shuffling*):** Menggunakan `shuffle()` untuk memastikan *instance* pelatihan IID. 
        * **Menginterleave Baris dari Beberapa File:** Membaca dari beberapa file secara bersamaan untuk *shuffling* yang lebih baik. 
    * **Pra-pemrosesan Data:** 
        * ***Feature Scaling*:** Mengimplementasikan lapisan standardisasi. 
        * **Encoding *Categorical Features* Menggunakan *One-Hot Vectors*:** Mengubah kategori menjadi representasi biner. 
        * **Encoding *Categorical Features* Menggunakan *Embeddings*:** Representasi kategori sebagai vektor padat yang dapat dilatih. 
    * **Menggabungkan Semuanya:** Fungsi pembantu untuk memuat, memproses, mengacak, dan *batching* data secara efisien. 
    * ***Prefetching*:** Mengambil *batch* data berikutnya secara paralel saat *batch* saat ini sedang diproses. 
    * **Menggunakan *Dataset* dengan `tf.keras`:** Mengintegrasikan *dataset* yang telah disiapkan dengan model Keras. 
    * **Format TFRecord:** Format biner pilihan TensorFlow untuk menyimpan data dalam jumlah besar secara efisien. 
        * **File TFRecord Terkompresi:** Menggunakan kompresi untuk mengurangi ukuran file. 
        * **Pengantar Singkat *Protocol Buffers*:** Format biner portabel dan efisien untuk menyimpan data terstruktur. 
        * ***TensorFlow Protobufs*:** Menggunakan *protobuf* `Example` dan `SequenceExample` untuk merepresentasikan *instance* data. 
        * **Memuat dan Mem-parsing *Example*:** Menggunakan `tf.io.parse_single_example()` atau `tf.io.parse_example()` untuk memproses data dari TFRecord. 
    * **Lapisan Pra-pemrosesan Keras:** Lapisan standar untuk pra-pemrosesan data secara langsung di dalam model. 
    * ***TF Transform*:** Mendefinisikan operasi pra-pemrosesan sekali dan menjalankannya secara *batch* sebelum pelatihan, serta mengekspornya ke *TF Function* untuk *deployment*. 
    * **Proyek *TensorFlow Datasets (TFDS)*:** Mengunduh dataset umum dengan mudah. 

5.  **Visi Komputer Mendalam Menggunakan Jaringan Saraf Konvolusional (Bab 14)**
    * **Arsitektur Korteks Visual:** Inspirasi dari neuron biologis dengan *receptive fields* lokal. 
    * **Lapisan Konvolusional (*Convolutional Layers*):** Blok pembangun utama CNN. 
        * ***Receptive Fields*, *Stride*, *Padding*:** Konsep-konsep penting dalam lapisan konvolusional. 
        * ***Filters*:** Bobot neuron yang merepresentasikan pola kecil. 
        * **Menumpuk Banyak *Feature Maps*:** Lapisan konvolusional memiliki banyak *filter*, menghasilkan banyak *feature map*. 
        * **Implementasi TensorFlow:** Menggunakan `tf.nn.conv2d()` dan `keras.layers.Conv2D`. 
        * **Kebutuhan Memori:** Lapisan konvolusional membutuhkan banyak RAM, terutama selama pelatihan. 
    * **Lapisan *Pooling*:** Mengecilkan gambar masukan untuk mengurangi beban komputasi, penggunaan memori, dan jumlah parameter. 
        * ***Max Pooling*:** Jenis *pooling* paling umum, hanya memilih nilai maksimum. 
        * **Implementasi TensorFlow:** Menggunakan `keras.layers.MaxPool2D` atau `tf.nn.max_pool()`. 
        * ***Global Average Pooling*:** Menghitung rata-rata seluruh *feature map*. 
    * **Arsitektur CNN:** 
        * **LeNet-5 (1998):** Arsitektur CNN klasik untuk pengenalan digit tulisan tangan. 
        * **AlexNet (2012):** Jauh lebih besar dan lebih dalam dari LeNet-5, menggunakan *dropout* dan *data augmentation*. 
        * ***GoogLeNet* (2014):** Menggunakan *inception modules* untuk efisiensi parameter. 
        * ***VGGNet* (2014):** Arsitektur sederhana dengan banyak lapisan konvolusional dan *filter* kecil. 
        * ***ResNet* (Residual Network) (2015):** Menggunakan *skip connections* untuk melatih jaringan yang sangat dalam. 
        * ***Xception* (2016):** Menggunakan lapisan konvolusional yang dapat dipisahkan secara mendalam (*depthwise separable convolution*). 
        * ***SENet* (Squeeze-and-Excitation Network) (2017):** Menambahkan blok SE untuk mengkalibrasi ulang *feature maps*. 
        * **Mengimplementasikan ResNet-34 CNN Menggunakan Keras:** Contoh implementasi. 
    * **Menggunakan Model *Pretrained* dari Keras:** Memuat model yang sudah dilatih sebelumnya (misal: ResNet-50) untuk tugas klasifikasi gambar. 
    * **Model *Pretrained* untuk *Transfer Learning*:** Menggunakan lapisan bawah model *pretrained* untuk tugas baru dengan data pelatihan terbatas. 
    * **Klasifikasi dan Lokalisasi:** Memprediksi *bounding box* di sekitar objek sebagai tugas regresi. 
    * **Deteksi Objek (*Object Detection*):** Mengklasifikasikan dan melokalisasi banyak objek dalam sebuah gambar. 
        * ***Fully Convolutional Networks (FCNs)*:** Mengganti lapisan *dense* dengan lapisan konvolusional untuk memproses gambar ukuran berapa pun. 
        * ***You Only Look Once (YOLO)*:** Arsitektur deteksi objek yang sangat cepat dan akurat. 
        * ***Mean Average Precision (mAP)*:** Metrik umum untuk mengevaluasi sistem deteksi objek. 
    * **Segmentasi Semantik (*Semantic Segmentation*):** Mengklasifikasikan setiap piksel berdasarkan kelas objeknya. 
        * ***Transposed Convolutional Layer*:** Digunakan untuk *upsampling* (meningkatkan ukuran gambar). 

6.  **Memproses Urutan Menggunakan RNN dan CNN (Bab 15)**
    * **Neuron dan Lapisan Berulang (*Recurrent Neurons and Layers*):** Jaringan saraf dengan koneksi mundur, memungkinkan "memori" temporal. 
    * **Sel Memori (*Memory Cells*):** Bagian dari jaringan saraf yang mempertahankan beberapa keadaan lintas langkah waktu. 
    * **Urutan Masukan dan Keluaran (*Input and Output Sequences*):** RNN dapat mengambil urutan masukan dan menghasilkan urutan keluaran. 
        * **Urutan-ke-Urutan (*Sequence-to-Sequence*):** Input dan output adalah urutan. 
        * **Urutan-ke-Vektor (*Sequence-to-Vector*):** Input adalah urutan, output adalah vektor tunggal. 
        * **Vektor-ke-Urutan (*Vector-to-Sequence*):** Input adalah vektor tunggal, output adalah urutan. 
        * ***Encoder–Decoder*:** Urutan-ke-vektor diikuti oleh vektor-ke-urutan. 
    * **Pelatihan RNN:** Menggunakan *backpropagation through time (BPTT)*. 
    * **Peramalan Rangkaian Waktu (*Time Series Forecasting*):** Memprediksi nilai masa depan dalam rangkaian waktu. 
        * **Metrik *Baseline*:** Pendekatan naive atau model linier sederhana sebagai perbandingan. 
        * **Mengimplementasikan RNN Sederhana:** Model RNN paling dasar. 
        * **RNN Mendalam (*Deep RNNs*):** Menumpuk beberapa lapisan *recurrent*. 
        * **Meramalkan Beberapa Langkah Waktu ke Depan:** Memprediksi beberapa nilai sekaligus. 
    * **Menangani Urutan Panjang:** 
        * **Melawan Masalah Gradien Tidak Stabil:** Menggunakan teknik seperti *Gradient Clipping* dan *Layer Normalization*. 
        * **Mengatasi Masalah Memori Jangka Pendek:** Menggunakan sel memori jangka panjang. 
            * **Sel LSTM (*Long Short-Term Memory*):** Sel yang dapat belajar mengenali masukan penting, menyimpannya dalam keadaan jangka panjang, dan mengekstraknya saat dibutuhkan. 
            * **Sel GRU (*Gated Recurrent Unit*):** Versi sederhana dari sel LSTM. 
        * **Menggunakan Lapisan Konvolusional 1D untuk Memproses Urutan:** Mempersingkat urutan masukan, membantu lapisan GRU mendeteksi pola yang lebih panjang. 
        * ***WaveNet*:** Arsitektur yang menumpuk lapisan konvolusional 1D dengan tingkat dilatasi yang berlipat ganda, memungkinkan pemrosesan urutan yang sangat panjang secara efisien. 

7.  **Pemrosesan Bahasa Alami dengan RNN dan *Attention* (Bab 16)**
    * **Membuat Teks Shakespearean Menggunakan *Character RNN*:** 
        * **Membuat *Training Dataset*:** Mengkodekan setiap karakter sebagai bilangan bulat. 
        * **Cara Membagi *Dataset* Sekuensial:** Penting untuk menghindari tumpang tindih antara *training*, *validation*, dan *test set*. 
        * **Memotong *Dataset* Sekuensial menjadi Banyak *Window*:** Menggunakan `window()` untuk mengubah urutan panjang menjadi *window* teks yang lebih kecil (*truncated backpropagation through time*). 
        * **Membangun dan Melatih Model *Char-RNN*:** Menggunakan lapisan GRU dan *softmax* keluaran. 
        * **Menggunakan Model *Char-RNN*:** Memprediksi karakter berikutnya. 
        * **Membuat Teks Shakespearean Palsu:** Menggunakan model untuk menghasilkan teks baru. 
    * ***Stateful RNN*:** Mempertahankan keadaan tersembunyi antara iterasi pelatihan untuk mempelajari pola jangka panjang. 
    * **Analisis Sentimen:** Mengklasifikasikan ulasan film sebagai positif atau negatif. 
        * **Preprocessing Teks:** Menggunakan `Tokenizer` atau teknik *subword tokenization*. 
        * ***Masking*:** Memberi tahu model untuk mengabaikan token *padding*. 
        * **Menggunakan *Embeddings Pretrained*:** Menggunakan model komponen yang sudah dilatih (`tensorflow_hub`). 
    * **Jaringan *Encoder–Decoder* untuk Terjemahan Mesin Saraf (*Neural Machine Translation - NMT*):** 
        * Model: *Encoder* mengkodekan kalimat masukan, *decoder* mengeluarkan terjemahan. 
        * ***Bidirectional RNNs*:** Menggunakan dua lapisan *recurrent* (kiri-ke-kanan dan kanan-ke-kiri) untuk melihat konteks maju dan mundur. 
        * ***Beam Search*:** Algoritma pencarian untuk menemukan terjemahan terbaik dengan melacak daftar kalimat paling menjanjikan. 
    * **Mekanisme *Attention*:** Memungkinkan *decoder* untuk fokus pada kata-kata yang relevan dari masukan pada setiap langkah waktu. 
        * ***Bahdanau Attention* (Concatenative/Additive Attention):** Menghitung skor keselarasan berdasarkan gabungan keluaran *encoder* dan keadaan *hidden* *decoder*. 
        * ***Luong Attention* (Multiplicative Attention):** Menghitung kesamaan menggunakan *dot product*. 
        * ***Visual Attention*:** Digunakan untuk menghasilkan *caption* gambar. 
        * ***Explainability*:** Memahami mengapa model menghasilkan keluaran tertentu. 
    * ***Attention Is All You Need: The Transformer Architecture*:** Arsitektur yang merevolusi NMT tanpa lapisan *recurrent* atau konvolusional, hanya mekanisme *attention*. 
        * ***Positional Embeddings*:** Vektor padat yang mengkodekan posisi kata dalam kalimat. 
        * ***Multi-Head Attention*:** Banyak lapisan *attention* yang berjalan secara paralel, masing-masing fokus pada karakteristik kata yang berbeda. 
    * **Inovasi Terbaru dalam Model Bahasa:** ELMo, ULMFiT, GPT, BERT. 

8.  **Pembelajaran Representasi dan Pembelajaran Generatif Menggunakan *Autoencoder* dan GAN (Bab 17)**
    * ***Autoencoder*:** Jaringan saraf tiruan yang belajar representasi padat dari data masukan (*latent representations/codings*) tanpa pengawasan. 
        * **Representasi Data yang Efisien:** Mempelajari pola untuk mengkodekan informasi secara efisien. 
        * **Melakukan PCA dengan *Undercomplete Linear Autoencoder*:** Jika *autoencoder* hanya menggunakan aktivasi linier dan fungsi biaya MSE, maka akan melakukan PCA. 
        * ***Stacked Autoencoders*:** *Autoencoder* dengan beberapa lapisan tersembunyi. 
            * **Visualisasi Rekonstruksi:** Membandingkan masukan dan keluaran. 
            * **Visualisasi *Fashion MNIST Dataset*:** Menggunakan *autoencoder* untuk mengurangi dimensi data untuk visualisasi. 
            * ***Unsupervised Pretraining* Menggunakan *Stacked Autoencoders*:** Melatih *autoencoder* pada data tidak berlabel, lalu menggunakan lapisannya untuk tugas klasifikasi. 
            * ***Tying Weights*:** Mengikat bobot lapisan *decoder* dengan lapisan *encoder* untuk mengurangi jumlah parameter. 
            * **Melatih Satu *Autoencoder* per Waktu:** Melatih *autoencoder* dangkal satu per satu lalu menumpuknya. 
        * ***Convolutional Autoencoders*:** Untuk menangani gambar, menggunakan lapisan konvolusional. 
        * ***Recurrent Autoencoders*:** Untuk menangani urutan, seperti rangkaian waktu atau teks. 
        * ***Denoising Autoencoders*:** Menambahkan *noise* ke masukan dan melatih model untuk memulihkan masukan asli yang bebas *noise*. 
        * ***Sparse Autoencoders*:** Menambahkan *term* penalti pada fungsi biaya untuk mengurangi jumlah neuron aktif di lapisan pengodean. 
        * ***Variational Autoencoders (VAEs)*:** *Autoencoder* probabilistik dan generatif yang dapat menghasilkan *instance* baru. 
            * **Menghasilkan Gambar *Fashion MNIST*:** Mengambil sampel *coding* acak dan mendeskodekannya menjadi gambar. 
            * **Interpolasi Semantik:** Melakukan interpolasi pada tingkat *coding* untuk menghasilkan gambar di antara dua gambar asli. 
    * **Jaringan Permusuhan Generatif (*Generative Adversarial Networks - GANs*) (Bab 17):** 
        * ***Generator*:** Mengambil distribusi acak sebagai masukan dan menghasilkan data (misal: gambar). 
        * ***Discriminator*:** Mengambil gambar palsu atau nyata sebagai masukan dan harus menebak apakah gambar itu palsu atau nyata. 
        * **Kesulitan Melatih GAN:** Rentan terhadap *mode collapse* dan ketidakstabilan pelatihan. 
        * ***Deep Convolutional GANs (DCGANs)*:** Pedoman untuk membangun GAN konvolusional yang stabil. 
        * ***Conditional GAN (CGAN)*:** Mengontrol kelas gambar yang dihasilkan. 
        * ***Progressive Growing of GANs*:** Menghasilkan gambar kecil di awal pelatihan, lalu secara bertahap menambahkan lapisan konvolusional untuk menghasilkan gambar yang lebih besar. 
        * ***StyleGANs*:** Menggunakan teknik *style transfer* untuk memastikan gambar yang dihasilkan memiliki struktur lokal yang sama dengan gambar pelatihan. 

9.  **Pembelajaran Penguatan (*Reinforcement Learning*) (Bab 18)**
    * **Belajar Mengoptimalkan Hadiah:** Agen belajar bertindak untuk memaksimalkan hadiah yang diharapkan. 
    * **Pencarian Kebijakan (*Policy Search*):** Algoritma yang digunakan agen untuk menentukan tindakannya. 
    * **Pengantar OpenAI Gym:** *Toolkit* yang menyediakan berbagai lingkungan simulasi untuk melatih agen. 
    * **Kebijakan Jaringan Saraf (*Neural Network Policies*):** Menggunakan jaringan saraf untuk mengestimasi probabilitas setiap tindakan. 
    * **Mengevaluasi Tindakan (*Credit Assignment Problem*):** Menilai tindakan berdasarkan jumlah semua hadiah yang datang setelahnya. 
    * **Penurunan Gradien Kebijakan (*Policy Gradients - PG*):** Mengoptimalkan parameter kebijakan dengan mengikuti gradien menuju hadiah yang lebih tinggi. 
    * **Proses Keputusan Markov (*Markov Decision Processes - MDPs*):** Model matematika untuk pengambilan keputusan di mana hasilnya sebagian acak dan sebagian dikendalikan oleh agen. 
        * **Persamaan Optimalitas Bellman:** Estimasi nilai keadaan optimal. 
        * **Algoritma Iterasi Nilai (*Value Iteration Algorithm*):** Menghitung nilai keadaan optimal secara iteratif. 
        * **Q-Values:** Nilai keadaan-tindakan optimal. 
    * **Pembelajaran Perbedaan Temporal (*Temporal Difference Learning - TD Learning*):** Mirip dengan Iterasi Nilai, tetapi memperbarui estimasi nilai keadaan berdasarkan transisi dan hadiah yang diamati. 
    * **Pembelajaran Q (*Q-Learning*):** Algoritma *TD Learning* yang melatih agen untuk mengestimasi Q-Values. 
        * **Kebijakan Eksplorasi (*Exploration Policies*):** Menggunakan kebijakan ε-greedy atau bonus untuk mendorong eksplorasi. 
    * **Pembelajaran Q Teraproksimasi dan Pembelajaran Q Mendalam (*Approximate Q-Learning & Deep Q-Learning - DQN*):** Menggunakan DNN untuk mengestimasi Q-Values. 
        * **Implementasi *Deep Q-Learning*:** Contoh implementasi untuk lingkungan CartPole. 
        * ***Catastrophic Forgetting*:** Model melupakan apa yang dipelajari di satu bagian lingkungan saat belajar di bagian lain. 
        * **Varian *Deep Q-Learning*:** 
            * ***Fixed Q-Value Targets*:** Menggunakan dua DQN (online dan target) untuk stabilitas. 
            * ***Double DQN*:** Memperbaiki estimasi berlebihan Q-Values. 
            * ***Prioritized Experience Replay (PER)*:** Mengambil sampel pengalaman penting lebih sering dari *replay buffer*. 
            * ***Dueling DQN*:** Model mengestimasi nilai keadaan dan keuntungan setiap tindakan secara terpisah. 
    * **Pustaka TF-Agents:** Pustaka RL berbasis TensorFlow dari Google untuk membangun sistem RL yang kuat. 
        * **Lingkungan TF-Agents:** Mirip dengan OpenAI Gym. 
        * **Spesifikasi Lingkungan:** Menyediakan spesifikasi observasi, tindakan, dan langkah waktu. 
        * ***Environment Wrappers* dan Pra-pemrosesan Atari:** Pra-pemrosesan umum untuk lingkungan Atari. 
        * **Arsitektur Pelatihan:** Driver dan agen bekerja paralel untuk mengumpulkan dan melatih *trajectory*. 
        * **Membuat *Deep Q-Network*:** Menggunakan `tf_agents.networks.q_network.QNetwork`. 
        * **Membuat Agen DQN:** Menggunakan `tf_agents.agents.dqn.dqn_agent.DqnAgent`. 
        * **Membuat *Replay Buffer* dan *Observer* yang Sesuai:** Menyimpan pengalaman dalam *replay buffer*. 
        * **Membuat Metrik Pelatihan:** Menghitung metrik seperti jumlah episode, langkah, dan hadiah rata-rata. 
        * **Membuat Driver Koleksi:** Objek yang mengeksplorasi lingkungan dan mengumpulkan pengalaman. 
        * **Membuat *Dataset*:** Menggunakan `replay_buffer.as_dataset()` untuk efisiensi. 
        * **Membuat Loop Pelatihan:** Mengubah fungsi utama menjadi *TF Functions* untuk kecepatan. 
    * **Tinjauan Beberapa Algoritma RL Populer:** *Actor-Critic algorithms*, *Asynchronous Advantage Actor-Critic (A3C)*, *Advantage Actor-Critic (A2C)*, *Soft Actor-Critic (SAC)*, *Proximal Policy Optimization (PPO)*, *Curiosity-based exploration*. 

10. **Melatih dan Menyebarkan Model TensorFlow pada Skala Besar (Bab 19)**
    * **Melayani Model TensorFlow (*Serving a TensorFlow Model*):** Membungkus model dalam layanan web untuk *deployment*. 
        * **Mengekspor *SavedModels*:** Menyimpan model ke format *SavedModel* TensorFlow. 
        * **Menginstal *TensorFlow Serving*:** Menggunakan Docker untuk menjalankan server model C++ yang efisien. 
        * **Membuat Kueri *TF Serving* melalui API REST:** Mengirim permintaan HTTP POST. 
        * **Membuat Kueri *TF Serving* melalui API gRPC:** Menggunakan *protocol buffer* yang lebih efisien untuk data besar. 
        * **Menyebarkan Versi Model Baru:** *TF Serving* secara otomatis menangani transisi ke versi model terbaru. 
        * **Menskalakan *TF Serving*:** Menyebarkan *TF Serving* di banyak server dengan *load balancing*. 
    * **Membuat Layanan Prediksi di Google Cloud AI Platform (GCP AI Platform):** 
        * **Pengaturan Awal:** Login ke GCP, buat proyek, atur *billing*, buat *bucket* GCS, unggah model. 
        * **Konfigurasi Model di AI Platform:** Membuat model dan versi model. 
    * **Menggunakan Layanan Prediksi:** Membuat *service account* untuk otentikasi dan mengkueri layanan prediksi. 
    * **Menyebarkan Model ke Perangkat Seluler atau Tersemat (*Embedded Device*):** Menggunakan pustaka *TFLite* untuk membuat model yang ringan dan efisien. 
        * **Mengurangi Ukuran Model:** Menggunakan format *FlatBuffers* dan mengoptimalkan komputasi. 
        * **Kuantisasi *Post-Training*:** Kuantisasi bobot ke *integer* 8-bit. 
        * **Pelatihan yang Sadar Kuantisasi (*Quantization-Aware Training*):** Menambahkan operasi kuantisasi palsu selama pelatihan. 
    * **TensorFlow di *Browser*:** Menggunakan *TensorFlow.js* untuk menjalankan model langsung di *web browser*. 
    * **Menggunakan GPU untuk Mempercepat Komputasi:** 
        * **Mendapatkan GPU Sendiri:** Memilih kartu GPU Nvidia dan menginstal *driver* serta pustaka yang diperlukan (CUDA, cuDNN). 
        * **Menggunakan *Virtual Machine* yang Dilengkapi GPU:** Menyewa VM dengan GPU di layanan *cloud* (misal: GCP AI Platform). 
        * ***Colaboratory* (Colab):** Cara termudah dan termurah untuk mengakses VM GPU secara gratis. 
        * **Mengelola RAM GPU:** Mengalokasikan RAM GPU secara efisien untuk beberapa proses. 
        * **Menempatkan Operasi dan Variabel pada Perangkat:** Menentukan perangkat (CPU/GPU) tempat operasi dan variabel dijalankan. 
        * **Eksekusi Paralel Lintas Beberapa Perangkat:** Bagaimana TensorFlow menjalankan operasi secara paralel. 
    * **Melatih Model pada Skala Menggunakan API Strategi Distribusi:** 
        * **Paralelisme Model (*Model Parallelism*):** Membagi model menjadi beberapa bagian dan menjalankan setiap bagian pada perangkat yang berbeda. 
        * **Paralelisme Data (*Data Parallelism*):** Mereplikasi model di setiap perangkat dan menjalankan setiap langkah pelatihan secara bersamaan pada semua replika. 
            * **Strategi Cermin (*Mirrored Strategy*):** Semua parameter model dicerminkan di semua GPU. 
            * **Parameter Terpusat:** Parameter model disimpan di luar perangkat GPU. 
            * ***Synchronous* vs. *Asynchronous Updates*:** Sinkron (menunggu semua gradien) atau asinkron (pembaruan segera). 
            * ***Bandwidth Saturation*:** Keterbatasan dalam mentransfer data. 
        * **Melatih pada Klaster TensorFlow:** Menggunakan `tf.distribute.experimental.MultiWorkerMirroredStrategy` atau `tf.distribute.experimental.CentralStorageStrategy` (untuk ParameterServerStrategy) untuk pelatihan terdistribusi. 
        * **Menjalankan Tugas Pelatihan Besar di Google Cloud AI Platform:** Menggunakan AI Platform untuk menyediakan dan mengelola VM GPU. 
        * ***Black Box Hyperparameter Tuning* di AI Platform:** Menggunakan layanan optimasi Bayesian (Google Vizier) untuk menyempurnakan *hyperparameter*. 

## Kesimpulan

Pembelajaran ini telah mencakup berbagai konsep fundamental dan teknik implementasi Machine Learning, mulai dari dasar-dasar seperti jenis pembelajaran dan persiapan data, hingga topik-topik canggih dalam *Deep Learning* seperti arsitektur jaringan saraf yang kompleks (CNN, RNN, Transformer), *autoencoder*, GAN, dan *Reinforcement Learning*. Penekanan pada implementasi praktis menggunakan Scikit-Learn, Keras, dan TensorFlow, serta pemahaman tentang cara menangani data besar dan *deployment* model pada skala besar, memberikan fondasi yang kuat bagi siapa pun yang ingin menjadi praktisi Machine Learning yang handal.

**Poin-poin Penting yang Dipelajari:**
* **Fondasi ML:** Memahami berbagai jenis pembelajaran, metrik evaluasi, dan tantangan umum.
* **Pipeline ML:** Menguasai alur kerja proyek ML dari awal hingga *deployment*.
* **Jaringan Saraf:** Membangun, melatih, dan menyempurnakan MLP, CNN, dan RNN dengan Keras dan TensorFlow.
* **Optimasi & Regularisasi:** Mengatasi masalah gradien, *overfitting*, dan mempercepat pelatihan menggunakan teknik-teknik canggih.
* **Pemrosesan Data Skala Besar:** Menggunakan `tf.data`, TFRecord, dan lapisan pra-pemrosesan Keras untuk *pipeline* data yang efisien.
* **Pembelajaran Tanpa Pengawasan:** Memanfaatkan *clustering*, *autoencoder*, dan GAN untuk analisis data, reduksi dimensi, dan generasi data.
* **Pembelajaran Penguatan:** Membangun agen yang dapat belajar melalui interaksi dengan lingkungan.
* **Penyebaran Model:** Menggunakan *TF Serving* dan layanan *cloud* untuk *deployment* model yang skalabel.
* **Komputasi Paralel:** Memanfaatkan GPU dan strategi distribusi untuk pelatihan model yang lebih cepat.

Melalui eksplorasi buku ini, diharapkan pembaca tidak hanya memahami teori tetapi juga mampu mengaplikasikan pengetahuan tersebut untuk membangun sistem cerdas yang efektif dan efisien.
