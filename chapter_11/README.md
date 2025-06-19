### Rangkuman Bab 11: Melatih Deep Neural Networks

Bab 11 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" membahas secara mendalam berbagai tantangan yang muncul saat melatih Jaringan Neural Dalam (DNN) dan menyediakan beragam teknik untuk mengatasinya. Tujuan utamanya adalah membekali pembaca dengan pemahaman dan alat praktis untuk membangun serta melatih model DNN yang lebih dalam dan lebih efektif.

**Tantangan Utama dalam Pelatihan DNN:**

1.  **Masalah Vanishing/Exploding Gradients**: Ini adalah salah satu masalah paling umum di DNN.
    * **Vanishing Gradients** terjadi ketika gradien menjadi sangat kecil saat mengalir mundur melalui lapisan-lapisan, menyebabkan bobot pada lapisan bawah hampir tidak berubah dan pelatihan tidak konvergen.
    * **Exploding Gradients** terjadi ketika gradien menjadi terlalu besar, menyebabkan pembaruan bobot yang ekstrem dan membuat algoritma divergen (tidak stabil).

**Teknik untuk Mengatasi Tantangan:**

1.  **Strategi Inisialisasi Bobot yang Lebih Baik**:
    * **Inisialisasi Glorot (Xavier Initialization)**: Mengatasi masalah gradien dengan memastikan varians output setiap lapisan sebanding dengan varians inputnya, direkomendasikan untuk fungsi aktivasi seperti sigmoid dan tanh.
    * **Inisialisasi He**: Mirip dengan Glorot tetapi disesuaikan untuk fungsi aktivasi ReLU dan variannya, seringkali memberikan kinerja yang lebih baik.

2.  **Fungsi Aktivasi Non-Saturasi**: Fungsi aktivasi sigmoid dan tanh cenderung "saturasi" (gradiennya mendekati nol) pada input besar, memperparah masalah vanishing gradients.
    * **ReLU (Rectified Linear Unit)**: Cepat dihitung dan tidak saturasi pada nilai positif, meskipun bisa mengalami "dying ReLUs".
    * **Leaky ReLU, PReLU, ELU, dan SELU**: Varian-varian ReLU yang mengatasi masalah "dying ReLUs" dan seringkali memberikan kinerja yang lebih baik. Khususnya, SELU dapat membuat jaringan "self-normalize" dalam kondisi tertentu, secara efektif mengatasi masalah gradien yang tidak stabil.

3.  **Batch Normalization (BN)**: Menambahkan lapisan normalisasi di setiap atau beberapa lapisan tersembunyi.
    * Ini menormalisasi input setiap mini-batch (menjadikan mean 0 dan standar deviasi 1), lalu menskalakan dan menggesernya menggunakan parameter yang dipelajari.
    * BN sangat mengurangi masalah gradien, memungkinkan *learning rates* yang lebih besar, dan bertindak sebagai regularizer.

4.  **Gradient Clipping**: Membatasi gradien agar tidak melebihi ambang batas tertentu selama *backpropagation*. Ini terutama berguna untuk mengatasi *exploding gradients*, sering diterapkan pada Recurrent Neural Networks (RNNs).

5.  **Penggunaan Model Terlatih (Pretrained Models)**:
    * **Transfer Learning**: Menggunakan kembali lapisan-lapisan bawah dari model yang sudah terlatih pada tugas serupa. Ini sangat mempercepat pelatihan dan membutuhkan lebih sedikit data. Lapisan yang digunakan kembali biasanya dibekukan pada awalnya, lalu di-*fine-tune*.
    * **Unsupervised Pretraining**: Melatih model tak berarah (misalnya autoencoder) pada data tidak berlabel yang melimpah, kemudian menggunakan kembali bagian dari model tersebut untuk tugas berarah dengan data berlabel terbatas.
    * **Pretraining on Auxiliary Task**: Melatih model pada tugas terkait yang memiliki banyak data berlabel yang mudah diperoleh, kemudian mentransfer pengetahuannya ke tugas utama.

6.  **Optimizer yang Lebih Cepat**: Mengganti *Stochastic Gradient Descent (SGD)* dasar dengan optimasi yang lebih canggih dapat secara drastis mempercepat pelatihan.
    * **Momentum Optimization**: Mempercepat SGD dengan menambahkan sebagian dari gradien sebelumnya ke pembaruan saat ini, membantu melewati lembah datar dan minimum lokal.
    * **Nesterov Accelerated Gradient (NAG)**: Varian Momentum yang lebih cepat, mengukur gradien sedikit di depan arah momentum.
    * **AdaGrad**: Menyesuaikan *learning rate* per parameter, menurunkannya lebih cepat untuk dimensi yang curam. Namun, dapat melambat terlalu cepat di DNN.
    * **RMSProp**: Mengatasi kelemahan AdaGrad dengan hanya mengakumulasi gradien terbaru, menjaga *learning rate* tetap adaptif.
    * **Adam (Adaptive Moment Estimation)** dan **Nadam**: Menggabungkan ide-ide momentum dan RMSProp, seringkali menjadi *optimizer* pilihan karena konvergensi cepat dan kebutuhan *tuning* yang minimal.

7.  **Penjadwalan *Learning Rate***: Mengubah *learning rate* selama pelatihan untuk mencapai konvergensi yang lebih cepat dan solusi yang lebih baik. Strategi umum meliputi *power scheduling*, *exponential scheduling*, *piecewise constant scheduling*, *performance scheduling*, dan *1cycle scheduling*.

8.  **Regularisasi untuk Menghindari Overfitting**: Selain *early stopping* dan *Batch Normalization*, teknik lain yang dibahas meliputi:
    * **L1 dan L2 Regularization**: Menambahkan *penalty term* pada bobot model untuk mendorong bobot kecil (L2) atau *sparse* (L1).
    * **Dropout**: Secara acak menonaktifkan neuron selama pelatihan untuk mencegah *co-adaptation* dan meningkatkan generalisasi.
    * **Monte Carlo (MC) Dropout**: Menggunakan model *dropout* selama inferensi untuk mendapatkan estimasi probabilitas yang lebih andal dan ketidakpastian.
    * **Max-Norm Regularization**: Membatasi norma bobot yang masuk ke setiap neuron.
