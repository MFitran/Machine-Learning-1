# Rangkuman Bab 8: Reduksi Dimensi

Bab 8 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" membahas konsep dan teknik **reduksi dimensi (dimensionality reduction)**, yaitu proses mengurangi jumlah fitur (dimensi) dalam sebuah dataset sambil berusaha mempertahankan informasi yang paling relevan.

Bab ini dimulai dengan menjelaskan masalah **"kutukan dimensi" (curse of dimensionality)**, di mana dataset dengan jumlah fitur yang sangat banyak dapat menyebabkan pelatihan model menjadi sangat lambat, sulit menemukan solusi yang baik karena data menjadi sangat jarang (sparse), dan meningkatkan risiko *overfitting*. Disebutkan bahwa untuk mencapai kepadatan data yang cukup di ruang berdimensi tinggi, jumlah instance pelatihan yang dibutuhkan akan tumbuh secara eksponensial.

Kemudian, bab ini memperkenalkan dua pendekatan utama untuk reduksi dimensi:
1.  **Proyeksi (Projection)**: Pendekatan ini mengidentifikasi subruang berdimensi rendah (misalnya, sebuah bidang di ruang 3D) tempat data berada (atau sangat dekat dengannya), lalu memproyeksikan data ke subruang tersebut. Ini efektif ketika instance pelatihan tidak tersebar secara seragam di semua dimensi dan fitur-fitur berkorelasi tinggi.
2.  **Manifold Learning**: Pendekatan ini digunakan ketika data terletak pada "manifold" berdimensi rendah yang melengkung atau berbelit-belit di ruang berdimensi tinggi (seperti "Swiss roll"). Manifold Learning bertujuan untuk "membuka" atau memodelkan struktur berdimensi rendah yang tersembunyi dalam data.

Algoritma reduksi dimensi yang paling populer adalah **Principal Component Analysis (PCA)**. PCA bekerja dengan mengidentifikasi sumbu (komponen utama) yang menjelaskan variansi terbesar dalam data, kemudian memproyeksikan data ke sumbu-sumbu tersebut. Bab ini menjelaskan cara kerja PCA menggunakan **Singular Value Decomposition (SVD)** dan bagaimana memilih jumlah dimensi yang tepat berdasarkan **rasio variansi yang dijelaskan (explained variance ratio)**. Selain PCA standar, dibahas juga varian-varian seperti **Randomized PCA** (untuk dataset besar yang muat di memori) dan **Incremental PCA** (untuk dataset yang tidak muat di memori atau pelatihan online). Untuk data nonlinier, diperkenalkan **Kernel PCA (kPCA)**, yang menggunakan "trik kernel" untuk melakukan proyeksi nonlinier.

Selain PCA, bab ini juga mengulas **Locally Linear Embedding (LLE)** sebagai teknik *Manifold Learning* nonlinier yang tidak bergantung pada proyeksi dan sangat baik untuk membuka manifold yang berbelit-belit. Beberapa teknik reduksi dimensi lainnya yang disebutkan secara singkat meliputi Random Projections, Multidimensional Scaling (MDS), Isomap, dan t-Distributed Stochastic Neighbor Embedding (t-SNE).

Secara keseluruhan, reduksi dimensi adalah alat yang ampuh untuk mengatasi tantangan "kutukan dimensi", mempercepat pelatihan model, dan memfasilitasi visualisasi data berdimensi tinggi, meskipun ada potensi kehilangan informasi.
