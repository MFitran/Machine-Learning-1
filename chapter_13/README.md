# Bab 13: Memuat dan Melakukan Pra-pemrosesan Data dengan TensorFlow

Bab ini secara umum membahas pentingnya dan bagaimana cara memuat serta melakukan pra-pemrosesan data secara efisien untuk proyek Machine Learning dan Deep Learning menggunakan TensorFlow. Data API TensorFlow (`tf.data`) adalah alat utama yang diperkenalkan untuk tujuan ini.

**Poin-poin Utama:**

* **Pentingnya Pra-pemrosesan Data yang Efisien:**
    * Dataset yang besar seringkali tidak muat di dalam memori RAM, sehingga perlu dimuat secara *streaming* dari disk.
    * Pra-pemrosesan yang lambat dapat menyebabkan *bottleneck* pelatihan, membuat GPU/TPU kurang dimanfaatkan.
    * Mengintegrasikan langkah pra-pemrosesan ke dalam model atau pipeline memastikan konsistensi antara tahap pelatihan dan inferensi.

* **The Data API (`tf.data`):**
    * Berpusat pada konsep `tf.data.Dataset`, yang merepresentasikan urutan item data.
    * Memungkinkan **Chaining Transformations** (merangkai transformasi) seperti `repeat()`, `batch()`, `map()`, dan `filter()`.
    * Mendukung **Shuffling Data** (pengacakan data) menggunakan `shuffle()` untuk memastikan instance IID (Independen dan Terdistribusi Identik), yang penting untuk Gradient Descent yang optimal.
    * Memungkinkan **Interleaving Lines from Multiple Files** (membaca baris secara bergantian dari banyak file) untuk dataset besar yang terbagi dalam beberapa file, meningkatkan efisiensi pengacakan dan pemuatan.
    * Memiliki metode `prefetch(1)` untuk optimasi performa krusial, yang memastikan dataset selalu "satu batch di depan" agar GPU/TPU tidak menganggur menunggu data.
    * Dapat digunakan dengan mudah dengan model `tf.keras` untuk pelatihan dan inferensi.

* **The TFRecord Format:**
    * Format yang disarankan TensorFlow untuk menyimpan data dalam jumlah besar dan membacanya secara efisien.
    * Merupakan format biner sederhana yang berisi urutan *record* biner dengan berbagai ukuran.
    * Dapat dibuat dalam **Compressed TFRecord Files** (file terkompresi) seperti GZIP untuk menghemat ruang disk dan mengurangi waktu unduh.
    * Biasanya berisi **Protocol Buffers (Protobufs)** yang diserialkan, yaitu format biner yang portabel, dapat diperluas, dan efisien untuk merepresentasikan data terstruktur. Contohnya adalah `tf.train.Example` untuk instance data tunggal, atau `tf.train.SequenceExample` untuk data urutan (daftar daftar).

* **Preprocessing the Input Features:**
    * Fitur mentah (misalnya, CSV strings) perlu diuraikan dan dinormalisasi.
    * Fitur kategorikal perlu diubah menjadi representasi numerik menggunakan **One-Hot Vectors** (untuk kategori berjumlah sedikit) atau **Embeddings** (untuk kategori berjumlah banyak). Embeddings adalah vektor padat yang dapat dilatih dan merepresentasikan setiap kategori.
    * Keras menyediakan **Keras Preprocessing Layers** (lapisan pra-pemrosesan Keras standar) yang dapat diintegrasikan langsung ke dalam model untuk konsistensi antara pelatihan dan *serving*.
    * **TF Transform (`tf.Transform`)** adalah alat yang lebih canggih dari TensorFlow Extended (TFX) untuk pra-pemrosesan *offline*. Ini memungkinkan penulisan fungsi pra-pemrosesan tunggal yang dapat dijalankan secara *batch* pada *training set* dan kemudian diekspor ke `tf.function` untuk konsistensi saat *serving*.

* **The TensorFlow Datasets (TFDS) Project:**
    * Memudahkan pengunduhan dan pemuatan banyak dataset umum secara langsung ke dalam format `tf.data.Dataset`, termasuk dataset besar seperti ImageNet.

Secara keseluruhan, Bab 13 membekali pembaca dengan pengetahuan dan alat untuk membangun pipeline input yang skalabel dan efisien, memastikan data dipersiapkan dengan baik dan tersedia untuk model Machine Learning dan Deep Learning, baik selama pelatihan maupun inferensi.
