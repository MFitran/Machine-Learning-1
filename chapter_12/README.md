# Rangkuman Bab 12: Model Kustom dan Pelatihan dengan TensorFlow

Bab 12 buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" menyelami lebih dalam API tingkat rendah TensorFlow untuk memungkinkan kustomisasi model dan proses pelatihan yang lebih mendalam. Meskipun `tf.keras` adalah API tingkat tinggi yang banyak digunakan dan cukup kuat untuk sebagian besar kasus, bab ini menunjukkan kapan dan bagaimana menggunakan fungsionalitas tingkat rendah TensorFlow untuk mendapatkan kontrol yang lebih presisi.

**Secara umum, bab ini mencakup poin-poin utama berikut:**

## 1. Tur Singkat TensorFlow 
TensorFlow adalah pustaka komputasi numerik yang kuat, dioptimalkan untuk Machine Learning skala besar. Fitur-fitur utamanya meliputi:
* **Inti mirip NumPy dengan dukungan GPU:** Memungkinkan komputasi array multidimensi yang efisien.
* **Komputasi terdistribusi:** Mampu menyebarkan komputasi ke banyak perangkat dan server.
* **Kompiler Just-In-Time (JIT):** Mengoptimalkan komputasi untuk kecepatan dan penggunaan memori dengan mengekstraksi *computation graph* dari fungsi Python.
* ***Computation Graphs* yang portabel:** *Graph* yang dihasilkan dapat diekspor dan dijalankan di berbagai lingkungan.
* ***Autodiff* dan *Optimizers*:** Mengimplementasikan diferensiasi otomatis (autodiff) dan menyediakan *optimizer* canggih seperti RMSProp dan Nadam.
Bab ini juga menyoroti ekosistem TensorFlow yang luas, termasuk TensorBoard untuk visualisasi, TensorFlow Extended (TFX) untuk produksi, TensorFlow Hub untuk model *pretrained*, TensorFlow Lite untuk perangkat seluler, dan TensorFlow.js untuk *browser* web.

## 2. Menggunakan TensorFlow seperti NumPy 
TensorFlow API berpusat pada *tensor*, yang sangat mirip dengan `NumPy ndarray`. Bab ini menjelaskan cara membuat dan memanipulasi *tensor*, melakukan operasi seperti pengindeksan, penjumlahan, perkalian, dan perkalian matriks. Penting untuk dicatat bahwa *tensor* TensorFlow bersifat *immutable* (tidak dapat diubah), berbeda dengan array NumPy yang *mutable*. Untuk nilai yang dapat diubah (seperti bobot model), `tf.Variable` digunakan. TensorFlow juga ketat dalam konversi tipe otomatis untuk menghindari masalah performa dan *error*.

## 3. Mengkustomisasi Model dan Algoritma Pelatihan 
Bagian ini adalah inti dari bab ini, menjelaskan bagaimana fleksibilitas TensorFlow memungkinkan kustomisasi yang mendalam:
* **Fungsi *Loss* Kustom:** Anda dapat mendefinisikan fungsi *loss* Anda sendiri (misalnya, *Huber loss*) sebagai fungsi Python sederhana atau dengan membuat *subclass* dari `tf.keras.losses.Loss` jika perlu menyimpan *hyperparameter*.
* **Menyimpan dan Memuat Model dengan Komponen Kustom:** Menjelaskan cara memastikan bahwa fungsi *loss* atau komponen kustom lainnya dapat disimpan dan dimuat bersama model.
* **Fungsi Aktivasi Kustom, Inisialisasi, Regularizer, dan Batasan:** Fitur-fitur ini juga dapat dikustomisasi sebagai fungsi Python biasa atau sebagai *subclass* dari kelas Keras yang sesuai (`tf.keras.regularizers.Regularizer`, `tf.keras.constraints.Constraint`, dll.).
* **Metrik Kustom:** Mirip dengan fungsi *loss*, metrik dapat didefinisikan sebagai fungsi atau sebagai *subclass* dari `tf.keras.metrics.Metric`, terutama untuk metrik *streaming* yang perlu melacak statusnya di banyak *batch* (misalnya, presisi dan *recall*).
* **Lapisan Kustom:** Untuk arsitektur yang tidak biasa atau blok lapisan yang dapat digunakan kembali, Anda dapat membuat lapisan kustom. Lapisan tanpa bobot dapat dibuat dengan `tf.keras.layers.Lambda`, sedangkan lapisan *stateful* (dengan bobot) memerlukan *subclass* dari `tf.keras.layers.Layer`.
* **Model Kustom:** Mirip dengan lapisan kustom, Anda dapat membuat model kustom dengan membuat *subclass* dari `tf.keras.Model`. Ini sangat berguna untuk arsitektur non-sequential yang kompleks seperti *Wide & Deep* atau model dengan *skip connections* dan *loop*.
* ***Loss* dan Metrik Berbasis Internal Model:** Menunjukkan cara mendefinisikan *loss* atau metrik yang bergantung pada bagian internal model, seperti bobot atau aktivasi lapisan tersembunyi, menggunakan metode `add_loss()` dan `add_metric()`.
* **Menghitung Gradien Menggunakan *Autodiff*:** Menjelaskan penggunaan `tf.GradientTape()` untuk secara otomatis menghitung gradien dari suatu fungsi terhadap *variabel*-nya, yang penting untuk *backpropagation*.
* **Loop Pelatihan Kustom:** Dalam kasus yang jarang terjadi di mana metode `fit()` Keras tidak cukup fleksibel (misalnya, menggunakan *optimizer* yang berbeda untuk bagian model yang berbeda), Anda dapat menulis *loop* pelatihan kustom Anda sendiri.

## 4. Fungsi dan *Graph* TensorFlow 
TensorFlow dapat mengoptimalkan fungsi Python dengan mengubahnya menjadi *TensorFlow Functions* (*TF Functions*). Ini mempercepat eksekusi secara signifikan. Bab ini menjelaskan konsep *concrete functions* (versi *TF Function* yang dioptimalkan untuk *signature input* tertentu) dan bagaimana TensorFlow menggunakan *AutoGraph* dan *tracing* untuk membangun *computation graph* dari kode Python. Penting untuk memahami aturan *TF Function*, seperti penggunaan operasi TensorFlow asli alih-alih fungsi Python eksternal, untuk memastikan *graph* yang efisien dan portabel.
