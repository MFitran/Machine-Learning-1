
Tentu, berikut adalah rangkuman Bab 3 "Classification" dalam format Markdown:

# Rangkuman Bab 3: Klasifikasi

Bab 3 buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" memperkenalkan dan menjelaskan konsep fundamental klasifikasi dalam Machine Learning. Klasifikasi adalah salah satu tugas pembelajaran terawasi (supervised learning) yang paling umum, di mana tujuannya adalah untuk mengategorikan atau mengklasifikasikan input ke dalam salah satu dari beberapa kelas diskrit.

Bab ini dimulai dengan memperkenalkan **dataset MNIST**[cite: 85], sebuah kumpulan 70.000 gambar tulisan tangan digit (0-9), yang sering disebut sebagai "hello world" dalam Machine Learning untuk klasifikasi gambar. Dataset ini digunakan sebagai contoh praktis untuk melatih dan mengevaluasi model. Proses pemuatan dan persiapan data, termasuk pemisahan menjadi set pelatihan (training set) dan set pengujian (test set) serta pengubahan tipe data label dari string ke integer, dibahas secara detail[cite: 85, 86, 87].

Selanjutnya, bab ini membahas tentang **Binary Classifier**, yaitu model yang hanya membedakan antara dua kelas. Contoh yang digunakan adalah "detektor-5" yang mengklasifikasikan gambar sebagai "5" atau "bukan-5". `SGDClassifier` dari Scikit-Learn diperkenalkan sebagai contoh classifier biner yang efisien untuk dataset besar[cite: 88].

Bagian penting dari bab ini didedikasikan untuk **Pengukuran Kinerja (Performance Measures)** classifier, karena evaluasi classifier seringkali lebih rumit daripada regressor. Metrik yang dibahas meliputi:
* **Akurasi (Accuracy)**: Meskipun merupakan metrik yang paling intuitif, akurasi bisa menyesatkan, terutama pada dataset yang miring (skewed datasets), di mana satu kelas jauh lebih dominan daripada yang lain. Contoh `Never5Classifier` secara jelas menunjukkan mengapa akurasi saja tidak cukup[cite: 89, 90].
* **Confusion Matrix**: Metrik yang jauh lebih baik ini menyajikan hitungan true positives, false positives, true negatives, dan false negatives, memberikan gambaran yang lebih komprehensif tentang kinerja classifier[cite: 90, 91].
* **Presisi (Precision) dan Recall (Daya Ingat)**: Kedua metrik ini sangat penting dan didefinisikan sebagai $Precision = \frac{TP}{TP + FP}$ dan $Recall = \frac{TP}{TP + FN}$. F1 Score, sebagai *harmonic mean* dari presisi dan recall, juga diperkenalkan sebagai cara untuk menggabungkan kedua metrik ini[cite: 91, 92].
* **Precision/Recall Trade-off**: Dijelaskan bahwa seringkali ada pertukaran antara presisi dan recall. Meningkatkan yang satu akan menurunkan yang lain, dan sebaliknya. Fungsi `decision_function()` dari Scikit-Learn digunakan untuk menunjukkan bagaimana mengubah ambang batas keputusan memengaruhi trade-off ini, dan `precision_recall_curve()` untuk memvisualisasikannya[cite: 93, 94, 95, 96].
* **Kurva ROC (Receiver Operating Characteristic)**: Metrik populer lainnya, yang memplot True Positive Rate (Recall) terhadap False Positive Rate (FPR). Area Under the Curve (AUC) dari kurva ROC digunakan untuk membandingkan kinerja classifier secara keseluruhan. `RandomForestClassifier` juga diperkenalkan dan dibandingkan dengan `SGDClassifier` menggunakan kurva ROC dan skor AUC-nya, menunjukkan kinerja yang superior[cite: 97, 98, 99].

Bab ini juga membahas tentang **Multiclass Classification** (klasifikasi multikelas atau multinomial), di mana model dapat membedakan lebih dari dua kelas. Dua strategi utama diperkenalkan: One-versus-the-Rest (OvR) dan One-versus-One (OvO). Scikit-Learn secara otomatis mengimplementasikan strategi ini ketika algoritma klasifikasi biner digunakan untuk tugas multikelas[cite: 100, 101, 102]. Pentingnya penskalaan fitur untuk meningkatkan akurasi dalam klasifikasi multikelas juga ditekankan[cite: 102].

**Analisis Kesalahan (Error Analysis)** dibahas sebagai langkah penting untuk meningkatkan model. Visualisasi confusion matrix yang dinormalisasi (dengan diagonal yang diisi nol untuk menyoroti kesalahan) digunakan untuk mengidentifikasi jenis-jenis kesalahan umum yang dibuat oleh classifier, seperti kebingungan antara digit "3" dan "5"[cite: 102, 103, 104, 105].

Terakhir, bab ini membahas dua jenis klasifikasi yang lebih kompleks:
* **Multilabel Classification**: Di mana sebuah instance dapat memiliki banyak kelas secara bersamaan (misalnya, mendeteksi beberapa orang dalam satu gambar). Classifier multilabel mengeluarkan banyak tag biner. Contoh dengan `KNeighborsClassifier` untuk menentukan apakah digit besar dan/atau ganjil disajikan[cite: 106].
* **Multioutput Classification**: Generalisasi klasifikasi multilabel di mana setiap label dapat berupa multikelas (misalnya, sistem yang menghilangkan *noise* dari gambar, di mana setiap piksel adalah label multikelas yang mewakili intensitas piksel bersih)[cite: 107, 108].

Secara keseluruhan, Bab 3 memberikan pemahaman yang kuat tentang konsep-konsep inti klasifikasi, metrik evaluasinya, dan berbagai jenis tugas klasifikasi, mempersiapkan pembaca untuk menerapkan sistem klasifikasi yang lebih canggih.
