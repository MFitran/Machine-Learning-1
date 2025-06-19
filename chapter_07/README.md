Tentu, berikut adalah rangkuman Bab 7 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" dalam format Markdown:

---

# Rangkuman Bab 7: Ensemble Learning and Random Forests

Bab 7 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" membahas *Ensemble Learning*, sebuah teknik Machine Learning yang menggabungkan prediksi dari beberapa prediktor (model) individu untuk mencapai kinerja yang lebih baik daripada prediktor terbaik sekalipun. Konsep utamanya adalah "kebijaksanaan keramaian" (*wisdom of the crowd*), di mana agregasi jawaban dari banyak individu seringkali lebih akurat daripada jawaban seorang ahli tunggal.

Bab ini secara umum menjelaskan beberapa metode *Ensemble* yang populer:

## 1. Voting Classifiers
* **Konsep**: Menggabungkan prediksi dari beberapa klasifikasi yang berbeda.
* **Hard Voting (Majority Voting)**: Memilih kelas yang paling sering diprediksi oleh klasifikasi individu.
* **Soft Voting**: Menghitung probabilitas kelas rata-rata dari klasifikasi yang dapat memprediksi probabilitas, dan memilih kelas dengan probabilitas tertinggi. Metode ini seringkali memiliki kinerja lebih tinggi.
* **Prinsip Kerja Terbaik**: *Ensemble* berfungsi paling baik ketika prediktor seindependen mungkin (membuat kesalahan yang tidak berkorelasi).

## 2. Bagging and Pasting
* **Konsep**: Melatih beberapa prediktor menggunakan algoritma pelatihan yang sama pada subset acak yang berbeda dari dataset pelatihan.
* **Bagging (Bootstrap Aggregating)**: Pengambilan sampel dilakukan dengan penggantian (*with replacement*). Ini memperkenalkan lebih banyak keragaman dan cenderung mengurangi varian *ensemble*.
* **Pasting**: Pengambilan sampel dilakukan tanpa penggantian (*without replacement*).
* **Keuntungan**: Proses pelatihan dapat diparalelkan, sehingga sangat skalabel.
* **Evaluasi Out-of-Bag (OOB)**: Dalam *bagging*, sekitar 37% dari instansi pelatihan tidak diambil untuk setiap prediktor dan dapat digunakan untuk evaluasi model tanpa set validasi terpisah.
* **Random Patches & Random Subspaces**: Metode yang melibatkan pengambilan sampel fitur juga, yang meningkatkan keragaman prediktor.

## 3. Random Forests
* **Konsep**: Sebuah *ensemble* dari *Decision Tree*, umumnya dilatih menggunakan metode *bagging*.
* **Keacakan Ekstra**: Algoritma ini memperkenalkan keacakan tambahan dengan mencari fitur terbaik di antara subset fitur acak saat membagi *node*, menghasilkan keragaman *tree* yang lebih besar.
* **Implementasi**: Scikit-Learn menyediakan kelas `RandomForestClassifier` yang dioptimalkan.

## 4. Extra-Trees
* **Konsep**: Mirip dengan Random Forests, tetapi lebih acak lagi. Selain memilih subset fitur acak, mereka juga menggunakan ambang acak untuk setiap fitur (bukan mencari ambang terbaik).
* **Keuntungan**: Mengorbankan sedikit bias untuk varian yang lebih rendah, dan cenderung lebih cepat dilatih dibandingkan Random Forests.

## 5. Feature Importance
* **Konsep**: Random Forests dapat mengukur kepentingan relatif setiap fitur dengan melihat seberapa banyak *node tree* yang menggunakan fitur tersebut mengurangi *impurity* rata-rata di seluruh *forest*. Hasilnya tersedia melalui atribut `feature_importances_`.

## 6. Boosting
* **Konsep**: Metode *Ensemble* yang secara berurutan menambahkan prediktor ke dalam *ensemble*, di mana setiap prediktor mencoba mengoreksi kesalahan pendahulunya.
* **AdaBoost (Adaptive Boosting)**: Memberi bobot lebih pada instansi pelatihan yang salah diklasifikasikan oleh prediktor sebelumnya, sehingga prediktor berikutnya lebih fokus pada kasus-kasus sulit.
* **Gradient Boosting**: Menyesuaikan prediktor baru dengan *residual errors* yang dibuat oleh prediktor sebelumnya.
    * **Early Stopping**: Digunakan untuk menemukan jumlah *tree* optimal dan mencegah *overfitting*.

## 7. Stacking (Stacked Generalization)
* **Konsep**: Melatih sebuah model (*blender* atau *meta learner*) untuk menggabungkan prediksi dari prediktor-prediktor individu, alih-alih menggunakan fungsi agregasi trivial.
* **Proses Pelatihan**: Melibatkan pelatihan model dasar pada satu subset data, lalu menggunakan prediksi mereka pada subset data kedua untuk melatih *blender*. Ini dapat diperluas ke beberapa lapisan *ensemble*.

Secara keseluruhan, Bab 7 menunjukkan bahwa dengan menggabungkan kekuatan beberapa model, *Ensemble Learning* dapat menghasilkan prediktor yang lebih kuat dan lebih akurat untuk berbagai masalah Machine Learning.

---
