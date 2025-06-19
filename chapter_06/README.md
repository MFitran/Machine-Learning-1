# Bab 6: Decision Trees

Bab 6 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" membahas secara mendalam tentang **Decision Trees**, sebuah algoritma Machine Learning yang serbaguna dan intuitif. Decision Trees dapat digunakan untuk tugas klasifikasi maupun regresi, serta mampu menangani masalah multi-output. Algoritma ini merupakan komponen fundamental dari **Random Forests**, salah satu algoritma Machine Learning yang paling kuat saat ini.

## Konsep Utama yang Dibahas:

* **Training dan Visualisasi Decision Tree**: Bab ini menjelaskan cara melatih Decision Tree dan memvisualisasikan strukturnya. Salah satu keunggulan Decision Trees adalah kebutuhan minimal akan persiapan data, bahkan tidak memerlukan penskalaan atau pemusatan fitur.
* **Mekanisme Prediksi**: Dijelaskan bagaimana Decision Tree membuat prediksi dengan melintasi node dari *root* hingga *leaf*. Setiap *node* di Decision Tree berisi pertanyaan tentang fitur, dan *leaf node* berisi prediksi kelas. Decision Tree juga dapat memperkirakan probabilitas suatu *instance* termasuk dalam kelas tertentu.
* **Algoritma Pelatihan CART**: Scikit-Learn menggunakan algoritma CART (Classification and Regression Tree) untuk melatih Decision Tree. Algoritma ini bekerja secara rekursif dengan membagi *training set* menjadi dua subset yang paling murni berdasarkan fitur dan nilai ambang tertentu. Dijelaskan pula kompleksitas komputasi dari proses pelatihan dan prediksi.
* **Gini Impurity vs. Entropy**: Bab ini membandingkan dua metrik utama untuk mengukur ketidakmurnian node: Gini Impurity dan Entropy. Meskipun Gini Impurity sedikit lebih cepat dihitung dan merupakan *default* yang baik, Entropy cenderung menghasilkan pohon yang lebih seimbang.
* **Regularisasi Decision Tree**: Decision Trees cenderung *overfitting* data pelatihan jika tidak dibatasi karena mereka adalah model non-parametrik yang memiliki banyak *degree of freedom*. Untuk mengatasi *overfitting*, berbagai *hyperparameter* regularisasi dapat diatur, seperti `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, dan `max_features`.
* **Regresi dengan Decision Tree**: Decision Trees juga efektif untuk tugas regresi, di mana setiap *node* memprediksi nilai numerik (rata-rata nilai target dari *instance* dalam *node* tersebut) alih-alih kelas. Fungsi biaya CART untuk regresi bertujuan untuk meminimalkan *Mean Squared Error* (MSE).
* **Keterbatasan dan Ketidakstabilan**: Meskipun Decision Trees mudah dipahami dan kuat, mereka memiliki keterbatasan, termasuk sensitivitas terhadap rotasi data dan variasi kecil dalam *training data*. Ketidakstabilan ini dapat diatasi dengan teknik *ensemble* seperti Random Forests, yang akan dibahas di bab selanjutnya.

Secara keseluruhan, Bab 6 memberikan pemahaman fundamental tentang cara kerja Decision Trees, bagaimana melatih dan mengevaluasinya, serta cara mengatasi masalah umum seperti *overfitting* dan ketidakstabilan.
