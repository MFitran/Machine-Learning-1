# Bab 4: Training Models

Bab ini membahas berbagai model Machine Learning dan algoritma pelatihannya, mulai dari model linear sederhana hingga model yang lebih kompleks, serta teknik-teknik untuk mengoptimalkannya. Tujuannya adalah memberikan pemahaman mendalam tentang cara kerja model dan cara melatihnya secara efektif.

**Secara umum, bab ini mencakup poin-poin utama berikut:**

## 1. Regresi Linear (Linear Regression) 
* **Definisi:** Model dasar yang memprediksi nilai target numerik berdasarkan hubungan linier dengan fitur input. 
* **Persamaan Normal (The Normal Equation):** Sebuah solusi *closed-form* (persamaan matematika langsung) untuk menghitung parameter model optimal yang meminimalkan *Mean Squared Error* (MSE). 
* **Kompleksitas Komputasi:** Menjelaskan efisiensi komputasi dari Persamaan Normal dan metode *Singular Value Decomposition* (SVD) yang digunakan Scikit-Learn, terutama bagaimana keduanya melambat dengan peningkatan jumlah fitur. 

## 2. Gradient Descent 
* **Konsep Umum:** Algoritma optimisasi iteratif yang secara bertahap menyesuaikan parameter model untuk meminimalkan fungsi biaya, seperti menemukan dasar lembah dengan menuruni lereng tercuram. 
* **Learning Rate:** Menyoroti pentingnya *learning rate* (ukuran langkah) yang tepat; terlalu kecil akan lambat, terlalu besar bisa membuat divergen. 
* **Batch Gradient Descent:** Menggunakan seluruh *training set* untuk menghitung gradien di setiap langkah, sehingga sangat akurat tetapi lambat pada dataset besar. 
* **Stochastic Gradient Descent (SGD):** Mengambil satu *instance* acak dari *training set* di setiap langkah, membuatnya sangat cepat dan cocok untuk dataset besar, meskipun gradiennya lebih "berisik". 
* **Mini-batch Gradient Descent:** Kompromi antara Batch GD dan SGD, menghitung gradien pada *mini-batch* (kelompok kecil *instance*), yang menyeimbangkan kecepatan dan stabilitas, serta memanfaatkan optimasi *hardware*. 

## 3. Regresi Polinomial (Polynomial Regression) 
* **Definisi:** Teknik untuk menyesuaikan model linear dengan data nonlinier dengan menambahkan pangkat fitur sebagai fitur baru. 
* **Kurva Pembelajaran (Learning Curves):** Plot yang menunjukkan kinerja model pada *training set* dan *validation set* seiring dengan peningkatan ukuran *training set*. Ini membantu mendiagnosis *underfitting* (model terlalu sederhana) atau *overfitting* (model terlalu kompleks). 
* **Bias/Variance Trade-off:** Menjelaskan bagaimana error generalisasi model dapat dibagi menjadi bias (asumsi model yang salah) dan varians (sensitivitas model terhadap variasi data training), serta bagaimana meningkatkan kompleksitas model memengaruhi keduanya. 

## 4. Model Linear yang Diregulasi (Regularized Linear Models) 
* **Tujuan:** Mengurangi *overfitting* dengan membatasi bobot model. 
* **Ridge Regression (Regresi Ridge):** Menambahkan istilah penalti ($\ell_2$ norm) pada fungsi biaya, yang memaksa bobot model tetap kecil. 
* **Lasso Regression (Regresi Lasso):** Menambahkan istilah penalti ($\ell_1$ norm) yang cenderung membuat bobot fitur yang tidak penting menjadi nol, sehingga secara otomatis melakukan pemilihan fitur (*feature selection*). 
* **Elastic Net:** Kombinasi dari Ridge dan Lasso, menawarkan fleksibilitas dalam menyeimbangkan efek keduanya. 
* **Penghentian Awal (Early Stopping):** Teknik regulasi yang efektif di mana training dihentikan segera setelah kinerja model pada *validation set* mulai menurun, mencegah *overfitting* lebih lanjut. 

## 5. Regresi Logistik (Logistic Regression) 
* **Tujuan:** Digunakan untuk tugas klasifikasi biner, memperkirakan probabilitas suatu *instance* termasuk dalam kelas tertentu. 
* **Fungsi Sigmoid:** Menggunakan fungsi logistik (sigmoid) untuk mengubah *weighted sum* fitur menjadi probabilitas antara 0 dan 1. 
* **Fungsi Biaya (Log Loss):** Menjelaskan fungsi biaya *cross-entropy* yang digunakan untuk melatih model ini, di mana tujuannya adalah memaksimalkan probabilitas prediksi yang benar. 
* **Batas Keputusan (Decision Boundaries):** Menunjukkan bagaimana model membuat keputusan klasifikasi berdasarkan probabilitas yang dihitung, seringkali dengan batas keputusan linier. 

## 6. Regresi Softmax (Softmax Regression) 
* **Tujuan:** Generalisasi dari Regresi Logistik untuk menangani klasifikasi multikelas (lebih dari dua kelas) secara langsung, tanpa perlu menggabungkan pengklasifikasi biner. 
* **Fungsi Softmax:** Mengubah skor mentah menjadi probabilitas untuk setiap kelas, memastikan bahwa semua probabilitas kelas berjumlah 1. 
* **Fungsi Biaya Cross-Entropy:** Digunakan untuk melatih model, penalizes model ketika memprediksi probabilitas rendah untuk kelas target yang benar. 

Bab ini memberikan dasar teoritis dan praktis yang kuat untuk memahami berbagai model regresi dan klasifikasi linear, serta algoritma optimisasi dan teknik regulasi yang penting dalam Machine Learning.
