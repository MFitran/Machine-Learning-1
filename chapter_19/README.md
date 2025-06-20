# Bab 19, "Training and Deploying TensorFlow Models at Scale"
Bab ini membahas langkah-langkah krusial setelah model *Machine Learning* selesai dilatih, yaitu bagaimana menerapkan dan mengelola model tersebut dalam lingkungan produksi skala besar. Bab ini dibagi menjadi beberapa bagian utama:

## **1. Menyajikan Model TensorFlow (Serving a TensorFlow Model)**
Bagian ini menjelaskan pentingnya membungkus model yang telah dilatih dalam sebuah layanan khusus, bukan hanya mengintegrasikannya langsung ke dalam aplikasi. Pendekatan ini memungkinkan pembaruan model yang lancar, penskalaan layanan secara independen, dan eksperimen A/B. TF Serving, sebuah *model server* yang efisien dan tangguh yang ditulis dalam C++, diperkenalkan sebagai solusi utama. Dijelaskan pula cara mengekspor model ke format SavedModel TensorFlow, yang merupakan format yang disukai untuk penerapan model karena menyimpan grafik komputasi dan bobot model. Proses *query* TF Serving melalui REST API dan gRPC API juga dijelaskan, dengan gRPC direkomendasikan untuk efisiensi yang lebih tinggi dalam transfer data besar. Konsep penerapan versi model baru dan transisi yang mulus antara versi juga dibahas.

## **2. Menerapkan Model ke Google Cloud AI Platform**
Bagian ini memperluas konsep penerapan model ke lingkungan *cloud* menggunakan Google Cloud AI Platform (sebelumnya Google Cloud ML Engine). Dijelaskan langkah-langkah untuk menyiapkan akun GCP, membuat bucket Google Cloud Storage (GCS) untuk menyimpan model, dan mengonfigurasi model dan versi di AI Platform. Keuntungan menggunakan platform *cloud* seperti penskalaan otomatis dan pengelolaan infrastruktur dibahas, yang dapat mengurangi beban operasional secara signifikan.

## **3. Menerapkan Model ke Perangkat Seluler atau Embedded**
Bab ini juga menyentuh penerapan model pada perangkat dengan sumber daya terbatas seperti ponsel atau perangkat *embedded*. TFLite diperkenalkan sebagai pustaka yang dirancang untuk tujuan ini, membantu mengurangi ukuran model dan komputasi yang diperlukan. Teknik kuantisasi (mengubah bobot model menjadi bilangan bulat 8-bit) dijelaskan sebagai cara untuk lebih lanjut mengurangi ukuran model, baik melalui kuantisasi pasca-pelatihan maupun pelatihan yang sadar kuantisasi.

## **4. Menggunakan GPU untuk Mempercepat Komputasi**
Melatih model *Deep Learning* yang besar bisa sangat memakan waktu. Bagian ini menyoroti peran GPU dalam mempercepat komputasi pelatihan, mengubah waktu pelatihan dari hari menjadi menit atau jam. Dibahas cara mengelola RAM GPU dan menempatkan operasi serta variabel secara strategis pada perangkat yang berbeda (CPU atau GPU) untuk memaksimalkan efisiensi komputasi paralel.

## **5. Melatih Model dalam Skala Besar Menggunakan Distribution Strategies API**
Untuk model yang lebih besar atau persyaratan kecepatan pelatihan yang lebih tinggi, bab ini memperkenalkan TensorFlow Distribution Strategies API. API ini menyederhanakan pelatihan model di beberapa GPU pada satu mesin atau bahkan di seluruh klaster server. Dua pendekatan utama dibahas: paralelisme data (mereplikasi model dan melatihnya dengan subset data yang berbeda) dan paralelisme model (membagi model ke seluruh perangkat). Paralelisme data umumnya direkomendasikan karena lebih sederhana untuk diimplementasikan dan lebih efisien dalam banyak kasus.

## **6. Menjalankan Pekerjaan Pelatihan Besar di Google Cloud AI Platform**
Terakhir, bab ini kembali membahas Google Cloud AI Platform sebagai solusi untuk menjalankan pekerjaan pelatihan skala besar. Dijelaskan cara mengirimkan pekerjaan pelatihan menggunakan alat baris perintah `gcloud`, memanfaatkan VM GPU yang disediakan *cloud*. Layanan penyetelan *hyperparameter* Google Vizier, yang menggunakan optimasi Bayesian, juga disorot sebagai alat untuk secara efisien menemukan kombinasi *hyperparameter* terbaik.
