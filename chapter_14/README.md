# Rangkuman Bab 14: Deep Computer Vision Menggunakan Convolutional Neural Networks (CNNs)

Bab 14 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Edisi ke-2) oleh Aurélien Géron, membahas secara komprehensif Convolutional Neural Networks (CNNs) sebagai tulang punggung visi komputer modern. Bab ini bertujuan untuk memberikan pemahaman menyeluruh tentang arsitektur, cara kerja, dan aplikasinya dalam berbagai tugas visual.

Secara umum, bab ini mencakup poin-poin utama berikut:

## 1. Arsitektur Korteks Visual
Bab ini dimulai dengan menelusuri inspirasi biologis CNN dari penelitian David H. Hubel dan Torsten Wiesel tentang korteks visual otak. Dijelaskan bagaimana neuron-neuron di korteks visual memiliki *receptive field* lokal yang kecil dan bagaimana fitur-fitur tingkat tinggi dibangun dari kombinasi fitur-fitur tingkat rendah. Konsep ini membedakan CNN dari Jaringan Saraf Tiruan (DNN) yang terhubung sepenuhnya, yang akan membutuhkan parameter yang sangat banyak untuk tugas-tugas gambar besar.

## 2. Lapisan Konvolusional (Convolutional Layers)
Ini adalah blok bangunan inti dari CNN. Dijelaskan bahwa neuron-neuron dalam lapisan konvolusional hanya terhubung ke piksel-piksel dalam *receptive field* lokal mereka, memungkinkan jaringan untuk fokus pada fitur-fitur kecil tingkat rendah dan kemudian menggabungkannya menjadi fitur-fitur tingkat tinggi yang lebih besar. Konsep-konsep penting seperti:
* **Filter (Kernel Konvolusi)**: Bobot neuron direpresentasikan sebagai filter yang dapat mendeteksi pola tertentu dalam gambar.
* **Feature Map**: Output dari lapisan yang menggunakan filter yang sama, menyoroti area yang mengaktifkan filter tersebut.
* **Weight Sharing**: Semua neuron dalam *feature map* berbagi parameter yang sama, mengurangi jumlah parameter model secara drastis dan memungkinkan pengenalan pola di lokasi mana pun dalam gambar.
* **Padding (Zero Padding)**: Menambahkan nol di sekitar input untuk mempertahankan ukuran spasial lapisan.
* **Stride**: Mengontrol pergeseran *receptive field*, yang dapat mengurangi dimensi spasial gambar.
* **Kebutuhan Memori**: Lapisan konvolusional memerlukan RAM yang sangat besar, terutama selama pelatihan karena *backward pass* dari *backpropagation*.

## 3. Lapisan Pooling (Pooling Layers)
Lapisan pooling digunakan untuk *subsample* (mengecilkan) gambar input untuk mengurangi beban komputasi, penggunaan memori, dan jumlah parameter. Lapisan ini tidak memiliki bobot dan mengagregasi input menggunakan fungsi seperti maksimum (*max pooling*) atau rata-rata (*average pooling*). Lapisan *max pooling* memperkenalkan invariansi terhadap terjemahan kecil dan umumnya lebih disukai karena mempertahankan fitur terkuat.

## 4. Arsitektur CNN
Bab ini menjelaskan struktur umum CNN yang mencakup tumpukan lapisan konvolusional dan pooling, diikuti oleh jaringan *feedforward* reguler. Beberapa arsitektur CNN populer dibahas secara rinci, termasuk:
* **LeNet-5 (1998)**: Arsitektur klasik untuk pengenalan digit tulisan tangan.
* **AlexNet (2012)**: Jauh lebih besar dan lebih dalam, memperkenalkan *dropout* dan augmentasi data untuk mengurangi *overfitting*.
* **GoogLeNet (2014)**: Memperkenalkan *inception modules* untuk efisiensi parameter yang lebih baik.
* **VGGNet (2014)**: Arsitektur sederhana dan klasik dengan filter 3x3 berukuran kecil tetapi banyak.
* **ResNet (Residual Network, 2015)**: Menggunakan *skip connections* untuk memungkinkan pelatihan jaringan yang sangat dalam.
* **Xception (2016)**: Mengganti modul *inception* dengan lapisan *depthwise separable convolution*.
* **SENet (Squeeze-and-Excitation Network, 2017)**: Memperkenalkan *SE block* untuk merekalibrasi *feature map*.

## 5. Menggunakan Model yang Sudah Dilatih (Pretrained Models)
Bab ini menunjukkan pentingnya dan kemudahan penggunaan model CNN yang sudah dilatih dari `keras.applications` untuk berbagai tugas visi komputer. Metode ini tidak hanya mempercepat pelatihan tetapi juga membutuhkan data pelatihan yang lebih sedikit. Konsep *transfer learning* dijelaskan sebagai teknik untuk menggunakan kembali lapisan bawah dari model yang sudah dilatih pada tugas-tugas serupa.

## 6. Klasifikasi dan Lokalisasi (Classification and Localization)
Tugas melokalisasi objek dalam gambar dijelaskan sebagai tugas regresi untuk memprediksi *bounding box*. Metrik `Intersection over Union (IoU)` diperkenalkan sebagai cara untuk mengevaluasi kualitas *bounding box*.

## 7. Deteksi Objek (Object Detection)
Bab ini membahas deteksi objek, yaitu mengklasifikasikan dan melokalisasi beberapa objek dalam satu gambar. Pendekatan tradisional melibatkan *sliding window* dan *non-max suppression*. Kemudian, konsep *Fully Convolutional Networks (FCNs)* diperkenalkan sebagai metode yang lebih efisien karena dapat memproses gambar dengan ukuran berapa pun hanya sekali. Model *You Only Look Once (YOLO)* disajikan sebagai arsitektur deteksi objek yang sangat cepat dan akurat.

## 8. Segmentasi Semantik (Semantic Segmentation)
Terakhir, bab ini membahas segmentasi semantik, di mana setiap piksel dalam gambar diklasifikasikan berdasarkan kelas objek yang menjadi bagiannya. Tantangan utama adalah mempertahankan resolusi spasial yang hilang dalam CNN. Solusi yang dibahas mencakup penggunaan lapisan *transposed convolutional* untuk *upsampling* dan *skip connections* untuk memulihkan detail spasial dari lapisan yang lebih rendah.

Secara keseluruhan, Bab 14 adalah panduan komprehensif untuk memahami dan menerapkan CNN dalam visi komputer, mencakup teori dasar hingga arsitektur canggih dan aplikasi praktis.
