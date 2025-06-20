# Bab 15: Memproses Urutan Menggunakan RNN dan CNN

Bab 15 dari buku *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* mengulas secara mendalam Jaringan Neural Berulang (Recurrent Neural Networks - RNNs) dan bagaimana Convolutional Neural Networks (CNNs) satu dimensi (1D) dapat digunakan untuk memproses data sekuensial. Bab ini menjelaskan konsep-konsep fundamental, tantangan dalam pelatihan, dan implementasi praktis menggunakan TensorFlow dan Keras.

Secara umum, bab ini terbagi menjadi beberapa bagian utama:

## Neuron dan Lapisan Berulang

* **Pengenalan RNN**: Berbeda dengan jaringan *feedforward* tradisional, RNN memiliki koneksi yang menunjuk ke belakang, memungkinkan informasi dari langkah waktu sebelumnya mempengaruhi langkah waktu saat ini, sehingga mereka memiliki semacam "memori".
* **Arsitektur Dasar**: Neuron berulang menerima input saat ini ($x^{(t)}$) dan outputnya sendiri dari langkah waktu sebelumnya ($y^{(t-1)}$). Lapisan neuron berulang menggabungkan input saat ini dengan output tersembunyi dari langkah waktu sebelumnya untuk menghasilkan output baru.
* **Sel Memori**: Bagian dari jaringan neural yang mempertahankan suatu keadaan (state) melintasi langkah waktu disebut sel memori. Neuron berulang sederhana adalah sel dasar yang hanya mampu mempelajari pola-pola pendek.

## Input dan Output Urutan

RNNs sangat fleksibel dan dapat dikategorikan berdasarkan format input dan output urutannya:
* **Urutan-ke-Urutan (Sequence-to-Sequence)**: Menerima urutan input dan menghasilkan urutan output (misalnya, peramalan deret waktu).
* **Urutan-ke-Vektor (Sequence-to-Vector)**: Menerima urutan input dan menghasilkan satu output vektor tunggal (misalnya, analisis sentimen).
* **Vektor-ke-Urutan (Vector-to-Sequence)**: Menerima satu input vektor tunggal dan menghasilkan urutan output (misalnya, *image captioning*).
* **Encoder-Decoder**: Kombinasi urutan-ke-vektor (encoder) dan vektor-ke-urutan (decoder), sering digunakan untuk terjemahan mesin neural.

## Pelatihan RNNs

* **Backpropagation Through Time (BPTT)**: RNN dilatih dengan "membuka gulungan" (unrolling) jaringan sepanjang waktu, lalu menerapkan algoritma *backpropagation* reguler. Gradien dihitung dan disebarkan mundur melalui jaringan untuk memperbarui bobot model.

## Tantangan dalam Menangani Urutan Panjang

Melatih RNN pada urutan panjang menimbulkan dua masalah utama:
* **Gradien Tidak Stabil**: Masalah *vanishing gradients* (gradien menjadi sangat kecil, mencegah lapisan bawah belajar) dan *exploding gradients* (gradien menjadi sangat besar, menyebabkan divergensi).
    * **Solusi**: Teknik seperti laju pembelajaran yang lebih kecil, fungsi aktivasi yang saturasi (`tanh`), *gradient clipping*, dan *Layer Normalization* digunakan untuk mengatasi masalah ini. *Layer Normalization* bekerja lebih baik di RNN daripada *Batch Normalization* karena menormalisasi sepanjang dimensi fitur dan menghitung statistik secara *on-the-fly*.
* **Memori Jangka Pendek Terbatas**: Informasi dapat hilang seiring data melewati RNN melalui banyak langkah waktu, sehingga state RNN kehilangan jejak input awal.
    * **Solusi**: Menggunakan sel memori yang lebih kompleks dan kuat:
        * **Sel Long Short-Term Memory (LSTM)**: Sel LSTM memiliki gerbang lupa, gerbang input, dan gerbang output yang memungkinkan jaringan untuk secara selektif menyimpan, membuang, dan membaca informasi dari state jangka panjangnya. Ini membuatnya sangat efektif dalam menangkap pola jangka panjang.
        * **Sel Gated Recurrent Unit (GRU)**: Versi sel LSTM yang disederhanakan, seringkali berkinerja sama baiknya dengan lebih sedikit parameter.

## Memproses Urutan dengan CNNs 1D

Selain RNNs, lapisan konvolusi 1D juga dapat digunakan untuk memproses urutan:
* **Cara Kerja**: Lapisan `Conv1D` menggeser filter (kernel) di sepanjang urutan, mendeteksi pola sekuensial pendek. Ini dapat mempersingkat urutan input, membantu lapisan GRU/LSTM mendeteksi pola yang lebih panjang.
* **WaveNet**: Arsitektur canggih yang menumpuk lapisan `Conv1D` dengan tingkat dilatasi yang berlipat ganda di setiap lapisan. Ini memungkinkan *receptive field* yang sangat besar dan efisien untuk memproses urutan yang sangat panjang (misalnya, puluhan ribu langkah waktu dalam audio). `padding="causal"` digunakan untuk memastikan prediksi hanya berdasarkan data masa lalu.
