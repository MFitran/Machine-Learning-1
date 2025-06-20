# Rangkuman Bab 17: Pembelajaran Representasi dan Pembelajaran Generatif Menggunakan Autoencoder dan GANs

Bab ini memperkenalkan dua arsitektur Jaringan Saraf Tiruan (ANN) yang powerful dalam pembelajaran tanpa pengawasan: **Autoencoder** dan **Generative Adversarial Networks (GANs)**. Keduanya berfokus pada pembelajaran representasi data yang efisien (disebut *latent representations* atau *codings*) dan kemampuan untuk menghasilkan data baru yang realistis.

## 1. Pembelajaran Representasi yang Efisien dengan Autoencoder

* **Apa itu Autoencoder?** Autoencoder adalah ANN yang dilatih untuk menyalin inputnya ke outputnya sendiri. Meskipun terdengar sepele, batasan-batasan yang diterapkan pada jaringan memaksa autoencoder untuk mempelajari representasi data yang efisien dan fitur-fitur penting dalam data.
* **Komponen Autoencoder**: Selalu terdiri dari dua bagian:
    * **Encoder (recognition network)**: Mengubah input menjadi representasi laten atau *coding*.
    * **Decoder (generative network)**: Mengubah representasi internal (coding) kembali menjadi output yang disebut *rekonstruksi*.
* **Fungsi Kerugian**: Umumnya menggunakan *reconstruction loss* yang memberikan penalti jika rekonstruksi berbeda dari input.
* **Undercomplete Autoencoder**: Lapisan *coding* memiliki dimensi yang lebih rendah daripada input dan output. Ini memaksa model untuk belajar fitur paling penting dan membuang yang tidak penting.
* **PCA dengan Autoencoder Linear**: Jika autoencoder hanya menggunakan aktivasi linear dan MSE sebagai fungsi kerugian, ia secara efektif melakukan Principal Component Analysis (PCA).

### Variasi Autoencoder:

* **Stacked Autoencoder**: Autoencoder dengan banyak lapisan tersembunyi. Arsitekturnya biasanya simetris terhadap lapisan *coding*.
    * **Tying Weights**: Teknik umum di autoencoder simetris di mana bobot lapisan decoder diikat (dibuat sama dengan transpose) bobot lapisan encoder, mengurangi jumlah bobot model dan mempercepat pelatihan.
    * **Greedy Layer-wise Training**: Melatih satu autoencoder dangkal pada satu waktu, lalu menumpuknya menjadi autoencoder bertumpuk yang lebih dalam. (Teknik ini kurang umum saat ini dibandingkan pelatihan end-to-end).
    * **Aplikasi**: Reduksi dimensi untuk visualisasi (misalnya dikombinasikan dengan t-SNE) dan *unsupervised pretraining* untuk tugas-tugas terawasi.
* **Convolutional Autoencoder**: Digunakan untuk data gambar, encoder-nya adalah CNN dan decoder-nya menggunakan *transpose convolutional layers* untuk *upsampling*.
* **Recurrent Autoencoder**: Digunakan untuk data sekuensial (deret waktu, teks), encoder-nya adalah RNN *sequence-to-vector* dan decoder-nya adalah RNN *vector-to-sequence*.
* **Denoising Autoencoder**: Menambahkan *noise* pada input dan melatih autoencoder untuk merekonstruksi input asli yang bebas *noise*. Ini dapat digunakan untuk visualisasi, *pretraining*, atau penghapusan *noise* dari gambar.
* **Sparse Autoencoder**: Mendorong model untuk mengurangi jumlah neuron aktif di lapisan *coding* dengan menambahkan penalti *sparsity* ke fungsi kerugian. Sering menggunakan Divergence Kullback–Leibler (KL) untuk penalti ini.
* **Variational Autoencoder (VAE)**: Autoencoder probabilistik dan generatif. Encoder menghasilkan *mean* ($\mu$) dan *standard deviation* ($\sigma$), dan *coding* diambil sampelnya dari distribusi Gaussian ini. Fungsi kerugiannya menggabungkan *reconstruction loss* dan *latent loss* (Divergence KL yang mendorong *coding* agar terlihat Gaussian). VAE dapat menghasilkan instans baru yang realistis dan memungkinkan *semantic interpolation*.

## 2. Generative Adversarial Networks (GANs)

* **Apa itu GAN?** Diperkenalkan pada tahun 2014, GAN terdiri dari dua jaringan saraf yang bersaing satu sama lain dalam permainan *zero-sum*:
    * **Generator**: Mengambil input acak (misalnya, *Gaussian noise*) dan menghasilkan data baru yang realistis].
    * **Discriminator**: Menerima data asli (dari set pelatihan) atau data palsu (dari generator) dan harus membedakan mana yang asli dan mana yang palsu.
* **Pelatihan GAN**: Terjadi dalam dua fase yang berlawanan di setiap iterasi pelatihan:
    1.  **Melatih Discriminator**: Diskriminator dilatih untuk mengklasifikasikan gambar asli sebagai 'asli' (label 1) dan gambar palsu sebagai 'palsu' (label 0). Bobot generator dibekukan.
    2.  **Melatih Generator**: Generator dilatih untuk menghasilkan gambar yang dapat menipu diskriminator agar mengklasifikasikannya sebagai 'asli' (label 1). Bobot diskriminator dibekukan.
* **Tantangan Pelatihan GANs**:
    * **Mode Collapse**: Generator menghasilkan output yang kurang beragam karena menemukan cara untuk menipu diskriminator dengan subset kecil dari data yang realistis].
    * **Ketidakstabilan**: Parameter generator dan diskriminator dapat berosilasi dan menjadi tidak stabil.
    * **Sensitivitas Hyperparameter**: GANs sangat sensitif terhadap *hyperparameter*.
* **Deep Convolutional GANs (DCGANs)**: Arsitektur GAN yang menggunakan jaringan konvolusional yang lebih dalam untuk menghasilkan gambar yang lebih besar dan stabil. Mengganti lapisan *pooling* dengan *strided/transposed convolutions* dan menggunakan *Batch Normalization* serta fungsi aktivasi tertentu.
* **Progressive Growing of GANs**: Teknik yang secara bertahap menambahkan lapisan konvolusional ke generator dan diskriminator selama pelatihan untuk menghasilkan gambar yang semakin besar (misalnya dari 4x4 hingga 1024x1024). Ini juga memperkenalkan teknik seperti *minibatch standard deviation layer*, *equalized learning rate*, dan *pixelwise normalization layer* untuk meningkatkan keragaman dan stabilitas.
* **StyleGANs**: Menggunakan teknik *style transfer* dalam generator untuk memastikan gambar yang dihasilkan memiliki struktur lokal yang sama dengan gambar pelatihan di setiap skala, meningkatkan kualitas gambar secara signifikan. Ini melibatkan *mapping network*, *synthesis network*, penambahan *noise* independen, dan *mixing regularization*.
