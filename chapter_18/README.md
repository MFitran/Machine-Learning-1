# Bab 18: Reinforcement Learning

Bab 18 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" membahas secara mendalam Reinforcement Learning (RL), sebuah paradigma pembelajaran mesin di mana agen belajar membuat keputusan optimal melalui interaksi dengan lingkungannya untuk memaksimalkan *reward* yang diharapkan dari waktu ke waktu.

Bab ini diawali dengan pengenalan konsep dasar RL, termasuk elemen-elemen kunci seperti **agen (agent)**, **lingkungan (environment)**, **aksi (actions)**, **observasi (observations)**, dan **reward**. Disajikan pula contoh-contoh aplikasi RL yang beragam, mulai dari robotika, bermain game (seperti Atari dan Go), sistem rekomendasi, hingga pengaturan termostat cerdas].

Selanjutnya, bab ini menjelaskan konsep **policy**, yaitu algoritma yang digunakan agen untuk menentukan aksinya. Dijelaskan pula berbagai pendekatan untuk mencari *policy* yang optimal, termasuk metode *policy search* yang lebih sederhana seperti *brute force* dan algoritma genetika (meskipun kurang skalabel), hingga pengenalan awal tentang **Policy Gradients (PG)**.

Untuk memfasilitasi pelatihan agen RL, bab ini memperkenalkan **OpenAI Gym**, sebuah *toolkit* yang menyediakan berbagai lingkungan simulasi. Pembaca akan belajar cara membuat dan berinteraksi dengan lingkungan Gym, serta memahami struktur observasi dan aksi.

Salah satu tantangan utama dalam RL, yaitu **masalah *credit assignment***, dibahas secara rinci. Untuk mengatasinya, diperkenalkan konsep **faktor diskon ($\gamma$)** dan **Return** (jumlah *reward* di masa depan yang didiskon), serta **Action Advantage** untuk mengevaluasi kualitas aksi. Bagian ini juga menyertakan implementasi kode untuk algoritma **Policy Gradients (REINFORCE)** pada lingkungan CartPole, menunjukkan bagaimana agen dapat belajar menyeimbangkan tiang.

Bab ini kemudian beralih ke pembahasan **Markov Decision Processes (MDPs)** sebagai kerangka matematis untuk memodelkan masalah RL. Dijelaskan konsep **Optimal State Value ($V^*(s)$)** dan **Bellman Optimality Equation**, serta algoritma **Value Iteration** untuk menghitungnya. Lebih lanjut, diperkenalkan **Optimal State-Action Values (Q-Values, $Q^*(s,a)$)** dan algoritma **Q-Value Iteration** sebagai cara untuk menemukan *optimal policy*. Implementasi kode untuk Q-Value Iteration pada MDP sederhana juga disertakan.

Sebagai transisi ke masalah RL di dunia nyata di mana probabilitas transisi dan *reward* tidak diketahui sebelumnya, diperkenalkan **Temporal Difference (TD) Learning**. Kemudian, algoritma **Q-Learning** dijelaskan sebagai adaptasi dari Q-Value Iteration, yang belajar Q-Values dengan mengamati interaksi agen dalam lingkungan. Konsep **kebijakan eksplorasi** seperti **$\epsilon$-greedy policy** juga dibahas untuk memastikan agen menjelajahi lingkungan secara efektif.

Puncak dari pembahasan Q-Learning adalah **Approximate Q-Learning** dan **Deep Q-Learning (DQN)**, yang menggunakan Jaringan Saraf Tiruan untuk mengaproksimasi Q-Values pada MDP yang besar. Bab ini menyertakan implementasi rinci DQN untuk CartPole, termasuk penggunaan **replay buffer** untuk menstabilkan pelatihan. Juga diungkapkan tantangan **catastrophic forgetting** yang umum terjadi dalam pelatihan RL.

Untuk mengatasi ketidakstabilan pelatihan DQN, berbagai varian diperkenalkan:
* **Fixed Q-Value Targets**: Menggunakan dua jaringan (online dan target) untuk membuat target Q-Value lebih stabil.
* **Double DQN**: Mengatasi *overestimation* Q-Values.
* **Prioritized Experience Replay (PER)**: Mengambil sampel pengalaman yang lebih "penting" lebih sering.
* **Dueling DQN**: Mengurai Q-Value menjadi nilai *state* dan keuntungan aksi.

Bagian terakhir bab ini memperkenalkan **TF-Agents**, pustaka Reinforcement Learning berbasis TensorFlow dari Google. Pembaca akan belajar cara menggunakan TF-Agents untuk melatih agen bermain game Atari Breakout, mencakup konfigurasi lingkungan (termasuk praproses Atari), pembuatan **QNetwork** dan **DQN Agent**, penggunaan **replay buffer**, **collect driver**, dan **metrik pelatihan**.
