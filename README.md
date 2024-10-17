# **Text-to-Speech**
<h2>Aim:</h2>
To develop a robust end-to-end Transformer-based Text-to-Speech (TTS) model that efficiently converts textual input into natural, high-quality speech output. The model aims to leverage the self-attention mechanism to capture long-range dependencies in text sequences, enabling more accurate prosody, intonation, and contextual understanding compared to traditional models. The goal is to create a system that can generalize well across various languages and speaking styles, ensuring smooth, realistic voice synthesis with minimal preprocessing and training time.

<h2>Details:</h2>
<ul>
  <li>A Pytorch Implementation of end-to-end Speech Synthesis using Transformer Network.</li>
  <li>This model can be trained almost 3 to 4 times faster than autoregressive models, since Transformers lie under one of the fast computing non-autoregressive models. The quality of the speech was retrieved.</li>
  <li>We learned the post network using CBHG(Convolutional Bank + Highway network + GRU) model of tacotron and converted the spectrogram into raw wave using griffin-lim algorithm, and in future we want to use pre-trained hifi-gan vocoder for generating raw audio.</li>
</ul>
<h2>Transformer Architecture</h2>
<img src="png/model.png">
<h2>Requirements</h2>
<ul>
  <li>Install python==3.11.10</li>
  <li>Install requirements:</li>
<pre>pip install -r requirements.txt</pre>
</ul>
