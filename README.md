<h1 align="center"><strong>OmniStyle: Filtering High Quality Style Transfer Data at Scale</strong></h1>

<p align="center">
  <a href="https://wangyephd.github.io/">Ye Wang<sup>1</sup></a>,
  Ruiqi Liu<sup>1</sup>,
  Jiang Lin<sup>2</sup>,
  Fei Liu<sup>3</sup>,
  <a href="https://is.nju.edu.cn/yzl_en/main.htm">Zili Yi<sup>2</sup></a>,
  <a href="https://yilinwang.org/">Yilin Wang<sup>4,*</sup></a>,
  <a href="https://ruim-jlu.github.io/#about">Rui Ma<sup>1,5,*</sup></a>
</p>

<p align="center">
  <sup>1</sup>School of Artificial Intelligence, Jilin University &nbsp;&nbsp; <br>
  <sup>2</sup>School of Intelligence Science and Technology, Nanjing University &nbsp;&nbsp; <br>
  <sup>3</sup>ByteDance 
  <sup>4</sup>Adobe &nbsp;&nbsp; <br>
  <sup>5</sup>Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, MOE, China<br>
  <sup>*</sup>Corresponding authors
</p>

<p align="center">
  <a href="https://wangyephd.github.io/projects/cvpr25_omnistyle.html">
    <img src="https://img.shields.io/badge/Project-OmniStyle-blue?style=flat-square"/>
  </a>
  <a href="https://arxiv.org/pdf/2505.14028">
    <img src="https://img.shields.io/badge/Paper-arXiv-green?style=flat-square"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Dataset-Open-orange?style=flat-square"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/HuggingFace-Model-yellow?style=flat-square"/>
  </a>
</p>


---

## 📢 News

- **[2025.07.11]** 🎉 Code and model weights for **OmniStyle** are now available!
- **[2025.07.05]** ✅ Released the [project page](https://wangyephd.github.io/projects/cvpr25_omnistyle.html).




---

🤖 <strong>OmniStyle</strong> is the first end-to-end style transfer framework based on the Diffusion Transformer (DiT) architecture, achieving high-quality 1K-resolution stylization by leveraging the large-scale, filtered OmniStyle-1M dataset. It supports both instruction- and image-guided stylization, enabling efficient and versatile style transfer across diverse styles.

🗂️ <strong>OmniStyle-1M</strong> is the first million-scale paired style transfer dataset, comprising over one million triplets of content, style, and stylized images across 1,000 diverse style categories. It provides strong supervision for learning controllable and generalizable style transfer models.

🧪 <strong>OmniStyle-150K</strong> is a high-quality subset of OmniStyle-1M, specifically filtered to train the OmniStyle model.


---
<h2>🛠️ TODO List</h2>
<ul>
  <li>🔜 Release <strong>OmniStyle-150K</strong>: The filtered high-quality subset used for training.</li>
  <li>🔜 Release <strong>Model weights</strong> and <strong>inference code</strong> for OmniStyle.</li>
</ul>

---


## 🧩 Installation & Environment Setup

We recommend creating a clean conda environment:

```bash
conda create -n omnistyle python=3.10 
conda activate omnistyle
# Install dependencies
pip install -r requirements.txt
```

---

## 📥 Checkpoints Download

You can download the pretrained **OmniStyle** model from Hugging Face:

👉 [https://huggingface.co/StyleXX/OmniStyle](https://huggingface.co/StyleXX/OmniStyle)


