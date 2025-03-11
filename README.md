# 🚀 Towards Ambiguity-Free Spatial Foundation Model  

[![📄 arXiv](https://img.shields.io/badge/arXiv-2503.06014-red.svg)](https://arxiv.org/abs/2503.06014)  
[![🎥 Video Demo](https://img.shields.io/badge/Video%20Demo-Watch-blue.svg)](https://www.youtube.com/watch?v=38aSFah2jds)  
[![📝 License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![🛠️ Planned Code Release: May 2025](https://img.shields.io/badge/Code%20Release-May%202025-brightgreen.svg)]()  

---

## 🔍 Introduction  

This repository provides resources for the paper **"Towards Ambiguity-Free Spatial Foundation Model: Rethinking and Decoupling Depth Ambiguity."**  

We tackle the challenge of **depth ambiguity** in spatial scene understanding, especially in **transparent and complex environments**. Traditional monocular depth models struggle with multi-layer depth perception. Our work introduces **Multi-Hypothesis Spatial Foundation Models (MH-SFMs)** to overcome these limitations, featuring:  

✅ **MD-3k Benchmark** – A dataset to evaluate multi-layer depth understanding and biases.  
✅ **Laplacian Visual Prompting (LVP)** – A training-free method to extract hidden depth from pre-trained models.  

Experiments confirm **LVP's effectiveness** for zero-shot multi-layer depth estimation and its advantages in downstream applications.  

<p align="center">
  <img src="./assets/pipeline.png" width="90%" alt="Towards Ambiguity-Free Multi-Hypothesis Spatial Foundation Model"/>
  <br>
  <em><b>🖼️ Figure 1. Motivation:</b> 3D spatial understanding, powered by (a) sensors and (b) algorithms, has been confined to a biased single-layer depth representation. (c) Existing methods collapse in complex 3D scenarios, particularly in ambiguous scenes like those with transparency. (d) We propose Laplacian Visual Prompting (LVP) to overcome this limitation, enabling Spatial Foundation Models to derive multi-hypothesis depth, unlocking ambiguity-free spatial understanding.</em>
</p>  

---

## 🏆 Key Contributions  

🔹 **MH-SFMs Paradigm** – Reformulating depth estimation as multi-hypothesis inference to handle spatial ambiguity.  
🔹 **MD-3k Benchmark** – A novel dataset and evaluation metrics for multi-layer depth perception and model biases.  
🔹 **Depth Bias Analysis** – A comprehensive study of depth biases in existing models using MD-3k.  
🔹 **Laplacian Visual Prompting (LVP)** – A training-free method for multi-hypothesis depth estimation using pre-trained models.  
🔹 **Extensive Validation** – Demonstrating LVP’s impact and advantages across various downstream applications.  

---

## 🚀 Getting Started (Code Release: **May 2025**)  

🗓️ **Planned Release**: May 2025. Stay tuned for:  

📂 **MD-3k Dataset** – Instructions for downloading and using our benchmark dataset.  
💡 **LVP Implementation** – Code for running Laplacian Visual Prompting on pre-trained models.  
📑 **Reproducibility Guide** – Example scripts for reproducing our experiments.  

---

📩 **Stay Updated!**  
For updates, please ⭐ star this repo and check back for the official release.  
