# 🚀 Towards Ambiguity-Free Spatial Foundation Model

[![📄 arXiv](https://img.shields.io/badge/arXiv-2503.06014-red.svg)](https://arxiv.org/abs/2503.06014)
[![🎥 Video Demo](https://img.shields.io/badge/Video%20Demo-Watch-blue.svg)](https://www.youtube.com/watch?v=38aSFah2jds)
[![📝 License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![🛠️ Planned Full Code/Data Release: May 2025](https://img.shields.io/badge/Full%20Release-May%202025-brightgreen.svg)]()

---

## 🔍 Introduction

This repository supports the paper **"Towards Ambiguity-Free Spatial Foundation Model: Rethinking and Decoupling Depth Ambiguity."** It provides code for benchmarking monocular depth estimation models on standard (DA-2K) and ambiguity-focused (MD-3k) datasets, implementing Laplacian Visual Prompting (LVP), and various analysis/visualization utilities.

Key contributions include:

✅ **MD-3k Benchmark**: A new dataset for evaluating multi-layer depth ambiguity and model biases. Details in [`./dataset/MD3K_Dataset_README.md`](./dataset/MD3K_Dataset_README.md).

✅ **Laplacian Visual Prompting (LVP)**: A training-free spectral prompting technique implemented in the benchmarking script.

✅ **Depth Bias Analysis**: Code to compute metrics revealing model layer preferences.

<p align="center">
  <img src="./assets/pipeline.png" width="90%" alt="Towards Ambiguity-Free Multi-Hypothesis Spatial Foundation Model"/>
  <br>
  <em><b>🖼️ Figure 1. Motivation:</b> 3D spatial understanding, powered by (a) sensors and (b) algorithms, has been confined to a biased single-layer depth representation. (c) Existing methods collapse in complex 3D scenarios, particularly in ambiguous scenes like those with transparency. (d) We propose Laplacian Visual Prompting (LVP) to overcome this limitation, enabling Spatial Foundation Models to derive multi-hypothesis depth, unlocking ambiguity-free spatial understanding.</em>
</p>

---

## 🏗️ Project Structure

```.
├── src/                      # Source code for experiments
│   ├── depth_estimation_mp.py # Multi-GPU depth map generation (RGB, LVP)
│   ├── DA2K_eval.py           # Evaluation script for DA-2K
│   ├── DA2K_stat.py           # Statistics calculation for DA-2K
│   ├── MD3K_eval.py           # Evaluation script for MD-3K
│   ├── MD3K_stat.py           # SRA-1 calculation for MD-3K
│   ├── MD3K_stat_SEP.py       # SRA-1/2 calculation for MD-3K (seperate)
│   ├── MD3K_stat_SRA2.py      # SRA-2 calculation for MD-3K
│   ├── MD3K_stat_com.py       # ML-SRA calculation for MD-3K
├── utils/                      # Source code for utilities
│   ├── json_vis.py            # Annotation visualization utility
│   ├── mask_eval.py           # Mask evaluation utility
│   ├── mask_distribution.py   # Mask distribution analysis utility
│   ├── depth_3D.py            # Interactive 3D depth visualization utility
│   └── vis_combined.py        # Combined visualization generation utility
├── dataset/                  # Dataset specific information and potentially data files
│   └── MD3K_Dataset_README.md # Detailed README for the MD-3K benchmark
├── data/                     # (Suggested) Location for datasets (DA-2K, MD-3K images/masks/annotations)
│   ├── DA-2K/
│   └── MD-3K/
├── assets/                   # Images for README
├── requirements.txt          # Python package dependencies
└── README.md                 # This file
```
*(Note: You need to create the `./data` directory and place the DA-2K and MD-3K datasets inside, or configure the paths in the `src/*.py` and `utils/*.py` scripts accordingly.)*

---

## ⚙️ Setup & Instructions for Experiments

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Xiaohao-Xu/Ambiguity-in-Space.git
    cd Ambiguity-in-Space
    ```
2.  **Create Environment & Install Dependencies:**
    *   Create a conda environment (or use your preferred environment manager):
        ```bash
        conda create -n mhsfm python=3.9 # Or your preferred python version
        conda activate mhsfm
        ```
    *   Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
3.  **Download Datasets:**
    *   **MD-3K Benchmark:**
        *   Download the MD-3k dataset (including RGB images sourced from GDD, masks, and `annotations.json`) using the provided links:
            *   [Google Drive Link (501MB)](https://drive.google.com/file/d/1bboSHFA5mzttK4Qpb_Vh2gSt1wx1CEuX/view?usp=sharing)
        *   Place the downloaded `MD-3K.zip`, unzip it into the folder `MD-3K`, and put the folder under the suggested `./data` directory (i.e., `./data/MD-3K/`).
        *   See [`./dataset/MD3K_Dataset_README.md`](./dataset/MD3K_Dataset_README.md) for more details on the MD-3k dataset structure and contents.
    *   **DA-2K Benchmark:**
        *   Obtain the DA-2K dataset by following the instructions in the [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2).
        *   Place the downloaded `DA-2K` folder into the suggested `./data` directory (i.e., `./data/DA-2K/`).

    *(The suggested final structure would be `./data/MD-3K/...` and `./data/DA-2K/...`. If you place datasets elsewhere, remember to update the paths in the `src/*.py` scripts.)*

---

<details>
<summary>📊 Click to Expand: Reproducing Results (Using Scripts in ./src)</summary>

This section guides you through using the scripts located in the `./src` directory to reproduce the main benchmarking experiments from the paper.

**⚠️ Important:** Before running any script, **modify the hardcoded paths** inside it (e.g., `input_folder`, `output_folder`, `json_path`, `depth_res_path`, `raw_path`, dataset paths often referencing `/media/hdd2/users/xiaohao/...`) to point to your local dataset locations and desired output directories.

**Workflow:**

1.  **Generate Depth Maps:** Use `src/depth_estimation_mp.py`. Run multiple times for different configs (RGB, LVP, Blur) and datasets (DA-2K, MD-3K), saving outputs to *unique* directories.
2.  **Evaluate:** Use `src/DA2K_eval.py` or `src/MD3K_eval.py` pointing to the generated depth maps.
3.  **Calculate Statistics:** Use `src/DA2K_stat.py` or `src/MD3K_stat*.py` pointing to the evaluation JSONs.

**Step-by-Step Guide:**

1.  **Generate Depth Maps (`src/depth_estimation_mp.py`)**
    *   **Purpose:** Generate 16-bit PNG depth maps using various models.
    *   **Configuration:** Modify internal paths. Set `gau_size` / `lvp_type` for desired input processing (RGB: 0/-1; LVP: 0/0; Blur: >0/-1). Choose appropriate `input_folder` (DA-2K or MD-3K). Set unique `output_folder`. Adjust `num_gpus`.
    *   **Execution:** `python src/depth_estimation_mp.py` (Repeat for each config/dataset).

2.  **Evaluate on DA-2K (`src/DA2K_eval.py`)**
    *   **Purpose:** Compare DA-2K depth maps against annotations.
    *   **Configuration:** Modify internal paths (`json_path` to DA-2K annotations, `depth_res_path` to DA-2K depth maps from Step 1, `raw_path` to DA-2K root, `output_folder` for eval JSONs).
    *   **Execution:** `python src/DA2K_eval.py`

3.  **Calculate DA-2K Statistics (`src/DA2K_stat.py`)**
    *   **Purpose:** Calculate SRA for DA-2K.
    *   **Configuration:** Modify `json_folder` to DA-2K eval JSONs from Step 2. **Adapt script loop/logic** to process all models if needed.
    *   **Execution:** `python src/DA2K_stat.py`

4.  **Evaluate on MD-3K (`src/MD3K_eval.py`)**
    *   **Purpose:** Compare MD-3K depth maps against annotations.
    *   **Configuration:** Modify internal paths (`json_path` to MD-3K annotations, `depth_res_path` to MD-3K depth maps from Step 1, `raw_path` to MD-3K root, `output_folder` for eval JSONs).
    *   **Execution:** `python src/MD3K_eval.py`

5.  **Calculate MD-3K Statistics (`src/MD3K_stat*.py`)**
    *   **Purpose:** Calculate SRA1, SRA2, ML-SRA metrics.
    *   **Configuration:** Modify `json_folder` (to MD-3K eval JSONs), `raw_json_path`. For `MD3K_stat_com.py`, set `json_folder_RGB`, `json_folder_LAP`.
    *   **Execution:**
        *   SRA1: `python src/MD3K_stat.py`
        *   SRA1 (Subset): Modify label filter in `src/MD3K_stat_SEP.py` then run `python src/MD3K_stat_SEP.py`
        *   SRA2: `python src/MD3K_stat_SRA2.py`
        *   ML-SRA: Adapt logic in `src/MD3K_stat_com.py` then run `python src/MD3K_stat_com.py`
        *   Layer Preference (α): Calculate manually: `SRA(2) - SRA(1)`.

</details>

---

<details>
<summary>🛠️ Click to Expand: Utilities and Visualization Tools (in ./utils)</summary>

The `./utils` directory also contains utility scripts for analysis and visualization:

1.  **Annotation Visualization (`utils/json_vis.py`)**
    *   **Purpose:** Visualize MD-3K annotations (points, labels) on images/masks.
    *   **Configuration:** Modify internal paths for images, masks, annotations file.
    *   **Execution:** `python utils/json_vis.py` 

2.  **Mask Evaluation (`utils/mask_eval.py`)**
    *   **Purpose:** Evaluate predicted segmentation masks against MD-3K ground truth masks.
    *   **Configuration:** Modify internal paths for predicted masks (`pred_dir`), GT masks (`gt_dir`), and optionally filter based on another dir (`depth_anything_dir`).
    *   **Execution:** `python utils/mask_eval.py`

3.  **Mask Spatial Distribution (`utils/mask_distribution.py`)**
    *   **Purpose:** Analyze and visualize the spatial distribution of MD-3K masks.
    *   **Configuration:** Modify internal paths for annotations file, mask directory, output directory.
    *   **Execution:** `python utils/mask_distribution.py`

4.  **3D Depth Visualization (`utils/depth_3D.py`)**
    *   **Purpose:** Interactively compare two depth maps (e.g., RGB vs LVP) in 3D, coloring by mask.
    *   **Configuration:** Modify internal paths for the two depth directories and the mask directory.
    *   **Execution:** `python utils/depth_3D.py <image_id>`

5.  **Combined Visualization (`utils/vis_combined.py`)**
    *   **Purpose:** Create static combined images (e.g., RGB, Depth1, GenRGB1, Depth2, GenRGB2).
    *   **Configuration:** Modify internal file paths and potentially the image loading/concatenation logic.
    *   **Execution:** `python utils/vis_combined.py` (Adapt `process_all_images` if needed).

</details>

---

## 📄 MultiDepth-3K (MD-3K) Dataset

The MD-3k benchmark is central to this work for evaluating depth ambiguity. For detailed information about its structure, annotations, metrics, and usage, please refer to the dedicated dataset README:

➡️ [`./dataset/MD3K_Dataset_README.md`](./dataset/MD3K_Dataset_README.md)

---

## 🏆 Key Contributions Summary 

🔹 **MH-SFMs Paradigm**: Reformulating generic spatial understanding and depth estimation as multi-hypothesis inference.

🔹 **MD-3k Benchmark**: Novel dataset and metrics for multi-layer depth and bias evaluation.

🔹 **Depth Bias Analysis**: Comprehensive study of layer preference biases in foundation models.

🔹 **Laplacian Visual Prompting (LVP)**: Training-free method for multi-hypothesis depth using pre-trained models.

🔹 **Extensive Validation**: Demonstrating LVP’s benefits in multi-layer depth estimation and downstream tasks.

---

## Citation


```bibtex
@article{xu2025towards,
  title={Towards Ambiguity-Free Spatial Foundation Model: Rethinking and Decoupling Depth Ambiguity},
  author={Xu, Xiaohao and Xue, Feng and Li, Xiang and Li, Haowei and Yang, Shusheng and Zhang, Tianyi and Johnson-Roberson, Matthew and Huang, Xiaonan},
  journal={arXiv preprint arXiv:2503.06014},
  year={2025}
}
```

```bibtex
@inproceedings{
xu2025towards,
title={Towards Multi-Hypothesis Spatial Foundation Model: Rethinking and Decoupling Spatial Ambiguity via Laplacian Visual Prompting},
author={Xiaohao Xu and Feng Xue and Xiang Li and Haowei Li and Tianyi Zhang and Matthew Johnson-Roberson and Xiaonan Huang},
booktitle={ICLR 2025 Workshop on Foundation Models in the Wild},
year={2025},
url={https://openreview.net/forum?id=gngOFExtxN}
}
```


---

## 📫 Contact
For updates on the full release, please ⭐ star this repo. Got questions? Reach out to Xiaohao Xu: xiaohaox[AT]umich.edu .
