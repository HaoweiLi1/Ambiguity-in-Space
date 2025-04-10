# MD-3k Benchmark Dataset README

## 1. Introduction

MD-3k is a benchmark dataset introduced in the paper ["Towards Ambiguity-Free Spatial Foundation Model: Rethinking and Decoupling Depth Ambiguity"](https://arxiv.org/abs/2503.06014). It is specifically designed to evaluate the ability of monocular depth estimation models, particularly foundation models, to handle **depth ambiguity** commonly found in scenes with transparent or reflective surfaces (e.g., glass).

The key challenges addressed by MD-3k are:
*   **Multi-Layer Depth:** Real-world ambiguous scenes often contain multiple depth layers at the same pixel location (e.g., the glass surface and the scene behind it). Standard datasets often lack annotations for this.
*   **Model Bias:** Existing depth models exhibit biases, preferentially estimating either the nearer or farther surface in ambiguous regions.
*   **Lack of Evaluation Metrics:** Traditional depth metrics do not adequately capture performance in multi-layer scenarios or quantify model bias.

MD-3k provides explicit annotations for multi-layer spatial relationships and enables the use of new metrics (SRA, ML-SRA, α) to rigorously assess model performance and bias in the context of depth ambiguity.

## 2. Dataset Characteristics

*   **Size:** 3,161 high-resolution (typically 720p) RGB images with masks of ambiguous regions and multi-layer spatial relationship annotations.
*   **Content:** Primarily indoor and outdoor scenes featuring glass doors, windows, display cases, etc.
*   **Ambiguity Focus:** Primarily transparency-induced ambiguity.
*   **Spatial Distribution:** Ambiguous regions exhibit a balanced spatial distribution with a slight center bias, reflecting natural scene compositions.

## 3. Suggested Dataset Layout

The benchmark dataset, when downloaded and extracted, should ideally be placed relative to the main project structure. The scripts in `../src` often assume a layout like this:

```
<your_project_root>/
├── src/
│   └── ... (scripts)
├── dataset/
│   └── MD3K_Dataset_README.md # This file
├── data/                     # Suggested data location
│   ├── MD3K/                 # Extracted MD-3K dataset folder
│   │   ├── images/           # RGB images 
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   ├── masks/            # Ground truth binary masks
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   └── annotations.json  # Core annotation file
│   └── DA-2K/                # Optional: DA-2K dataset location
│       └── ...
└── ... (other project files)
```
*Remember to configure the paths inside the `../src/*.py` scripts to point to your actual `data/MD3K` (or `data/DA-2K`) location if you place them differently.*

## 4. Annotation Format (`annotations.json`)

The `annotations.json` file maps image paths (relative to the `MD3K` root directory, e.g., `./images/group1/1.jpg`) to annotation pairs.

```json
{
  "./images/1.jpg": [
    {
      "point1": [X1, Y1],
      "point2": [X2, Y2],
      "label": L
    }
    // ... potentially more pairs for this image
  ],
  // ... more images
}
```

*   `point1`, `point2`: A sparse pair of `[x, y]` coordinates sampled within the ambiguous region of the image.
*   `label`: An integer indicating the target relationship for evaluation metrics:
    *   **`label: 1`**: Targets the "first"/dominant depth layer relationship (used for SRA1). Correctness depends on the depth model type (Metric vs. Relative) as defined in `MD3K_eval.py`.
    *   **`label: 2`**: Targets the "second"/hidden depth layer relationship (used for SRA2). Correctness depends on the depth model type (Metric vs. Relative) as defined in `MD3K_eval.py`.
    *   **`label: 3`**: (If present) Indicates cases where the relationship is unclear or annotators were unsure. These are typically skipped during evaluation.

The dataset is also conceptually divided into subsets based on annotation consistency (see paper, Sec 3.2):
*   **Same Subset:** Point pairs where the relative depth order between layers is consistent.
*   **Reverse Subset:** Point pairs where the relative depth order between layers is reversed. The `label` field implicitly defines which category a sample belongs to, used by scripts like `MD3K_stat_SEP.py`.

## 5. Evaluation Metrics & Corresponding Scripts

MD-3k enables calculating metrics defined in the paper using scripts in `../src`:

*   **SRA(1):** Accuracy for the first layer. Calculated by `../src/MD3K_stat.py`.
*   **SRA(2):** Accuracy for the second layer. Calculated by `../src/MD3K_stat_SRA2.py`.
*   **Depth Layer Preference (α):** Model bias (`SRA(2) - SRA(1)`). Calculate manually from SRA1/SRA2 results.
*   **ML-SRA:** Simultaneous accuracy for both layers (requires comparing results from two different depth map sets, e.g., RGB vs LVP). Calculated by `../src/MD3K_stat_com.py`.
*   **SRA(1) on Subsets:** Accuracy for specific subsets (e.g., 'Same'/'Reverse'). Calculated by `../src/MD3K_stat_SEP.py` (requires modifying the label filter).

Refer to the main project README (`../README.md`) for detailed execution instructions for these scripts.

## 6. Dataset Download

The MD-3k dataset (images, masks, annotations) can be downloaded from:

*   [Dropbox Link](<YOUR_DROPBOX_LINK_HERE>)
*   [Google Drive Link](<YOUR_GOOGLE_DRIVE_LINK_HERE>)

*(Please replace `<YOUR_..._LINK_HERE>` with the actual links)*

## 7. Usage and Citation

1.  **Download** the MD-3k dataset using the links above.
2.  **Organize** the extracted files (e.g., as suggested in Section 3).
3.  **Configure paths** in `../src/*.py` scripts to point to your dataset location.
4.  **Run evaluation scripts** located in `../src`.

If you use the MD-3k benchmark in your research, please cite the following paper:

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

## 8. Datasheet

For detailed information regarding the dataset's motivation, composition, collection process, recommended uses, and limitations, please refer to the Datasheet section (Appendix D) in the supplementary material of the [arXiv paper](https://arxiv.org/abs/2503.06014).
