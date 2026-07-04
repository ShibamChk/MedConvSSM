# MedConvSSM

This repository contains the Jupyter notebook implementation for **MedConvSSM**, a lightweight hybrid Mamba-CNN framework for medical image classification.

MedConvSSM combines a convolutional stem for local texture extraction with a compact Vision-Mamba sequence modeling module for global spatial context. The model first downsamples medical images to a compact feature map, converts the feature map into patch tokens, applies 2D selective scan based Mamba blocks, and uses fused average and max pooling for final classification.

The repository is intended to support reproducibility for the manuscript:

**MedConvSSM: A Hybrid Lightweight Mamba-CNN Framework with Patch Embedding for Medical Image Classification**

## Repository contents

```text
.
├── MedConvSSM.ipynb
└── README.md
```

## Method overview

The proposed framework follows this high-level pipeline:

1. **Input preprocessing**
   - Images are resized to `224 x 224`.
   - Images are converted to tensors.
   - Normalization is computed from the training split.

2. **CNN stem**
   - A lightweight convolutional stem extracts local texture and edge features.
   - The stem reduces the input image resolution before tokenization.

3. **Patch embedding**
   - The CNN feature map is divided into compact patch tokens.
   - The final architecture uses a compact token representation suitable for efficient Mamba processing.

4. **Vision-Mamba encoder**
   - The token sequence is processed by three Vision-Mamba blocks.
   - Each block uses 2D selective scan style sequence mixing to model spatial context efficiently.

5. **Classification head**
   - Average pooling and max pooling are fused.
   - A lightweight classifier produces the final class prediction.

## Datasets

The datasets used in the manuscript are publicly available from the following sources. The datasets are not redistributed in this repository. Users should download them from the original sources and follow their respective licenses, citation requirements, and usage policies.

| Dataset group | Dataset names used in the study | Source |
|---|---|---|
| MedMNIST v2 | PathMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, BreastMNIST, BloodMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST, TissueMNIST | https://medmnist.com/ |
| CPN X-ray | Chest X-ray dataset for COVID-19, Normal, and Pneumonia classification | https://data.mendeley.com/datasets/rscbjbr9sj/2 |
| Kvasir | Gastrointestinal endoscopy image dataset | https://datasets.simula.no/kvasir/ |

## Data availability

All datasets used in this study are publicly available. The MedMNIST v2 benchmark datasets are available from the official MedMNIST repository at https://medmnist.com/. The CPN X-ray dataset is available from Mendeley Data at https://data.mendeley.com/datasets/rscbjbr9sj/2. The Kvasir gastrointestinal endoscopy dataset is available from Simula Datasets at https://datasets.simula.no/kvasir/. No new clinical dataset was generated or redistributed in this repository.

## Code availability

The implementation of the proposed MedConvSSM framework is provided in `MedConvSSM.ipynb`. The notebook includes the model definition, training pipeline, evaluation metrics, and experimental utilities required to reproduce the training and testing workflow.

## Environment

The notebook was developed using PyTorch with CUDA acceleration. A GPU environment is recommended, especially because the implementation uses Mamba-related CUDA kernels.

Main dependencies include:

```text
torch
torchvision
numpy
pandas
scikit-learn
Pillow
matplotlib
seaborn
einops
medmnist
torchinfo
ptflops
causal-conv1d
mamba-ssm
```

A minimal installation command is:

```bash
pip install torch torchvision numpy pandas scikit-learn pillow matplotlib seaborn einops medmnist torchinfo ptflops causal-conv1d mamba-ssm
```

Depending on the CUDA version and operating system, `mamba-ssm` and `causal-conv1d` may require compatible PyTorch and CUDA builds.

## How to run

1. Clone the repository.

```bash
git clone <repository-url>
cd <repository-name>
```

2. Open the notebook.

```bash
jupyter notebook MedConvSSM.ipynb
```

3. Run the cells sequentially.

The notebook contains:

- Library imports
- Random seed setup
- MedConvSSM model definition
- Dataset wrapper and preprocessing pipeline
- Training loop
- Validation loop
- Testing and metric calculation
- Model complexity estimation utilities

For MedMNIST datasets, the notebook can download the selected dataset through the `medmnist` Python package. For external datasets such as CPN X-ray and Kvasir, download the datasets manually from the official links above and adjust the dataset loading section according to the local directory structure.

## Reproducibility notes

- The notebook sets a fixed random seed for reproducibility.
- Experiments in the manuscript used images resized to `224 x 224`.
- MedMNIST datasets use their official train, validation, and test splits.
- External datasets use a train, validation, and test split as described in the manuscript.
- Dataset files are not included in this repository.
- The notebook is designed for training the model from scratch.

## Results reported in the manuscript

The final MedConvSSM configuration reported in the manuscript uses:

- 3 Vision-Mamba blocks
- Embedding dimension of 256
- Patch-based tokenization
- Fused average and max pooling
- Approximately 3.67 million parameters
- Approximately 0.33 GMACs

The manuscript reports strong performance across multiple medical image classification datasets, including MedMNIST v2 tasks, CPN X-ray, and Kvasir.

## Citation

If you use this code or build upon this work, please cite the manuscript after publication.

```bibtex
@article{medconvssm,
  title   = {MedConvSSM: A Hybrid Lightweight Mamba-CNN Framework with Patch Embedding for Medical Image Classification},
  author  = {Chakrabarty, Amitabha and Das, Sourav and Chakraborty, Shibam and Hasan Turja, Sakibul and Islam Rifat, Md. Tanvirul and Datta, Nirjhor and Aziz, Azwad and Hakimi, Halimaton and {Siti Sarah Binti Maidin}},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

Please add an appropriate license before public release. If there are no institutional restrictions, an open-source license such as MIT or Apache-2.0 can be used for the code. Dataset licenses remain governed by the original dataset providers.
