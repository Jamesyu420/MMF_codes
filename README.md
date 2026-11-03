# Mixture Matrix Factorization (MMF)

This repository contains the software and analysis code for Mixture Matrix Factorization (MMF), a reference-guided statistical model for identifying cell-type-specific spatial gene programs from spatial transcriptomic data.

## Overview

MMF represents spatial gene expression as a mixture of cell types, with nonnegative gene loadings and spatial program activities within each type. Variational inference and backfitting estimate the model and select the number of programs.

## Repository structure

```text
├── MMF
├── Simulation
├── Mouse_brain
├── Colorectal_cancer
└── README.md
```

- **MMF** contains the model, variational inference, program selection and shared utilities.
- **Simulation** contains data generation, simulation studies, method comparisons, evaluation and plotting code.
- **Mouse_brain** contains preprocessing and analysis code for the two mouse brain Visium sections, including comparator methods and Figures 4–6.
- **Colorectal_cancer** contains preprocessing, fitting and plotting code for the selected Visium HD and Xenium analysis in Figure 7.

## Installation

```bash
pip install -r MMF/requirements.txt
pip install -e ./MMF
```

## Contact

If you have any problem, please feel free to contact [Prof. Xianghong Hu](mailto:huxh@szu.edu.cn) and [Baichen Yu](mailto:baichen.yu@stu.pku.edu.cn).