# Numerical experiments

This repository contains a set of Jupyter notebooks for reproducing the results of the paper **Priorconditioned sparsity-promoting projection methods for deterministic and Bayesian linear inverse problems**, by Jonathan Lindbloom, Mirjeta Pasha, Jan Glaubitz, and Youssef Marzouk.

## CPU only

A conda environment sufficient for running the notebooks can be installed via:

```sh
conda env create -f environment.yml
conda activate preconditioned-irls3
```

This is enough to run all six notebooks on the CPU, including tomography. This also installs [TRIPs-Py v1.0.0](https://github.com/mpasha3/trips-py).

## With GPU (optional)

For GPU support, use **CuPy 13.3.0** with **CUDA 12.6**. See the [CuPy installation instructions](https://docs.cupy.dev/en/v13.3.0/install.html). **The tomography results in the paper were produced using this GPU option.** Here the tomography notebooks automatically use the GPU preconditioner when available and fall back to the CPU otherwise. All 1D examples were produced using only CPU. 

## Run

From this directory, with the environment activated (you may also need to install Jupyter notebooks into the environment):

```sh
jupyter lab
```

Then you may open a notebook to reproduce the corresponding figure from the paper:

| Notebook                          | Outputs                  |
| --------------------------------- | ------------------------ |
| `01_intro.ipynb`                | Figures 1–2             |
| `02_1d_comparison.ipynb`        | Tables 1–2 and Figure 4 |
| `03_spectrum.ipynb`             | Figure 3                 |
| `04_epsilon.ipynb`              | Figure 5                 |
| `05_tomography.ipynb`           | Figures 6–9 and Table 3 |
| `06_tomography_nonconvex.ipynb` | Table 4                  |

The notebooks write results and figures to `results/`. The epsilon sweep and tomography runs can take a while (particularly if GPU is not used).

AI assistance was used after the original submission to reorganize and adapt the authors’ existing code for the supplementary reproducibility repository. The authors assume responsibility for all content.
