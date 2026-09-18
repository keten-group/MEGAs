# MEGAs — binary Bar–Y superlattices

Machine-learning code accompanying *Binary superlattices enable programmable tensile strength and ductility in magneto-elastic granular architectures*.

Given the branch lengths of the two-branch (Bar) and three-branch (Y) units, $\mathbf{x} = [L_{0,\mathrm{Bar}},\, L_{0,\mathrm{Y}}]$, the framework predicts

1. the **configuration type** of the assembled superlattice (Bar-dominant, balanced, Y-dominant or expanded),
2. the **magnetic cluster configuration** of contracted systems (five geometric parameters, from which the whole superlattice is reconstructed), and
3. the **tensile strength and toughness** with a calibrated predictive uncertainty,

and uses the property surrogate to find the **Pareto-optimal designs** at a chosen level of confidence.

## Notebooks

Run them in this order; each one reads only `data/` and `mega_utils.py`, and the later ones load the models saved by the earlier ones.

| notebook | task | model | saves |
|---|---|---|---|
| `configuration_type_classification.ipynb` | configuration type of 176 designs | Gaussian process classifier (scikit-learn); compared with *k*-NN, SVM and random forest over 20 random splits | `models/gpc_classifier.joblib` |
| `cluster_configuration_geometry_reconstruction.ipynb` | cluster geometry of the 118 contracted designs | multi-output Gaussian process (GPyTorch, intrinsic-coregionalisation kernel); compared with *k*-NN, random forest and a polynomial response surface, scored by the IoU of the reconstructed clusters | `models/gp_cluster_config.pt` |
| `mechanical_property_prediction.ipynb` | strength and toughness of the 118 contracted designs | bootstrap ensemble of 20 neural networks (PyTorch) with the uncertainty calibration of Palmer et al. (2022); architecture search, benchmark against GPR / *k*-NN / polynomial RSM / random forest, calibration diagnostics, prediction intervals and learning curve | `models/nn_ensemble/` |
| `pareto_optimization.ipynb` | Pareto sets of strength and toughness | grid search (0.02 cm) over the design space with the GPC as feasibility filter; Pareto sets of the lower confidence bounds $\hat\mu - \kappa\hat\sigma_{\mathrm{cal}}$ | — |

Tables are written to `results/` and figures to `figures/`; the correspondence with the supplementary tables and figures of the paper is listed at the end of each notebook.

## Data

The notebooks read the supplementary data files of the paper:

| file | sheet | content |
|---|---|---|
| `data/Supplementary Data 3.xlsx` | `strength_toughness` | tensile strength and toughness of every simulation replicate of the 118 contracted designs |
| `data/Supplementary Data 4.xlsx` | `configuration type` | configuration type of the 176 designs on the 11 × 16 grid of branch lengths |
| `data/Supplementary Data 4.xlsx` | `1 Bar-dominant`, `2 Balanced`, `3 Y-dominant` | cluster configuration parameters of the 118 contracted designs |

## Installation

```bash
pip install -r requirements.txt
jupyter lab
```

Tested with Python 3.13, NumPy 1.26, pandas 3.0, scikit-learn 1.8, PyTorch 2.13 (CPU), GPyTorch 1.15 and Shapely 2.1. All computations run on CPU; the notebooks use `joblib` to parallelise the repeated splits and ensemble members (`N_JOBS` at the top of each notebook). The property-prediction notebook trains about 950 small networks and takes roughly 10–20 minutes on a 16-core workstation, the others a few minutes each. All random seeds are fixed, so the reported numbers are reproducible.
