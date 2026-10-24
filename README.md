> Review snapshot: start with [REVIEWER_README.md](REVIEWER_README.md) for package scope, anonymous data access, the supplied CPU demonstration and source-only weight availability.

# GIFT

**Generator Identification by Field Tomography**

Learning continuous-time flow dynamics from vorticity-field trajectories, with explicit generator components for prediction and equation-parameter identification.

[Method](#method) · [Setup](docs/SETUP.md) · [Experiments](EXPERIMENTS.md) · [Models](REVIEWER_README.md#reference-weights-and-full-experiments) · [References](#references)

[![GIFT architecture](assets/figures/gift_architecture.png)](docs/ALGORITHM.md)

[![W1 predictions and signed errors for trajectory 100005](assets/figures/w1_flow_prediction_confirmation_traj100005_recursive_keyframes_20261007.png)](EXPERIMENTS.md)

*Architecture and W1 predictions at t = 5, 6, 7 and 8 for confirmation trajectory 100005. Each image links to its detailed description.*

## Method

GIFT identifies a continuous-time generator from flow-field trajectories. For the vorticity field $\omega$, the true dynamics and the learned generator are written as

$$\frac{\mathrm{d}\omega}{\mathrm{d}t}=G^{\dagger}(\omega),\qquad G(\omega)=C+A(\omega)+Q(\omega,\omega).$$

Field tomography separates the bias, linear and quadratic components through their amplitude responses: $A(\lambda\omega)=\lambda A(\omega)$ and $Q(\lambda\omega,\lambda\omega)=\lambda^2Q(\omega,\omega)$. Learnable spectral filters represent the spatial operators. A high-frequency branch and recursive local correction complete the flow predictor, which advances the state using fourth-order Runge–Kutta integration. With the generator frozen, fitting physical equation terms to its channel outputs yields equation-parameter estimates. See the [algorithm description](docs/ALGORITHM.md).

## Results

The W1/W2 protocol uses 300 training trajectories and 76 observation nodes at intervals of 0.04 seconds; GIFT-Lite uses the first 50 training trajectories. Prediction results below use the 180-trajectory confirmation set.

- **W1 — flow prediction:** GIFT's mean relative L2 error over the 75 future nodes is **2.4687%**, with **3.8113%** at t = 8. GIFT-Lite reaches **3.6608% / 6.5787%**. The comparison includes FNO-2D, FNO-3D, U-NO, U-Net and ANIE.
- **W2 — cross-resolution prediction:** trained at N64, GIFT achieves mean relative L2 errors of **2.1098% / 2.0633% / 2.0610%** over the 25 future nodes at N64 / N96 / N128.
- **W3 — equation-parameter identification:** five methods are evaluated on 50 noise-free parameter combinations, using one trajectory per combination. [Per-case estimates, error summaries and measured costs](EXPERIMENTS.md) are available.

S1–S3 evaluate the high-frequency branch, recursive local correction and initial-condition distributions. Full results, training recipes and evaluation splits are recorded in [EXPERIMENTS.md](EXPERIMENTS.md).

## Quick start

The reference prediction environment uses Python 3.10.19, PyTorch 2.10.0 and CUDA 12.6. Run these commands from the repository root:

```shell
python -m venv ../gift-env
# Activate ../gift-env using your shell's activation command.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-reproduction.txt
python -m pip install -e .

export GIFT_DATA_ROOT=/absolute/path/GIFT-dataset
export GIFT_EXTERNAL_ROOT=/absolute/path/gift-external
export GIFT_ANIE_SOURCE_ROOT=/absolute/path/gift-external/anie
export GIFT_EXTERNAL_ADAPTER_ROOT=/absolute/path/gift-external-runtime

python -m scripts.verify_data --root "$GIFT_DATA_ROOT"
python -m scripts.prepare_external --root "$GIFT_EXTERNAL_ROOT"
```

Prepare the external methods and runtime interfaces outside GIFT using the [adaptation guide](docs/EXTERNAL_MANUAL_ADAPTATION.md). Source versions are recorded in [external_sources.json](external_sources.json) and [anie_sources.json](anie_sources.json). [SETUP.md](docs/SETUP.md) includes PowerShell commands and the separate PINN environment.

- **Flow prediction:** [W1/W2 training and reference-model evaluation](docs/CURRENT_REPRODUCTION.md).
- **Parameter identification:** [W3 data preparation, training and aggregation](docs/W3_REPRODUCTION.md).

## Data and models

The dataset has been uploaded to ScienceDB. Review access: [ScienceDB review link](https://www.scidb.cn/en/anonymous/YnVVUjNl). See the [dataset layout](docs/data_package/README.md) and [generation workflow](docs/DATA_GENERATION.md). Set `GIFT_DATA_ROOT` to the dataset directory.

[Reference-model access information](REVIEWER_README.md#reference-weights-and-full-experiments) for the seven prediction methods and two paired GIFT generators. File hashes are listed in [CURRENT_MODEL_BUNDLE.json](docs/CURRENT_MODEL_BUNDLE.json); activation and evaluation commands are in the [reproduction guide](docs/CURRENT_REPRODUCTION.md#reference-model-bundle).

## Documentation

- [Algorithm](docs/ALGORITHM.md) — generator structure and field tomography.
- [Experiments](EXPERIMENTS.md) — numerical results and configurations.
- [Training](docs/TRAINING_PROTOCOL.md) — data splits, budgets and checkpoints.
- [Models](docs/CHECKPOINTS.md) — reference weights and model identities.
- [Validation](docs/VALIDATION.md) — tests and numerical checks.

## References

External implementations and their associated publications:

- **FNO-2D / FNO-3D** — [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator). Li, Z. et al. [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895). ICLR (2021); Li, Z. et al. [Neural Operator: Graph Kernel Network for Partial Differential Equations](https://arxiv.org/abs/2003.03485). arXiv:2003.03485 (2020).
- **U-NO** — [ashiq24/UNO](https://github.com/ashiq24/UNO). Rahman, M. A., Ross, Z. E. & Azizzadenesheli, K. [U-NO: U-shaped Neural Operators](https://openreview.net/forum?id=j3oQF9coJd). TMLR (2023).
- **U-Net baseline** — [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets). Wang, R., Kashinath, K., Mustafa, M., Albert, A. & Yu, R. [Towards Physics-informed Deep Learning for Turbulent Flow Prediction](https://arxiv.org/abs/1911.08655). KDD (2020).
- **ANIE** — [emazap7/ANIE](https://github.com/emazap7/ANIE). Zappala, E. et al. [Learning integral operators via neural integral equations](https://doi.org/10.1038/s42256-024-00886-8). Nature Machine Intelligence **6**, 1046–1062 (2024).
- **Galerkin attention used by ANIE** — [scaomath/galerkin-transformer](https://github.com/scaomath/galerkin-transformer). Cao, S. [Choose a Transformer: Fourier or Galerkin](https://openreview.net/forum?id=ssohLcmn4-r). NeurIPS **34**, 24924–24940 (2021).
- **PINN-SR** — [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery). Chen, Z., Liu, Y. & Sun, H. [Physics-informed learning of governing equations from scarce data](https://doi.org/10.1038/s41467-021-26434-1). Nature Communications **12**, 6136 (2021). Raissi, M., Perdikaris, P. & Karniadakis, G. E. [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://doi.org/10.1016/j.jcp.2018.10.045). Journal of Computational Physics **378**, 686–707 (2019).
- **PDE-FIND** — [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND). Rudy, S. H., Brunton, S. L., Proctor, J. L. & Kutz, J. N. [Data-driven discovery of partial differential equations](https://doi.org/10.1126/sciadv.1602614). Science Advances **3**, e1602614 (2017).
- **TensorFlow optimizers** — [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/tree/v1.15.0). Abadi, M. et al. [TensorFlow: Large-scale machine learning on heterogeneous systems](https://www.tensorflow.org/about/bib) (2015).
