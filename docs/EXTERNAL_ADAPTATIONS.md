# External algorithm sources and necessary adaptations

**The current working tree retains only external algorithm provenance, written adaptation steps and generic interfaces.** This document and the [manual adaptation instructions](EXTERNAL_MANUAL_ADAPTATION.md) record official repositories, pinned versions, required operations and external interfaces. Users must download, adapt and retain external algorithms outside the GIFT repository folder; Git ignore rules do not permit storing them inside it. The [new model release](../REVIEWER_README.md#reference-weights-and-full-experiments) removes source metadata from public copies and supersedes the old package; old attachments remain, and Git history and old tags have not been rewritten. See the [external algorithm boundary policy](EXTERNAL_REPOSITORY_POLICY.md).

GIFT's `adapters/` only connects runtime interfaces prepared externally by users. `GIFT_EXTERNAL_ADAPTER_ROOT` points to a root containing external `adapters/` modules and defaults to the adjacent `gift-external-runtime/`; `GIFT_EXTERNAL_ROOT` points to the parent of official checkouts; `GIFT_ANIE_SOURCE_ROOT` points to the ANIE checkout. Original official checkouts remain read-only; necessary adaptations are implemented and retained externally. These local dependencies are not part of the GIFT Git repository and are not distributed with it. New users must provide interfaces by following the written steps; download-only tools do not adapt algorithms.

## Official sources

| Method | Official repository | Pinned commit |
| --- | --- | --- |
| FNO-2D / FNO-3D | [NeuralOperator](https://github.com/neuraloperator/neuraloperator) | `01d2aeca407f85ce1c1e78b313edb749973fc051` |
| U-NO | [UNO](https://github.com/ashiq24/UNO) | `19462d82729ef64ef7b9e97056ddcaaf3044ad47` |
| U-Net | [Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | `229da3e01acefa6f4dd6c77244da5015c9b965af` |
| ANIE | [ANIE](https://github.com/emazap7/ANIE) | `2071ad709b260a6c305450affcf7168fd825f692` |
| PDE-FIND | [PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | `86911349b9fac82996e35465a13d06205730cc05` |
| PINN-SR | [EQDiscovery](https://github.com/isds-neu/EQDiscovery) | `9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496` |
| Native PINN optimizers | [TensorFlow](https://github.com/tensorflow/tensorflow), tag `v1.15.0` | `590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b` |

Repository files `external_sources.json` and `anie_sources.json` retain only provenance, versions, selected files and written operation identifiers. Pinned commits and hashes record provenance but do not replace upstream licenses. No explicit license was found for the pinned ANIE version, and additional written authorization has not been obtained; written descriptions, paper citations and public-copy metadata cleanup cannot substitute for authorization. Full license status and original research citations are in [README](../README.md); installation, independent commands and directory settings are in [SETUP.md](SETUP.md).

## Current single-initial-state W1/W2 profile

The current [fixed protocol](W1_W2_PROTOCOL.md) uses `single_native76_n300`, with one fixed-seed model per method and no hyperparameter search or multiple seeds. Source adaptation and experiment settings must be distinguished: using a public architecture does not mean running the authors' original data and training scripts, nor equal supervision horizons or budgets across methods.

| Method | Necessary external operations | Preserved computations and current task settings |
| --- | --- | --- |
| FNO-2D / FNO-3D | Isolate module-level examples; change inputs to 3/4 at the original unique fc0, comprising one field and two/three coordinates; provide original utility classes. | Retain spectral layers, forward computation, initialization, original LpLoss and normalizer. FNO-2D closed-loop curriculum is part of the training protocol; FNO-3D outputs 75 future points. W2 retains 75 joint queries, Fourier-interpolates the frozen spatial normalizer and selects the first 25 outputs, without fitting target-grid statistics. |
| U-NO | Provide original model/integral operators/Adam/LpLoss; do not reset training seeds on import; use original constructor arguments in_width=5, width=32, pad=0, factor=.75. | One field plus original four coordinates; retain spectral computations, interpolation gradients and BatchNorm. The protocol declares current full closed-loop supervision for 20 steps, non-reentrant activation checkpoints, per-epoch scheduling and terminal-state selection. |
| U-Net | Provide U_net from original Baselines/U-net.py; use original constructor arguments for input/output channels 1/1, kernel_size=3, dropout_rate=0. | Retain convolution/BN/skip connections and initialization. Four-step training and the optimizer are project recipes; N64/N96/N128 use one fully convolutional model, whose pixel-kernel physical scale varies with N. |
| ANIE | Provide original NS network and integral solver; align spatial and temporal coordinates with states in space-major/time-minor order; functional attention dropout follows train/eval, retaining probability 0.5 for training. | Current aligned_ns_single_v1 uses the original c=0/y_init branch; retain original global decoder, integral normalization, iteration rules and autograd. Training/evaluation both use 76 queries; micro6/effective32 organizes GPU memory. W1 only. |

ANIE coordinate and evaluation-dropout corrections change the uncorrected function, so current results are described as experiments using the official notebook architecture with explicit corrections, not a verbatim reproduction of the authors' implementation. Repository-boundary reorganization only relocates existing local dependencies; it does not introduce these numerical changes again or switch old weights to another profile.

Parameter sources, supervision coverage, losses, actual budgets and numerical environments are described in the [single fixed-recipe implementation](W1_W2_PROTOCOL.md), [additional baseline implementation](W1_W2_PROTOCOL.md), [W2 implementation](W1_W2_PROTOCOL.md) and [result interpretation scope](W1_W2_PROTOCOL.md). ANIE corrections and current configuration are documented in the [ANIE implementation notes](ANIE_ADAPTATION.md). Actual model and weight identities are in [CHECKPOINTS.md](CHECKPOINTS.md), and costs in [PERFORMANCE.md](PERFORMANCE.md).

## Independent 46-frame and equation-identification protocols

Historical 46-frame prediction uses independent input projections: 48/49 for FNO-2D/FNO-3D, 50 for U-NO and 46 input channels for U-Net. Single-state weights and 46-frame-history weights are not interchangeable. ANIE profiles must likewise load the coordinate and free-term branches recorded with their weights; do not silently upgrade the computation path of an old model.

PDE-FIND and PINN-SR serve independent W3 and do not change with W1/W2. Their external operations are:

| Method | Operation scope |
| --- | --- |
| PDE-FIND | Provide original library-construction/regression functions; replace the unique empty-index comparison with an explicit length check for every NumPy version (1.26 already raises an error). Execute final least squares for nonempty indices as originally intended; see manual notes for the old single-element-array misclassification. Retain seed0 samples and order; local RNG and membership lookup only isolate/accelerate sampling. Periodic velocity, SVD and derivative processing follow the declared protocol. |
| PINN-SR network | Provide original PhysicsInformedNN; TF1 reference variables ensure original SciPy candidate feeds enter the objective; known library 4/open library 90. Retain Xavier initialization, network, losses and full populations. Default native FP32; optional equivalent coordinate-derivative/residual evaluation and XLA require complete paired accuracy acceptance. |
| PINN-SR sparse regression | Use original TrainSTRidge/STRidge; retain original seed0 random training-row order and float32 inherited coefficients, rcond=−1 and the unique empty-index length check. Assign explicitly and immediately, with side-effect-free reads; ADO fixes regression coefficients and updates only the network, while post-training fixes final STRidge support and jointly refines. |
| TensorFlow optimizers | Use pinned official NAdam and ScipyOptimizerInterface without rewriting kernels. Retain the same NAdam slots/beta; record actual SciPy termination and objective changes. Save regression support together with model/optimizer/FSM; redo interrupted internal L-BFGS from phase boundaries without claiming internal solver-state continuation. |

Each of the two current W3 PINN methods completed 50 independent trainings, producing 100 formal terminal models under `paper_ado_v1` from paper Methods equations (6b)/(8), retaining fixed-regression-coefficient ADO and post-training with final regression support. TF1 reference-variable, pure-read and original regression-row-order operations are described in the [manual adaptation instructions](EXTERNAL_MANUAL_ADAPTATION.md#pinn-sr-and-tensorflow-optimizers); complete data and actual results are in the [W3 protocol](W3_PROTOCOL.md).

Target files, construction order, tensor axes, coordinate rules, library column order, interface names and checks are detailed in the [manual adaptation instructions](EXTERNAL_MANUAL_ADAPTATION.md). Small-scale loading, forward or continuation checks validate only their specific computational conditions and cannot replace complete scientific experiments.

## Acknowledgments

We thank the authors and contributors of NeuralOperator, U-NO, Turbulent-Flow-Nets, ANIE, EQDiscovery, PDE-FIND, TensorFlow and SciPy. Cite the research actually used and comply with upstream terms; the GIFT license grants no additional rights to external code.
