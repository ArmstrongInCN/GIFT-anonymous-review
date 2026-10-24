# Experiment identifiers and algorithm names

Project code, entry points, experiment records and result manifests consistently use the paper identifiers W1, W2, W3, S1, S2 and S3. These identifiers directly designate experiments; no alternative numbering or conversion entry point is used.

| ID | Experiment | Current results | Main-text figure |
| --- | --- | --- | --- |
| W1 | Flow evolution prediction | [W1](../results/formal/W1_flow_prediction_20261007/) | Figure 2A error curves, Figure 2B prediction fields and signed residuals |
| W2 | Cross-resolution prediction | [W2](../results/formal/W2_cross_resolution_20261007/) | Figure 3A cross-grid curves, Figure 3B N128 field panels |
| W3 | Equation-parameter identification | [W3](../results/formal/W3_single_trajectory_20261010/): 50 noise-free parameter combinations, 250 slots, 750 readout rows | Per-case values and summary tables |
| S1 | High-frequency branch ablation | [S1](../results/formal/S1_high_frequency_branch/) | Supplementary high-frequency branch measurements |
| S2 | Recursive local correction ablation | [S2](../results/formal/S2_recursive_local_correction/) | Supplementary local-correction measurements |
| S3 | Initial-condition distribution comparison | [S3](../results/formal/S3_initial_distribution/) | Supplementary initial-condition distribution figures |

Supplementary experiment and figure identifiers are specified separately in the experiment index and figure captions.

W1 orders methods as GIFT, GIFT-Lite, FNO-2D, FNO-3D, U-NO, U-Net and ANIE; W2 uses the first six, excluding ANIE from cross-resolution evaluation. W3 uses GIFT, PDE-FIND, PDE-FIND-KC, PINN-SR and PINN-SR-KC.

W1/W2 GIFT-Lite uses the first 50 trajectories from the shared ordered N300 list, with source model key `GIFT-N50`; phase epochs remain unchanged while total updates decrease. S1–S3 GIFT-Lite uses 50 training trajectories under an independent N1000 protocol, separate schedules and validation-based model selection; weights from these two configurations are not interchangeable.

```shell
python -m scripts.run_experiment W3 --help
python -m scripts.run_experiment S3 --help
python -m scripts.verify_published_results --experiment W1 --result-dir results/formal/W1_flow_prediction_20261007
python -m scripts.verify_published_results --experiment W2 --result-dir results/formal/W2_cross_resolution_20261007
```

For current N300/single-state W1/W2 interfaces, see the [execution instructions](EXPERIMENT_EXECUTION.md); independent N1000 models, inputs and statistical scopes for S1–S3 are described in the [training protocol](TRAINING_PROTOCOL.md). The experiment index points directly to these actual computational entry points and scientific result packages.

Original source paths and file identities are preserved in an external provenance registry. The repository retains current experiment identities, scientific measurements and references to verifiable provenance records; see [provenance and integrity](SOURCE_PROVENANCE.md).
