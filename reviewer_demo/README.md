# GIFT: a small CPU software demonstration

This anonymous package contains four original GIFT numerical API modules, a
small simulated dataset and a bounded demonstration. The original modules are
copied without numerical changes; their SHA-256 identities are listed in
`SOURCE_INTEGRITY.json`. The software is licensed under MIT; see `LICENSE`.

The demonstration exercises the constant, linear and quadratic generator
components, gradient updates, model saving/loading and RK4 integration. Its toy
objective is a linear-decay derivative on four smooth 8 × 8 periodic fields. It
does not run the manuscript's full training controller, Navier–Stokes benchmarks
or equation-parameter recovery experiments, and its loss is not a paper result.

## System requirements and tested environment

The measured environment was Windows 11 (`Windows-11-10.0.26200-SP0`), AMD64,
Python 3.12.14, NumPy 1.26.4 and PyTorch 2.10.0+cu126. The demonstration ran on
an Intel Core i9-12950HX CPU with one Torch thread. It uses no GPU computation
and needs no special hardware. The measured system has approximately 127.7 GiB
of physical RAM; this is an observed machine specification, not a minimum
requirement. No minimum RAM benchmark was performed. Linux and macOS were not
tested in this demonstration.

Only NumPy and PyTorch are needed. The supplied core source is loaded directly
from `src/`; a separate editable installation of the full project is unnecessary.

## Installation

Use Python 3.12 in a suitable existing environment, or create a virtual
environment and install the two dependencies:

```shell
python -m venv gift-demo-env
# Activate gift-demo-env using your shell's activation command.
python -m pip install -r requirements-demo.txt
```

Choose a PyTorch distribution compatible with your platform. The recorded test
used an already installed CUDA-enabled build on CPU; a fresh CPU-wheel
installation was not tested. Installation was not independently timed. Its
duration depends on the chosen wheel, network connection and existing packages;
the measured runtime below excludes dependency installation.

## Run the supplied demonstration

From this directory, run:

```shell
python -B smoke_demo.py --output demo_run
```

Use a new output directory for every invocation. Existing output paths are
refused. The command loads `demo_data.npz`, initializes a rank-one GIFT generator
with spectral cutoff one, performs 30 Adam updates, verifies the component
amplitude responses, saves/reloads the model and integrates four RK4 steps.

Three fresh process launches on the recorded machine completed successfully in
approximately **2.9–3.1 seconds**, with a median of approximately **2.9 seconds**.
The process duration includes Python startup, imports, numerical work and output
writing. This is a small demonstration measurement on one machine; full paper
experiments have separate budgets and costs. `MEASURED_ENVIRONMENT.json` records
the actual timings and environment.

## Expected output

The console reports `"status": "passed"`. The new output directory contains:

- `summary.json`: measured environment, configuration, losses, checks and hashes.
- `smoke_model.pt`: this toy run's generated checkpoint, not a pretrained paper model.
- `smoke_arrays.npz`: derivative predictions `[4,8,8]`, a short trajectory
  `[5,8,8]` and 31 loss values including the post-update loss.

On the measured environment the initial MSE was approximately `0.0177653` and
the final MSE approximately `2.29971e-5`. The constant, linear and quadratic
components obeyed their amplitude factors of 1, 2 and 4 for a doubled input;
their maximum absolute discrepancies were zero in the recorded run. Reloading
the saved checkpoint reproduced that run's prediction exactly, and all saved
numerical outputs were finite. These are interface checks on the supplied toy
data, not a physical coefficient-identification accuracy claim.

`expected_output/` contains the recorded first successful final run. Small
floating-point differences can occur on other numerical environments. New
output-file hashes and timings need not equal those supplied; the driver checks
finite outputs, decreasing toy loss, component relations and equality between
its own pre-save and reloaded predictions.

## Run the small API example on your own data

Create an NPZ containing `state` and `time_derivative`, both finite float32
arrays of shape `[batch,8,8]` with at least one field, then run:

```shell
python -B smoke_demo.py --data my_fields.npz --output my_run
```

The spatial grid is a periodic square with domain length `2*pi`; the two trailing
axes are `[y,x]`. `time_derivative` is the supervised target for this bounded API
example. Supply consistent physical units and a target suitable for the fixed
cutoff-one model. The demonstration does not estimate derivatives from a time
series and does not adapt its fixed learning rate or budget to arbitrary data.
Its decreasing-loss guard may reject data requiring another configuration.

For manuscript experiments, the complete GIFT source and current protocol
documentation must also be supplied through the anonymous code-access route.
The full project accepts HDF5 trajectories with axes `[trajectory,time,y,x]`, a
physical time vector, trajectory IDs and the declared dataset roles. Its current
protocol documentation is `docs/CURRENT_REPRODUCTION.md` (W1/W2),
`docs/W3_REPRODUCTION.md` (W3), `docs/DATA_GENERATION.md` (regeneration) and
`docs/data_package/DATA_DICTIONARY.md` (input schema). Those full experiment
instructions and third-party dependencies are outside the scope of this small
core-API package. External algorithms remain subject to their own licenses and
setup procedures.

## Interpretation and testing scope

This package was executed by the preparing workflow on the recorded machine.
No unfamiliar-colleague installation test was performed. No full-suite pass
count, formal training run or paper result is certified by this demonstration.
No author-account repository URL, dataset DOI, machine hostname or local user
directory is needed to execute the supplied package.
