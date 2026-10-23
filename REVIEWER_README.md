# GIFT anonymous source snapshot

This package contains the complete original Python source, dependency files,
experiment configuration and reproduction documentation for GIFT. Python package
and module names are preserved. The project MIT licence and all third-party
source attributions, fixed versions and licensing information are retained.

## Start here

For a small bounded CPU example that does not require the full experiment data,
see `reviewer_demo/README.md`. The independent demonstration is supplied in the
same package; it is not a reproduction of the manuscript's full experiments.

For the full project, follow `docs/SETUP.md`. Scoped source-package installation timings are recorded in `REVIEW_INSTALLATION.md`. Current protocols are documented in
`docs/CURRENT_REPRODUCTION.md` (W1/W2), `docs/W3_REPRODUCTION.md` (W3),
`docs/S3_PROTOCOL.md` and `docs/DATA_GENERATION.md`. Data fields are described in
`docs/data_package/DATA_DICTIONARY.md`.

The data review entry point supplied with the submission is
https://www.scidb.cn/en/anonymous/YnVVUjNl . Set `GIFT_DATA_ROOT` to the extracted
dataset root containing `dataset.json`. This source snapshot does not contain the
large archived dataset, generated result arrays or full trained model weights.

## Per-case source data

`source_data/Supplementary_Table_S5.csv` contains the complete W3 per-case results
corresponding to Supplementary Table S5: 50 equation-parameter cases x 5
identification methods x 3 parameters, giving 750 rows. The table includes the
true parameter values, estimates, absolute percentage errors, status and neutral
training-host groups. These are the archived results used in the manuscript,
not a new experiment. All scientific fields are preserved; the host groups use
the neutral labels `host_A` and `host_B` for review.

## Reference weights and full experiments

This source-only review snapshot omits trained weights, generated training
histories and large result exports. Catalogue files retain architecture/protocol and
checksum information for inspection. References in historical documentation to
`results/`, training-cost receipts and completion records describe the separate
scientific archive and are not claims that those generated files are bundled.
The data archive and this source package serve different purposes.

For independent reproduction, prepare the documented data and train from the
current protocol entry points. Reference-weight activation commands require a
separately supplied compatible model bundle; this package contains no model
download endpoint. The small CPU demonstration uses its own toy data and performs
its own short training run. Reproducing external baselines additionally requires
the fixed upstream sources and manual adaptations described in
`docs/EXTERNAL_MANUAL_ADAPTATION.md`; upstream code is not copied into this package.

## Portable paths in this review snapshot

The three source files that previously contained experiment-machine absolute
paths use existing environment variables or sibling-directory defaults here.
Set `GIFT_DATA_ROOT`, `GIFT_EXTERNAL_ROOT` and `GIFT_EXTERNAL_ADAPTER_ROOT` as in
the current reproduction guides. The legacy noisy-W3 queue also accepts
`GIFT_W3_CLEAN_PATH`, `GIFT_TORCH_PYTHON` and `GIFT_PINN_PYTHON`. Select the
documented separate TensorFlow/PINN Python interpreter for `GIFT_PINN_PYTHON`;
its default is the invoking interpreter and does not create a new environment.
No training budget, seed, data split, solver, loss or model equation is changed.

`PACKAGE_MANIFEST.json` lists the supplied files and their SHA-256 values.
Private source-to-snapshot audit records and the original version-control
history are not included in this review package.
