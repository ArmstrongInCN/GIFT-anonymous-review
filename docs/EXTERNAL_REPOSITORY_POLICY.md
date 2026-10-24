# External algorithm boundaries and equivalence checks

Current GIFT working files must not contain external algorithm source, copied algorithm implementations, executable source patches or automatic source rewriters. Official repositories, pinned versions and necessary manual adaptations are documented in the [external algorithm adaptation instructions](EXTERNAL_MANUAL_ADAPTATION.md). `adapters/__init__.py` provides only generic path resolution, module connections and release-metadata handling; external networks, solvers, source loading and adaptation implementations must reside outside the repository. This requirement also covers source text inside checkpoints, NPZ, JSON and ZIP files; checking only `.py` files is insufficient.

At runtime, `GIFT_EXTERNAL_ADAPTER_ROOT` selects the user-prepared external interface directory, defaulting to adjacent `gift-external-runtime/`; neither this directory nor its resolved location may be inside GIFT. New GIFT users must download official code, adapt it according to the written instructions and provide interfaces outside the repository. Training entry points, evaluation protocols and GIFT's own algorithms remain in GIFT.

## Current release boundaries

| Object | Current handling and limitations |
| --- | --- |
| Current source working tree | Retains repository references, pinned-version metadata, written adaptation steps and generic interfaces; users prepare concrete external implementations outside GIFT. |
| New public model copies | The [source-metadata-cleaned release](../REVIEWER_README.md#reference-weights-and-full-experiments) provides `GIFT-current-models-20261020-source-free.zip`, removing source metadata and verifying unchanged numerical values tensor by tensor. The [manifest](CURRENT_MODEL_BUNDLE.json) records direct download addresses and identities. |
| Original weights and experiment records | Remain read-only; public-copy hashes do not replace original measurement identities, and experimental values remain unchanged. |
| Old online model attachment | The ANIE checkpoint in `GIFT-current-models-20261020.zip` still contains four `before`/`after` source strings totaling 441 characters; the new release supersedes that package, while the old attachment remains as a provenance record and has not been physically deleted. |
| Git history and old tags | Old `main` commits and tag `reference-checkpoints-20260913` still expose seven old adapter files; history has not been rewritten and old tags have not moved. |

A new ordinary commit records only a new tree; it cannot remove code from old commits, old tags or published attachments. A passing current-file check therefore applies only to those working files or the specified attachment and cannot establish that all Git history or old releases are free of external code. Removing old code from all historical versions requires a separate history-handling plan, execution and review.

No explicit license was found at ANIE commit `2071ad709b260a6c305450affcf7168fd825f692`, and no additional written authorization is currently available. Source attribution and metadata cleanup address disclosure and content boundaries; they do not grant use, modification or distribution rights. Pinned sources appear in [external_sources.json](../external_sources.json) and [anie_sources.json](../anie_sources.json), with original research citations in [README](../README.md#references).

## Public model copies

`scripts.package_current_models export` checks and removes source metadata from checkpoints intended for publication by default, writing only to a new output directory. Files without fields requiring cleanup are copied byte for byte; cleaned files are serialized again and checked tensor by tensor for dtype, shape and values. Model entries in `ORIGINS.json` separately record source-file and public-copy sizes and SHA256, `removed_source_fields` and `tensor_values_unchanged`; top-level `tensor_values_changed` is false and records whether source metadata was removed. Tensor equivalence is actually checked during export; `verify` checks nested public-copy contents and manifest bindings without requiring recipients to possess source files from the original training machine.

Original-file hashes, original measurement hashes and public-copy hashes are three potentially different evidence identities. An ANIE public copy with cleaned metadata cannot be described as byte-identical to the original; its model tensors remain bitwise identical, and the cleanup grants no additional license. The public package contains neither training data nor complete optimizer recovery logs; see [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md) for use.

## Equivalence evidence for the 2026-10-07 migration

Post-migration execution was checked with formal weights saved before migration, fixed inputs and independent CPU processes:

| Validation object | Scope | Result |
| --- | --- | --- |
| FNO2D, FNO3D, ANIE | Full 76-point predictions with the same formal weights | Bitwise-identical outputs, maximum absolute difference 0 |
| U-NO, U-Net | Two-step predictions with the same formal weights | Bitwise-identical outputs, maximum absolute difference 0 |
| Five external networks | Parameter counts, state tensors, provenance identities and checkpoint SHA256 | Identical |
| 10 relocated modules | AST comparison of functions, classes and algorithm constants | Identical; only project-root and source-file locations adjusted |
| Original ANIE EarlyStopping | Sequence of 1,011 losses including improvements, ties and patience=1000 | Identical best, counter and stopping decisions |
| Formal weight loading | Original model specifications, seed, epoch and metadata checks | Passed |
| 465 scientific result files | Arrays, numerical records, weights, figures and reference hashes | 431 files byte-identical; 34 JSON files only had source-fragment metadata removed and corresponding hashes/sizes updated |
| 14 formal result packages | Release manifests, file hashes and result-package references | Passed |

Source-patch text was removed from result JSONs covered by these migration checks. Changes were limited to patch-description `before`/`after` fields and hash/size references affected by their byte changes; experimental metrics, training settings, prediction arrays and figures were neither recomputed nor modified. That check did not cover checkpoint strings inside the later model ZIP or remove Git history. Original metadata, original run records and `external_boundary_20261007` migration evidence remain outside the repository.

These checks support that this directory reorganization preserved the verified computations and existing experimental results. They include fixed-input CPU tests and algorithm-structure comparisons, without repeating full GPU training or establishing bitwise determinism across all devices and inputs.

## Maintenance checks

From the GIFT root, run `python -B -m scripts.audit_repository` to check boundaries of current working files proposed for release. Ignored files and caches require separate physical-file scans; Git file selection alone is insufficient. Specify attachments separately so that scanning current text does not miss PT or NPZ files inside ZIPs:

```shell
python -B -m scripts.audit_repository --public-file /exports/GIFT-current-models-20261020-source-free.zip
python -B -m scripts.audit_repository --history-risk-ref reference-checkpoints-20260913
```

The second command reports risks in the specified historical reference read-only and does not clean history. A comprehensive review must cover all relevant branches and tags; one reference cannot represent unchecked history. Checks do not stage, commit, upload or delete files. After old adapters leave the working tree, their removal records must accompany generic interfaces and documentation in the subsequent proposed release commit; committing only new documentation cannot change the old file selection.
