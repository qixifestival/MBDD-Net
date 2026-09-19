# Reproducibility notes

## Code snapshot

This draft repository was assembled from the local `gd_mbdd_fam_fixed` experiment tree. It uses the corrected FAM implementation and exposes deterministic seeded training entry points for paired Full/NoFAM experiments.

## Reported metrics and retraining

The manuscript's reported metrics are tied to archived checkpoints and the archived test protocol. A fresh training run can differ because of GPU kernels, dependency versions, pretrained-weight conversion, and stochastic optimisation even when the same seed is supplied.

Before public release, the authors should:

1. identify the exact checkpoint corresponding to every headline manuscript result;
2. record its SHA-256 checksum;
3. evaluate it with the release version of `src/test.py`;
4. archive the checkpoint and its `run_config.json` in a versioned GitHub Release or Zenodo record if redistribution is approved; and
5. add a result-to-checkpoint manifest to this repository.

Do not state that a fresh run is bitwise identical to the manuscript result unless that claim has been verified in a clean environment.

## Determinism

`src/train.py` seeds Python, NumPy, PyTorch, CUDA, DataLoader generators, and worker-local augmentation state. It also disables cuDNN benchmarking and requests deterministic cuDNN behaviour. These settings improve repeatability but do not remove every source of nondeterminism across hardware and software releases.

