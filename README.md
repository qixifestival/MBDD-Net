# MBDD-Net

Reference implementation of **MBDD-Net**, a localisation-guided multimodal network for severe post-earthquake building-damage segmentation using pre-event optical, post-event optical, and post-event SAR imagery.

This repository is a clean research release prepared from the code used for the IJRS manuscript. It includes the model, training and evaluation code, the paper split manifests, and scripts for the Full/FAM and NoFAM protocols. Provider-restricted imagery, generated cloud assets, trained checkpoints, and third-party pretrained weights are not bundled.

## Repository layout

```text
.
├── src/                    # model, data loader, losses, train.py, test.py
├── scripts/                # portable training and evaluation entry points
├── configs/                # paper protocol and data-layout documentation
├── splits/                 # train/validation/test sample identifiers
├── tools/                  # pretrained-weight preparation utility
├── tests/                  # FAM and auxiliary-loss schedule tests
├── assets/clouds/          # place the 12 cloud PNG assets here when authorised
└── docs/                   # data, weights, release, and licensing notes
```

## Data

The code expects the processed BrightEx directory layout below. The imagery is not included because the additional very-high-resolution optical data remain subject to the respective provider terms.

```text
DATA_ROOT/
├── pre-event-opt/
├── post-event-opt/
├── post-event-sar/
├── target/
└── test/
    ├── pre-event-opt/
    ├── post-event-opt/
    ├── post-event-sar/
    └── target/
```

Each modality uses the filename patterns documented in [`docs/DATA.md`](docs/DATA.md). The exact sample identifiers used in the paper are provided in [`splits/`](splits/).

## Environment

The manuscript records Python 3.9, Ubuntu 22.04 LTS, CUDA 11.8, and cuDNN 8.8. The exact original Python package lock file was not preserved, so [`requirements.txt`](requirements.txt) records compatible version floors rather than claiming an exact frozen environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build appropriate for your CUDA runtime first.
python -m pip install -r requirements.txt
```

Prepare the Sentinel-1 ResNet-18 checkpoint as described in [`docs/PRETRAINED_WEIGHTS.md`](docs/PRETRAINED_WEIGHTS.md). The optical ConvNeXt-Tiny weights are obtained by `timm` on first use.

## Training

Set the paths and run the paired protocols with the same seed:

```bash
export DATA_ROOT=/path/to/data-256Ex
export CLOUD_DIR=/path/to/cloud-assets
export SAR_PRETRAINED=/path/to/bigearthnet_s1_final.pth
export OUTPUT_ROOT=/path/to/outputs

SEED=42 bash scripts/train_full.sh
SEED=42 bash scripts/train_nofam.sh
```

The paper protocol uses a total sample budget of 500,000, batch size 16, 800 warm-up steps, learning rate `1e-4`, weight decay `0.01`, validation every 300 optimiser steps, and validation damage IoU for checkpoint selection. See [`configs/paper_protocol.yaml`](configs/paper_protocol.yaml).

## Evaluation

```bash
export DATA_ROOT=/path/to/data-256Ex
export MODEL_PATH=/path/to/best_model.pth
export OUTPUT_ROOT=/path/to/evaluation

# Natural test set only
bash scripts/evaluate.sh

# Natural test set plus the deterministic p=0.3 synthetic-cloud condition
export CLOUD_DIR=/path/to/cloud-assets
bash scripts/evaluate.sh
```

Add `DISABLE_FAM=1` when evaluating a NoFAM checkpoint. Evaluation writes `evaluation_scores.json` into each output directory.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover FAM window partition/reversal, batch independence, zero-flow identity, gradient finiteness, and the auxiliary-loss schedule.

## Reproducibility boundary

- The repository contains code and split identifiers, not provider-restricted imagery.
- The 12 Nano Banana 2 cloud assets are required for the paper's cloud-mixing protocol but are not included until redistribution permission is confirmed.
- Trained model weights are not included in the public repository and remain available from the corresponding author upon reasonable request.
- The original environment was not exported with `pip freeze`; do not describe the current requirements file as an exact lock file.
- The random seed controls Python, NumPy, PyTorch, CUDA, workers, and cuDNN deterministic settings, but exact bitwise identity across GPU/PyTorch versions is not guaranteed.

## Citation

Software citation metadata are provided in [`CITATION.cff`](CITATION.cff). The article DOI, volume, issue, and pages will be added after publication.

## Licence

The MBDD-Net release is distributed under the [MIT License](LICENSE). Third-party components and pretrained models remain subject to the notices in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
