"""Prepare the Sentinel-1 ResNet-18 state dictionary used by MBDD-Net."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


REPO_ID = "BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0"
PREFIX = "model.vision_encoder."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Existing Hugging Face model directory. Downloads the upstream model when omitted.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    if source_dir is None:
        source_dir = Path(snapshot_download(repo_id=REPO_ID))

    source_file = source_dir / "model.safetensors"
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing upstream checkpoint: {source_file}")

    wrapped = load_file(str(source_file), device="cpu")
    converted = {
        key[len(PREFIX) :]: value
        for key, value in wrapped.items()
        if key.startswith(PREFIX)
    }
    if "conv1.weight" not in converted:
        raise RuntimeError("Unexpected upstream state-dict layout: conv1.weight was not found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print(f"Wrote {len(converted)} tensors to {args.output}")


if __name__ == "__main__":
    main()

