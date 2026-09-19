# Pretrained weights

## Optical encoder

The optical branch uses `convnext_tiny.fb_in22k_ft_in1k_384` from `timm`. With internet access, `timm` downloads the corresponding public weights on first use.

## SAR encoder

The paper protocol initialises the SAR branch from the MIT-licensed model card **BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0**, a ResNet-18 pretrained on BigEarthNet v2.0 Sentinel-1 bands.

Prepare the checkpoint expected by this repository:

```bash
python tools/prepare_sar_checkpoint.py \
  --output pretrained/bigearthnet_s1_final.pth
```

Alternatively, download the Hugging Face model folder yourself and pass it with `--source-dir`.

The conversion utility strips the model-card wrapper prefix (`model.vision_encoder.`) and writes a plain torchvision-compatible state dictionary. Fully connected classifier weights are retained in the converted file but ignored by `SarEncoder`.

Do not commit downloaded or converted weights to Git. The repository `.gitignore` excludes `.pth` and `.safetensors` files.

Please cite the reBEN and ConfigILM publications requested by the upstream model card when this pretrained model is used.

