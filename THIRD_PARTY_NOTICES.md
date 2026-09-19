# Third-party notices

## Lovasz loss implementation

`src/calc_utils/lovasz_loss.py` identifies itself as the PyTorch Lovasz-Softmax/Jaccard-hinge implementation by Maxim Berman (2018), ESAT-PSI KU Leuven, under the MIT License. Its upstream licence is preserved in `third_party/LovaszSoftmax_LICENSE`; upstream project: <https://github.com/bermanmaxim/LovaszSoftmax>.

## BigEarthNet Sentinel-1 pretrained model

The SAR initialisation is derived from `BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0`. Its local model card declares the MIT License and requests citation of the reBEN and ConfigILM publications. The weights are downloaded from the upstream source and are not redistributed here.

## Python dependencies

PyTorch, torchvision, timm, Albumentations, OpenCV, tifffile, ttach, Pillow, NumPy, tqdm, torchinfo, TensorBoard, Hugging Face Hub, and safetensors remain subject to their respective licences.

## Dataset and imagery

No third-party imagery is included. Users must obtain BRIGHT and provider-controlled optical imagery under the applicable source terms.
