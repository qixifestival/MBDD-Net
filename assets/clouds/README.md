# Synthetic cloud assets

Place the 12 RGB PNG cloud textures used by the paper in this directory after the authors confirm that they may be redistributed publicly.

Expected filenames are `Cloud1.png` through `Cloud12.png`. During preprocessing, grayscale intensity is converted to an alpha channel in memory. The training pipeline selects one texture with nominal probability 0.3, crops 0.3--0.6 of its original height and width, resizes it to the optical patch, and applies a global alpha multiplier sampled from 0.7--1.0.

The assets were generated with Nano Banana 2 and are disclosed as generative-AI-assisted research material in the manuscript. They are intentionally not copied into this draft release folder.

