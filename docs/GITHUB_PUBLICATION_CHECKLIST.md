# GitHub publication checklist

## Required before making the repository public

- [ ] All five authors approve public code release and repository name.
- [x] Select and add the repository-wide MIT `LICENSE` file.
- [x] Preserve the complete upstream MIT notice/link for `lovasz_loss.py`.
- [ ] Confirm whether the 12 generated cloud PNG files may be redistributed.
- [ ] Decide whether trained MBDD-Net weights may be released; if yes, place them in a versioned GitHub Release or Zenodo record rather than Git history.
- [ ] Export `pip freeze` or `conda env export` from the actual training environment if it is still accessible.
- [ ] Test `tools/prepare_sar_checkpoint.py` in a clean Python 3.9 environment.
- [ ] Run the unit tests and at least one training/evaluation smoke test on Linux with a CUDA GPU.
- [ ] Confirm the split manifests contain identifiers only and no provider-restricted image content.
- [ ] Create the GitHub repository without importing the old Gitee `.git` history.
- [ ] Enable GitHub secret scanning and push protection.
- [ ] Create a versioned release (for example, `v0.1.0`).
- [ ] Archive that release in Zenodo to obtain a DOI if long-term code citation is desired.
- [ ] Replace `CITATION.cff.template` with `CITATION.cff` after bibliographic metadata are final.
- [ ] Update the manuscript Code/Data Availability wording only after the public URL or DOI resolves.

## Files intentionally excluded

- BrightEx imagery and labels
- cloud PNG assets pending rights confirmation
- model checkpoints and pretrained weights
- old Gitee metadata and commit history
- server logs, TensorBoard runs, predictions, and evaluation maps
- `__pycache__`, IDE settings, local absolute paths, and AutoDL paths

## Suggested repository settings

- Repository visibility: private during final verification, public only after author approval
- Default branch: `main`
- Issues: optional
- Discussions: optional
- Releases: enabled
- Preserve a tagged release corresponding to the submitted manuscript
