# Data layout and access boundary

## Access classification

- **BRIGHT pre-event optical, post-event SAR, and damage labels:** reused public source; cite the BRIGHT dataset and its repository record.
- **Additional post-event very-high-resolution optical imagery:** third-party restricted; obtain it from the providers named in the manuscript and follow their terms.
- **Processed BrightEx collection:** not redistributed in this code repository because it contains provider-restricted imagery.
- **Split manifests:** included in `splits/` because they contain sample identifiers only.
- **Synthetic cloud assets:** author-generated with Nano Banana 2 but withheld from this draft repository until redistribution rights are confirmed.

## Expected directories and filenames

Training and validation data are read from the dataset root:

```text
DATA_ROOT/pre-event-opt/{sample_id}_pre_disaster.tif
DATA_ROOT/post-event-opt/{sample_id}_post_disaster.tif
DATA_ROOT/post-event-sar/{sample_id}_post_disaster.tif
DATA_ROOT/target/{sample_id}_building_damage.tif
```

Test data use the same four subdirectories below `DATA_ROOT/test/`.

The label remapping used by the code is:

| Original value | Localisation task | Severe-damage task |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 1 | 255 (ignore) |
| 3 | 1 | 1 |

## Split manifests

| File | Samples | Purpose |
|---|---:|---|
| `splits/train.txt` | 7,927 | training, including overlap-sampled training patches |
| `splits/val.txt` | 633 | checkpoint selection |
| `splits/test.txt` | 1,277 | final evaluation |

The manifests were extracted from the archived experiment dataset. They do not contain image pixels or provider-restricted data.

## Data citation

Before making the repository public, replace this section with the verified formal BRIGHT dataset citation used in the manuscript reference list and include its persistent identifier. Do not apply an open licence to third-party imagery.

