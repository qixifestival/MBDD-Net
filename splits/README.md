# Paper split manifests

- `train.txt`: 7,927 training patches
- `val.txt`: 633 validation patches
- `test.txt`: 1,277 test patches

Each line is a sample identifier without a filename suffix. The training and validation loaders resolve identifiers against the dataset root; the test loader resolves them against `DATA_ROOT/test`.

These lists support exact reuse of the archived paper split. They do not redistribute the underlying imagery.

`SHA256SUMS` records the checksum of each manifest so that later repository releases can detect accidental changes.
