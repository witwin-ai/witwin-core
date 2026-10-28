# Fixed structure assignment controls

Trainable priority/enabled inputs now reject before Python integer/boolean conversion. Material and geometry Tensor gradients remain live. Five new contracts reproduce four original failures and retain a fixed-control geometry/material-gradient positive control.

The proposed core package has 155 passing CPU tests. An initial all-tests run retained four existing GPU skips and was correctly rejected as incomplete. The separate CPU run explicitly deselected those four known GPU contracts. Root then executed all four original CUDA contracts on the identical proposed core bytes, alongside three Maxwell FP64 public tests: all seven passed, zero skipped, with unchanged source and exact binary provenance. Both the full seven-test XML and source manifest are retained here; only four of its nodes are core contracts.

This commit changes the core checkout. It does not install the wheel while the source-bound maintained baseline still depends on the installed predecessor. The final integrated source requires a fresh maintained regression.
