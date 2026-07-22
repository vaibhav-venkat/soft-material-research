# cluster_analysis

Structural crystal-cluster analysis for cylindrical big-Lx safetensor data.
The Zig library returns only one `A / SA` or `sqrt(A / SA)` sample per cluster;
Python owns replicate pooling, probability distributions, and plotting.

The numerical cluster kernel is intentionally a TDD placeholder. Build the
library and compile all tests with `zig build check`; `zig build test` is
expected to fail at the two structural-result tests until that kernel exists.

`vendor/kdtree` contains the C library from
<https://github.com/jtsiomb/kdtree> at commit
`384f9eba646018f99fe37c1f5d0c9e21a63d5057`, under its 3-clause BSD license.
