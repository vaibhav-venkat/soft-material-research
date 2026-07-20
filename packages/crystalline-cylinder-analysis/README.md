# Crystalline-cylinder analysis CLI

The numerical tensor backend is Tenferro CUDA. Host-only streaming, periodic
unwrapping, plotting, and KD-tree work use the Rayon pool selected by
`--threads`; tensor reductions and linear algebra use the device selected by
`--device cuda:N`.

On the Linux CUDA host, configure the CUDA and Tenferro library search paths as
needed by the installation:

```bash
export CUDA_PATH=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH"
export TENFERRO_CUTENSOR_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
export TENFERRO_CUSOLVER_PATH="$CUDA_PATH/lib64/libcusolver.so.12"
export TENFERRO_CUBLAS_PATH="$CUDA_PATH/lib64/libcublas.so.12"
export CUBECL_DEBUG_LOG=0
```

Then run, for example:

```bash
pixi run cargo run --release -p crystalline-cylinder-analysis-cli -- \
  --device cuda:0 \
  --threads 8 \
  --input-dir hexatic/big_lx \
  --input-dir hexatic/confinement_comparison \
  --output-dir cluster_analysis_output \
  clusters \
  --frame-start 600 \
  --frame-stop 1000 \
  --snapshot-frames 700
```

The `clusters` command currently selects cylindrical cases only. Its structural
and coherent-motion distributions use the equivalent-circle circumference
`C_cluster = pi D sqrt(N)` divided by the equivalent circumference of the full
cylinder surface, `C_surface = 2 sqrt(pi Lx C)`. The resulting horizontal
coordinate is `C_cluster/C_surface`.
Static cluster views assign an overlapping particle to the cluster with the
larger equivalent circumference (and use the smaller local cluster ID to break
exact ties).
`--frame-start` is inclusive and `--frame-stop` is exclusive. Omitting the stop
uses the trajectory length; snapshots outside the selected interval are skipped.
The command writes log-binned cluster-count distributions and linear
circumference-weighted distributions for both structural and coherent-motion
clusters. Each cluster contributes in proportion to its equivalent
circumference.
It also writes a structural–motion overlap distribution. Each same-frame
structural/motion cluster pair is located by particle-membership Jaccard overlap
and weighted by the equivalent circumference of the particles shared by that
pair.

Tenferro uses explicit host/device transfers. This CLI uploads graph inputs and
downloads final host-visible results; unsupported CUDA operations fail instead
of silently executing on the CPU.

See the official Tenferro [devices and GPU guide](https://tensor4all.org/tenferro-rs/guides/devices-and-gpu.html)
for current CUDA requirements and provider behavior.
