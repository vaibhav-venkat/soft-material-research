from __future__ import annotations

from pathlib import Path
import tempfile
import importlib

import hoomd
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.flax import load_file, save_file


def main() -> None:
    print(f"HOOMD version: {hoomd.version.version}")
    if hoomd.version.version != "7.0.1":
        raise SystemExit(f"Expected HOOMD 7.0.1, found {hoomd.version.version}")
    if not hoomd.device.GPU.is_available():
        for reason in hoomd.device.GPU.get_unavailable_device_reasons():
            print(f"HOOMD GPU unavailable: {reason}")
        raise SystemExit("HOOMD GPU support is unavailable")
    print(f"HOOMD devices: {hoomd.device.GPU.get_available_devices()}")
    devices = jax.devices()
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {devices}")
    if not any(device.platform == "gpu" for device in devices):
        raise SystemExit("JAX did not discover a CUDA GPU")
    # Confirm that the requested 7.0.1 constrained GPU classes exist.
    required = (
        "TwoStepRATTLEBDCylinderGPU",
        "ActiveForceConstraintComputeCylinderGPU",
    )
    _md = importlib.import_module("hoomd.md._md")

    missing = [name for name in required if not hasattr(_md, name)]
    if missing:
        raise SystemExit(f"Missing HOOMD GPU constraint classes: {missing}")
    with tempfile.TemporaryDirectory(prefix="confinement-safetensors-") as directory:
        path = Path(directory) / "roundtrip.safetensors"
        expected = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
        save_file({"values": expected}, path)
        actual = load_file(path)["values"]
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise SystemExit("safetensors JAX round trip failed")
    print("confinement GPU prerequisites: ok")


if __name__ == "__main__":
    main()
