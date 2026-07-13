from __future__ import annotations

from pathlib import Path
import tempfile

import hoomd
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.flax import load_file, save_file


def main() -> None:
    version_module = getattr(hoomd, "version", None)
    print(f"HOOMD version: {getattr(version_module, 'version', 'unknown')}")
    if not hoomd.device.GPU.is_available():
        reasons = hoomd.device.GPU.get_unavailable_device_reasons()
        for reason in reasons:
            print(f"HOOMD GPU unavailable: {reason}")
        raise SystemExit("HOOMD GPU support is unavailable")
    print(f"HOOMD devices: {hoomd.device.GPU.get_available_devices()}")

    devices = jax.devices()
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {devices}")
    if not any(device.platform == "gpu" for device in devices):
        raise SystemExit("JAX did not discover a CUDA GPU")

    with tempfile.TemporaryDirectory(prefix="big-lx-safetensors-") as directory:
        path = Path(directory) / "roundtrip.safetensors"
        expected = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
        save_file({"values": expected}, path)
        actual = load_file(path)["values"]
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise SystemExit("safetensors JAX round trip failed")
    print("safetensors JAX round trip: ok")


if __name__ == "__main__":
    main()
