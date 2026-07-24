from __future__ import annotations


def main() -> None:
    import hoomd  # type: ignore
    import jax

    if not hoomd.version.gpu_enabled:
        raise RuntimeError("HOOMD was not built with GPU support")
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("JAX did not discover a CUDA GPU")
    print(
        f"HOOMD {hoomd.version.version} platform={hoomd.version.gpu_platform}; "
        f"JAX GPUs={devices}"
    )


if __name__ == "__main__":
    main()
