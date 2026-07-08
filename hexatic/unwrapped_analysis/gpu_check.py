from __future__ import annotations

import hoomd


def main() -> None:
    version_module = getattr(hoomd, "version", None)
    print(f"HOOMD version: {getattr(version_module, 'version', 'unknown')}")
    if not hoomd.device.GPU.is_available():
        reasons = hoomd.device.GPU.get_unavailable_device_reasons()
        if reasons:
            print("GPU unavailable reasons:")
            for reason in reasons:
                print(f"- {reason}")
        raise SystemExit(
            "HOOMD GPU support is not available in this environment. "
            "Install a GPU-enabled HOOMD build before running the sweep."
        )

    devices = hoomd.device.GPU.get_available_devices()
    print("Available HOOMD GPU devices:")
    for index, device in enumerate(devices):
        print(f"{index}: {device}")


if __name__ == "__main__":
    main()
