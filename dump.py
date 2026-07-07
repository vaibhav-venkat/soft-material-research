import inspect
import numpy as np

for frame_info in inspect.stack():
    frame = frame_info.frame
    if "coarse" in frame.f_locals:
        coarse = frame.f_locals["coarse"]
        np.savez_compressed(
            "hexatic/rho_fitting/output/radius_15D_coarse_fields_checkpoint.npz",
            **coarse,
        )
        break
