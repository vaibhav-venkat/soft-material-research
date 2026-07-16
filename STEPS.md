# Steps

## New sims
Change L_y and L_z within the the channel/prism case such that instead of the volume being same as the `C = 60.5D` case, it is the same surface area.

So `2LxLy + 2LxLz` = `4L * Lx` where `L = Ly = Lz`. Thus `L = S/(Lx * 4)` so `L = pi * R/4` (double check this)

Also run a 2-plane versio ninstead of 4 walls its two walls which are atop the x plane, i.e x and y are peridic with z being the walls Lz/2 and  -Lz/2. Do this both for the case in which the volume of the box LzLyLx where Lz = Ly is the same as the cylinder 60.5D case, and also where the surface area is the same i.e 2LLx = S.

And also Run a 2D version itself within HOOMD (look up how, mostl likely fixing Z = 0 and other thigns like integrals), and don't worry about getting the same volume or anything, just use the same Ly from the case of the 2-plane. There will just be two walls at y = Ly/2 and y = -Ly/2 with periodic x.

For the 2d version no hexatic order calculations, just render it alongsisde its coords as well in the safetensor, nothing else (and polarization).

Run all 3 concurrently, with the first channel on GPU 0 and the other 2 on GPU 1. Don't run anything now, just give the command similar to the `big_lx` right now. Exclude the fixed cylinder cases, but don't delte them just exclude them from this run. FOr reference, this is how the current pipeiline is run:
```bash
CONDA_OVERRIDE_CUDA=12.9 \
pixi run python -u -m hexatic.confinement_comparison.run_analysis \
  --all \
  --gpu-ids 0,1 \
  --workers 3 \
  --backend auto \
  --output-root /mnt/drive3/vaibhav_data/confinement_comparison_production_run1 \
  --require-gpu \
  --overwrite
```
So like include a `--sim` option which allows things:
```
cylinder_rattle_tangent
cylinder_rattle
prism_volume
prism_surface_area
sandwich_surface_area
sandwich_volume
two_dimension
```

## New Analysis


Create a folder called `big_lx_analysis` in the `hexatic/` dir. This will host the new analysis. 

1. Calculate using the same methods as in the `big_lx`: the COM velocity. Then plot the `<V(0)V(t)>` which is the average kind of correlation. And make sure to normalize it at 1 at t = 0, i.e `<V(0)V(t)>/<V(0)V(0)>`.

DO this for `Lx` multiples `1` and `16`, we can add more if needed so make it modular and not hardcoded. 
Use the safetensors for this calculations simalarly, do not use any GSD things. This function will be called `C_v(t)`. Do this for positive `t`. 

You might have to do this for like frames `0...995` but not the very end since they'll have maybe little samples (confirm this or double check it upon execution, not now).

2. Do the same as the above for `C_(psi6)` which is the hexatic order which should be between `0` and `1`. This psi_6 will be using all the particles so no COM thing. Still the x axis is the lag time `t`.

3. You will plot this on one plot (remember everyhting is normalized so range is 0-1) for both the Lx cases for now, with the `C_v` being solid and `C_(psi 6)` being solid. 4 things on one plot

Just do this for now in the analysis. Only using safetensors, do not run anything yet, but give the command for it
