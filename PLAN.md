# Plan - Fitting J to values

Within [density_analysis](~/structures_and_functions/hexatic/density_analysis), create a file called `run_fitting.py` and a folder called `fitted`.
- `run_fitting.py` should be similar to run_film_continuity.py
- Within fitting, this functionality should be present:
For each frame/translation:
`rho(x,y, t)` must be calculated according to the guassian kernel functionality present within [common.py](~/structures_and_functions/hexatic/active_matter_cylinder/common.py). 

**NOTE**: This field may already exist within the npz folder, specifically within the active_matter_fields.npz. It MUST use the guassian_kernel, however. You can double check this within he active_matter_cylinder folder. (if not, recalculate)

Also, use `y = R theta`. If this can be done after loading from the npz, use the npz. If not, recalculate.

For the same grid and same time transition:
``
J_x(x,y,t+1/2)
J_y(x,y,t+1/2)
``
Using the same grid as `rho`

## Then, compute density gradients:
For each frame, take the 2d FFT of the guassian-smoothed density field
`rho_hat = fft2[rho(x, y)]` (preferably use scipy)
Then build the wavenumber arrays:
``
kx = 2*pi m / Lx
ky = 2 * pi * n / Ly
Ly = 2 * pi * R
Lx = defined Lx within the constants file; or, as in the film_continuity, from the cached x_edges
(partial_x rho)_hat = i kx rho_hat
(partial_y rho)_hat = i ky rho_hat
partial_x rho = IFFT2((partial_x rho)_hat)
partial_y rho = IFFT2((partial_y rho)_hat)
``

Then make the regression target and fit:
`J_x = c_x partial_x rho`
`J_y = c_y partial_y rho`

Use the linear least squares regression for now.
Then plot
`c_x partial_x rho vs J_x_measured`
`c_y partial_y rho vs J_y measured`
And I mean the residual on a heatmap.
