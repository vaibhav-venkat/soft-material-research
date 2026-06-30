The goal is reproduce and fit partial_t rho. 

First, look through ./hexatic/model_fitting too se a good idea of the variable names, process, etc. because i will be referencing them more.

First, we must create core packages and most raw computing in rust. These rust functions will be used via PyO3 and maybe maturin in a new folder in hexatic/rho_fitting. Only there, even if these rust functions can be applied to other folders don't do it.

I have attached the two papers we will reference the most. We will follow a simiar process as in `model_fitting`, just way less verbose and more clean. 

In terms of Rust:
- Use the existing pixi environment present in the `pixi.toml` to set up rust, I already have rustup installed
- Some of the packages are not installed that are required for the program, so put them in the toml or install them someway
- All rust code will be separate from python and go in `rust/` in the root.
- You will need at least `PyO3` for interop and, I believe, `maturin`

1. The loading of current trajectory *.gsd and *.npz will be loading through Python specifically, and then passed into Rust. Look into the `numpy` crate in Rust, as well as the `ndarray` crate that converts the numpy for further use.

2. While you have coarsed grain for now in Python, including separate packages like `math_utils.py`, etc. You will switch the coarse grain mechanism to Rust. Use a guassian kernel sum defined:
```
rho[t, X, Y] = sum_i K(grid - x_i[t])
P[t, X, Y, :] = sum_i K(grid - x_i[t]) * p_i[t]
```
Keep in mind you may need to recompute P, it is currently within most `*.npz`'s (double check this, however) defined as a polarization noramlized by density. However, the paper wants it to be not normalized and be the polariazation density. In this case, p_i is just the particle's vector.

Keep in mind for most of this, like before `x = x` and `y = R \theta`, account for that later but most of the logic is already in the python for you to see and double check.

3. Differing slightly from the current `model_fitting` we represent both rho and polarization using Chebyshev modes in time and Foureir modes in space:
```
rho(t, x) = Σ rho_hat [n, q] T_n(t) F_q(x)
p(t, x) = Σ p_hat [n, q] T_n(t) F_q(x)
```

Then, randomly sample `N_d = 5e5` time-space points for the regression system with `x = x` and `y = R \theta`, so we don't have to calculate over every single area. Rust will generate the actual sampled rows/mask from this.

Then, you will find all the `dx`, `dy`, `lap.. rho`, `div rho` etc. using the Fourier, and the `dt`, etc. using the Chebyshev. Use the mask with the `N_d`

You will split this off into calculating the former fields in rust with `rustfft`, and the time component using Python with `Chebyshev` in either scipy or numpy.

4. Define the candidate library in python using a list of strings, or whatever.
Should be one for density and one for polarization. Start off simple, with:
```
```
But we will soon, later, add more like what we did with the `model_fitting`
Rust itself is the one that will build the candidate library, still using the `N_d` like it always will for ever.
For the poliarzation equation, stack `x` and `y` componetns so both use the same coefficient.

5. Then, you run the STLSQ. First do this in python preferably, using the PySINDY package already insetalled.
Here is the algorithm:
```
for each threshold:
    for each random 50% subsample:
        run STLSQ
        record selected terms
compute importance scores
keep terms above 0.6
refit selected terms on full data
```

With the STLSQ settings:
`alpha = 1e-6`, `masx_iter = 20`, `normalize_columns=True`, `fit_intercept=False`
And for looping over threasholds, you choose `N = 40` values for the STLSQ threshold paramter `tau` on a log10 scale over:
`[tau_max, eps tau_max]` with `eps = 1e-2`
You can get tau_max by doing, just its defined so nothing survives.
```python
coef0, *_ = np.linalg.lstsq(Xn, yn, rcond=None)
tau_max = np.max(np.abs(coef0)) * 1.01
```

6. For the next step, just refit on the full, unnormalized data as per the algorithm
7. Then, validate:
In python:
Plot the predicted vs true, imporotance scores, coefficients, kymographs.
Use `snapshots, heatmaps, vector fields, etc.`

Once again, use PyO3 + numpy + ndarray and other things to move data between rust and python. 

ALways trust the paper more than me. I will ask you to make a plan. If you have any questions/follow-ups/conflicts of this with the paper than raise it up. This plan should be fairly detailed
