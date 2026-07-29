We are adding new file within the new-sims-analysis which is called `plot_all`. Similar to the `plot_correlation` (you should look at this file for inspiration), this should support these types of safetensor outputs

- 2d or 3d (read manifests to identify)
- all new-sims analysis allowed, including non-interacting ones
- All the confinement cases, which include the prism, RATTLE constrained manifold, and RATTLE + active tangent constrained.
- The original big-lx cases for all Lx

Only read things for circumference `60.5D`. If that is not available for one of the safetensors, then throw an error and fail fast.

Now, within plot_all there should be two sections, which should be split up as different files:

1. First, is the laplacian grid. We already have formal defintions for this, but heres whats differnt. You will calculate the laplacian for all the values given. 
Then, you will plot a grid of all different cases, so for N cases there are N plots in one image, which showcases only the log10(|L(r, omega)|). 
It is very important you select the same r range and omega range. In addition, the color scale for log10 should be the same for all cases, because the amplitude/magnitude differs heavily. 

That also means just get the max and min across all cases to determine the plotting scale. That only affects the plots, not the analysis itself.

2. Second, is the entropy.

We will do this for L(r = 0, omega)
define the discrete normalized: 
`x_k = (|L(0, omega_k)|^2)/[sum_j (|L(0, omega_j)|^2)]`
And the define `H_spec = -(sum_k x_k ln x_k)`
Also account for cases where `L(0, omega_k) = 0`, since ln is not defined for that, most likely just skip that term.

the \(\omega\) range, frequency resolution, and number of bins should be identical for ALL CASES. Validate this before plotting. If any case has a different omega grid, fail fast.

Now here is I/O specs:
Treat things with the same id/manifesti id or case as different seeds, and calculate the correlation and laplace accordingly. Just treat them all like different data points.

Here is the plot specs:
- First plot is the grid with all cases, same color scale and everything.
- Second is a bit more complicated, x axis is the L_x multiple (1..16). The y is the entropy of course. Since a lot will be at L_x = 1 use different symbols for the different cases. On your discretion, since there are a lot of cases, you can group some into the same symbol or color and change there symbol or color based on the preivous decision. E.g, keep same symbol and diff color, or diff symbol + same color. Do not show the little STD bars because it will get to cluttered if you do that.

Correct anything I made mistakes, and ask rigorous questions if you are in the planning stage. You don't need to cover every edge case for this, and keep everything pretty simple.

In terms of the CLI, we won't be doing the input-dirs with the flags because there will be so many, it should be within the root plot_all file as an array of strings i can edit, using absolute paths.
