Within the band_analysis_id, do as follows. We will generate 2 images, image one will have 2 separate plots and image 2 will have a gridded 2x2 4 plots. Plot using Seaborn and SVG. But before we do that, here is how to handle I/O. Currently we are reading from *.safetensor files to determine the identification for the bands, including the shell mask. Now, here is what you will do:

## I/O

Keep support from reading from the safetensors, and by default assume all passed safetensor share the same geometry with the different seeds (do not verify the manifests at all). Essentially for the plots you will treat each band as an individual data point.

However, you will also add support for gsd from a flag `--gsd`. This will require one safetensor to be inputted as a `--base-dir` in which the shell mask will be determined. The GSD file of course stores positions so everything should still be able to be calculated. 

You will also have an option `--max-frame` in which is the maximum frame you are to do the analysis on, not in simulation units. E.g `--max-frame 900` will truncate to 900 frames. 

First, before plotting, you will identify the bands.

## Plots

### Image 1

**Plot 1**

Plot 1 will consist of the probability distribution as a function of the band area $A$, i.e $P(A)$ . You will calculate the probability distribution of the Area change $\Delta A$ during a time increment as default $\tau = 1$ (Where $\tau$ is in simulation units). The x axis is the band area in units $\sigma^2$ (i.e divide the actual area by the Particle diameter squared), and the y axis is the $\Delta A$ in units $\sigma^2$. The plot will have a color scale and is a heatmap, where the color maps to $P(A)$, for now just make the color scale linear and I'll see if we need to make anything a log scale based on how things work.

**Plot 2**

This will contain two functions of the bubble area $A$ in $\sigma^2$ units on the x axis: first moment $\langle \Delta A \rangle$ with circle symbols and second moment $\langle \Delta A^2 \rangle$ scaled by $\sqrt{A}$ . You will mark the first moment in red, second moment in blue. Each neesd a differnt y-scale.

### Image 2

Like before all area is in units of $\sigma^2 = D^2$

**Plot 1**

Plot the stationary probability distribution of $P(A)$ i.e the probability of a band being area $A$ across all times and all seeds.Make each data point a red circle, since we may fit this later.

**Plot 2**

Plot the mean first-passage time $T(A)$ in simulation units i.e $\tau$ .

Plot these also as red circles. This is just the time for a band at area $A$ to shrink to completely zero or die essentially. Treat a split and merge also as a death, we may change that defintioin later. But remember using `--persistence-time` or whatever the stable bands should only be tracked.

**Plot 3**

Find the 4 quartiles for the stargin area of all bands, i.e when a band is birthed what area does it have? And we will find the distribution of lifetimes so $P(\tau)$ for each fo the 4 quartiles. Separet teh quartles by color mostly and keep them all as circles. Allow each quartle to have a reprsentative intiial area $A_0$ which is just the median of the quartile, purely for display purposes (i.e put it on the plot). 

**Plot 4**
A more generalized version of **3**, essentially just $P(\tau)$ the probability distribution of a bands lifetime across all intiial areas and all stable bands in general, with the red circles again.



Don't run any tests for these, I'll test them on my own and report back. 
