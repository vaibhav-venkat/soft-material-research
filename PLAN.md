Within the hexatic/new_sims_analysis/plot_correlation.py:

1. Separate the velocity correlation:
  a. Keep the COM velocity correlation on each plot
  b. Then, plot the correalation of essentially the orientation of particle i with particle j over all particles within each frame, similar to the plot_ideal_orientation_autocorrelation but instead of having orientation i correlate with orientation i at lag, its oritneation i correlate with orientation j at lag (self correlation is still allowed i.e i can equal j). Should still be one value per each frame.
  c. The above should represent the relative U_0 * P within the normal function `v = U_0 P + F_gamma`, so just account for that
  d. Then plot the difference between correlation of the COM v and q

Here is more info if needed, its my derivations so they may be incorrect, in that case you correct them and ask for my confiremationbefore doing the implementation. This is just so my plan aligns with yours:
so essentially `COM = 1/N sum_i v_i = V`
`V(0)V(t) = 1/N^2 sum_i sum_j V_i(0)V_j(t) = 1/(N^2) sum_i sum_j (U_0q_i(0)q_j(t) + F_i(0) F_j(t)`

So the idea is we can see what causes what, i.e if the current correaltoin has oscillations we can separate to understand wether the forces or the orienations cause it. 

All of my derivations are in terms of the autocorrelation but you will still do the pearson correlation

If its my understanding the Time complexity will be `~O(T N^2)` if theres any way to reduce this without comprising the actual algorithm, let me know before implementation.
