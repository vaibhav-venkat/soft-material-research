"""Small plotting utilities for the new-sims safetensor outputs."""

import matplotlib

# Every module here writes files rather than opening windows, and this runs
# before any submodule imports pyplot.
matplotlib.use("Agg")
