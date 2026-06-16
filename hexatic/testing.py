if __package__:
    from hexatic import analysis as hx
    from hexatic.constants import cylinder
else:
    import analysis as hx
    from constants import cylinder

import numpy as np
data = np.load("hexatic/output/cylinder/active_matter_fields.npz")
print(sum)
