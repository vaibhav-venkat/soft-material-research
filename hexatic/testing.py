import numpy as np
data = np.load("hexatic/output/cylinder/active_matter_shell_fields.npz")
print(data.files)
print(data['rho_count'])
