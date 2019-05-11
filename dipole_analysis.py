import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import TdseMie

sns.set_style("white")

# SETTING THE PATHS
root_ = os.path.abspath(os.path.sep)
base_dir = os.path.join(root_, "home", "julian", "Downloads", "dipole_response")
file_path = os.path.join(base_dir, "high_res_scan")

# GET THE HELPER
wavelength = 800
max_int = 20
tdse_mie = TdseMie(file_path, lam=wavelength, max_int=max_int)

# CUT THE RIGHT SIDE OF d AND A
# tdse_mie.apply_envelope(on="A", width=4.42)
tdse_mie.apply_envelope(on="d", width=4.42)

# GET ALPHA
lo = 15  # lower energy limit in eV
hi = 30  # upper energy limit in eV
res = 5000  # number of values to interpolated between 'lo' and 'hi'
tdse_mie.get_alpha(low=lo, high=hi, res=res)
# GET N
tdse_mie.get_n()

# PLOTTING
plt.close("all")
# ALPHA
tdse_mie.plot_real_imag_scan("a")

# REF. INDEX
tdse_mie.plot_real_imag_scan("n")

# PLOT SINGLE ALPHA
tdse_mie.plot_dependence_on_energy("a")
tdse_mie.plot_dependence_on_energy("n", include_ref=True)

# PLOT RADIAL PROFILES
droplet_radius = 300  # in nm
angle_res = 500
theta = np.linspace(0, 30, angle_res)
tdse_mie.radial_profiles(theta, droplet_radius, 13)
tdse_mie.radial_profiles(theta, droplet_radius, 15)
# SHOW IT
plt.show()
