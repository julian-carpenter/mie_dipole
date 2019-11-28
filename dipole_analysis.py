import os
import numpy as np
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils import TdseMie
from contextlib import nullcontext

sns.set_style("white")

# SETTING THE PATHS
root_ = os.path.abspath(os.path.sep)
base_dir = os.path.join(root_, "path", "to", "tdse_data")

# Should the plots be printed out
printing = True
# GET THE HELPER
wavelength = 785
max_int = 20
ints = []
for wavelength in range(760, 820):
    print("Working on: {}nm IR".format(wavelength))
    save_folder = os.path.join(os.curdir, "tdse_projection_plots", str(wavelength))
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    tdse_mie = TdseMie(file_path, lam=wavelength,
                       max_int=max_int, save_folder=save_folder)

    # CUT THE RIGHT SIDE OF d AND A
    tdse_mie.apply_envelope(on="A", width=32)
    tdse_mie.apply_envelope(on="d", width=32)

    # GET ALPHA
    lo = 15  # lower energy limit in eV
    hi = 30  # upper energy limit in eV
    res = 25000  # number of values to interpolated between 'lo' and 'hi'
    tdse_mie.get_alpha(low=lo, high=hi, res=res)
    # GET N
    tdse_mie.get_n()

    # PLOTTING
    # ALPHA
    tdse_mie.plot_real_imag_scan("a", printing=printing)
    # REF. INDEX
    tdse_mie.plot_real_imag_scan("n", printing=printing)

    # PLOT OVER ENERGY
    tdse_mie.plot_dependence_on_energy("a", printing=printing)
    tdse_mie.plot_dependence_on_energy("n", include_ref=True,
                                       printing=printing)

    # PLOT Intensity Waterfall
    tdse_mie.plot_intensity_waterfall("a", max_idx=17000,
                                      printing=printing)
    tdse_mie.plot_intensity_waterfall("n", printing=printing)

    # PLOT OVER INTENSITY
    tdse_mie.plot_dependence_on_intensity("n", arg="imag",
                                          printing=printing)
    tdse_mie.plot_dependence_on_intensity("n", arg="real",
                                          printing=printing)

    tdse_mie.plot_dependence_on_intensity("a", arg="imag",
                                          printing=printing)
    tdse_mie.plot_dependence_on_intensity("a", arg="real",
                                          printing=printing)

    # PLOT RADIAL PROFILES
    droplet_radius = 750  # in nm
    angle_res = 1000
    max_angle = 30
    harm0 = 13
    harm1 = 15
    theta = np.linspace(0, max_angle, angle_res)
    rads_13 = tdse_mie.radial_profiles(theta, droplet_radius, harm0,
                                       printing=printing)
    rads_15 = tdse_mie.radial_profiles(theta, droplet_radius, harm1,
                                       printing=printing)

    # PLOT SCATTERING IMAGES
    # tdse_mie.plot_scat_images(rads_13,
    #                           ints=(0, -1),
    #                           n=harm0,
    #                           max_angle=max_angle,
    #                           use_log=True,
    #                           r=droplet_radius)
    #
    # tdse_mie.plot_scat_images(rads_15,
    #                           ints=(0, -1),
    #                           n=harm1,
    #                           max_angle=max_angle,
    #                           use_log=True,
    #                           r=droplet_radius)
    # plt.close("all")
    int_ = tdse_mie.plot_scat_images(np.array(rads_13) + np.array(rads_15),
                                     ints=(0, -1),
                                     max_angle=max_angle,
                                     use_log=True,
                                     r=droplet_radius,
                                     printing=printing)

    ints.append(int_)
    del tdse_mie

# Plot the Brightness on the detector w.r.t. the wavelegth of the IR
if printing:
    context = plt.rc_context({'font.size': 26,
                              "text.usetex": True,
                              "text.latex.preamble": r'\usepackage{mathpazo}'})
else:
    context = nullcontext()

save_folder = os.path.join(os.curdir, "tdse_projection_plots")
ints = np.array(ints)

np.savez(os.path.join(save_folder, "detector_brightness_dependence_on_ir_wavelength.npz"),
         ints=ints, i_w_cm_2=tdse_mie.i_w_cm_2)
with sns.axes_style("whitegrid"), context:
    if printing:
        f, ax = plt.subplots(1, 1, figsize=(32, 16))
    else:
        f, ax = plt.subplots(1, 1, figsize=(16, 8))

    ax.plot(tdse_mie.i_w_cm_2, ints,
            linewidth=1)
    ax.set_xlabel(r"Intensity [$W/cm^2$]")
    ax.set_title("Detector Brightness dependence on the IR wavelength")
    f.tight_layout()
    if printing:
        f.savefig(os.path.join(save_folder, "detector_brightness_dependence_on_ir_wavelength.pdf"),
                  dpi=300)
        plt.close(f)
