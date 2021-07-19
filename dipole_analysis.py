import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
from utils import TdseMie
from utils.ref_literature import get_turbo
from contextlib import nullcontext

sns.set_style("white")

# SETTING THE PATHS
root_ = os.path.abspath(os.path.sep)
# base_dir = os.path.join(root_, "home", "julian", "Downloads", "results_tdse_nlevel")
# base_dir = os.path.join(root_, "home", "julian", "Downloads", "tdse_solver")
# base_dir = os.path.join(root_, "mnt", "storage", "MBI", "MarcSimulations")
base_dir = os.path.join(root_, "run", "media", "julian", "usb1", "dipole_response")
# base_dir = os.path.join(root_, "home", "julian", "Downloads", "dipole_response")
# file_path = os.path.join(base_dir, "13_15_20fs_deact_states")
file_path = os.path.join(base_dir, "high_res_scan")

# Should the plots be printed out
printing = True
nlevel = False
iap = True

if printing:
    matplotlib.use('agg')
# GET THE HELPER
wavelength = 798
max_int = 29
ints_13_15 = []
ints_13 = []
ints_15 = []
for wavelength in range(790, 806):
    print("Working on: {}nm IR".format(wavelength))
    save_folder = os.path.join(os.curdir, "tdse_projection_plots_no_clipping_{}".format("n_level" if nlevel else "full"),
                               str(wavelength))
    if not iap:
        save_folder += "_20fs"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    tdse_mie = TdseMie(file_path, lam=wavelength,
                       max_int=max_int,
                       save_folder=save_folder,
                       nlevel=nlevel,
                       iap=iap,
                       cut_absorption=False)

    # CUT THE RIGHT SIDE OF d AND A (or E)
    w = tdse_mie.apply_envelope(on="d", optimize_window=True, use_hann=True)
    tdse_mie.apply_envelope(on="A", width=w, use_hann=True)

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
    # tdse_mie.plot_dependence_on_energy("a", printing=printing, ints=(0, -1))
    tdse_mie.plot_dependence_on_energy("n", include_ref=True, ints=(0, -1),
                                       printing=printing)

    # PLOT Intensity Waterfall
    tdse_mie.plot_intensity_waterfall("a", printing=printing,
                                      min_idx=8000, max_idx=17000, max_int=max_int)  # The absorption cross-section
    # tdse_mie.plot_intensity_waterfall("n", printing=printing, max_int=max_int)

    # PLOT OVER INTENSITY
    # tdse_mie.plot_dependence_on_intensity("n", arg="imag",
    #                                       printing=printing)
    # tdse_mie.plot_dependence_on_intensity("n", arg="real",
    #                                       printing=printing)
    tdse_mie.plot_dependence_on_intensity("n", arg="both", printing=printing, ints=(0, -1))

    # tdse_mie.plot_dependence_on_intensity("a", arg="imag",
    #                                       printing=printing)
    # tdse_mie.plot_dependence_on_intensity("a", arg="real",
    #                                       printing=printing)

    # PLOT RADIAL PROFILES
    droplet_diameter = 1000  # in nm
    angle_res = 1000
    max_angle = 30
    harm0 = 13
    harm1 = 15
    theta = np.linspace(0, max_angle, angle_res)
    rads_13 = tdse_mie.radial_profiles(theta, droplet_diameter, harm0,
                                       printing=printing)
    rads_15 = tdse_mie.radial_profiles(theta, droplet_diameter, harm1,
                                       printing=printing)

    mmax_scat = np.nanmax([x[50:].max() for x in rads_13] + [x[50:].max() for x in rads_15] +
                          [np.nanmax(x[50:] + y[50:]) for x, y in zip(rads_13, rads_15)])
    mmin_scat = np.nanmin([x[50:].min() for x in rads_13] + [x[50:].min() for x in rads_15] +
                          [np.nanmin(x[50:] + y[50:]) for x, y in zip(rads_13, rads_15)])
    # PLOT SCATTERING IMAGES
    int_13 = tdse_mie.plot_scat_images(rads_13,
                                       n=harm0,
                                       max_angle=max_angle,
                                       cmap_norm=np.log(mmax_scat),
                                       use_log=True,
                                       r=droplet_diameter,
                                       printing=printing)

    int_15 = tdse_mie.plot_scat_images(rads_15,
                                       n=harm1,
                                       max_angle=max_angle,
                                       cmap_norm=np.log(mmax_scat),
                                       use_log=True,
                                       r=droplet_diameter,
                                       printing=printing)

    int_13_15 = tdse_mie.plot_scat_images(1.47 * np.array(rads_13) + np.array(rads_15),
                                          max_angle=max_angle,
                                          use_log=True,
                                          cmap_norm=np.log(mmax_scat),
                                          r=droplet_diameter,
                                          printing=printing)

    ints_15.append(int_15)
    ints_13.append(int_13)
    ints_13_15.append(int_13_15)
    del tdse_mie
    # plt.close("all")

# Plot the Brightness on the detector w.r.t. the wavelegth of the IR
if printing:
    context = plt.rc_context({'font.size': 32,
                              "text.usetex": True,
                              "text.latex.preamble": r'\usepackage{mathpazo}'})
else:
    context = nullcontext()

save_folder = os.path.join(os.curdir, "tdse_projection_plots_{}".format("n_level" if nlevel else "full"))
if not iap:
    save_folder += "_20fs"
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

for iint, ii_name in zip((ints_15, ints_13, ints_13_15), ("15", "13", "13_15")):
    iint = np.array(iint)
    np.savez(os.path.join(save_folder, "detector_brightness_dependence_{}_harm_on_ir_wavelength.npz".format(ii_name)),
             ints=iint, wavelengths=np.arange(790, 806))
    with sns.axes_style("whitegrid"), context:
        if printing:
            f, ax = plt.subplots(1, 1, figsize=(32, 16))
        else:
            f, ax = plt.subplots(1, 1, figsize=(16, 8))

        ax.plot(np.arange(790, 806), iint, linewidth=2)
        ax.axhline(1, color="tab:red", linestyle="--")
        ax.set_xlabel(r"Intensity [$W/cm^2$]")
        ax.set_title("Detector Brightness dependence on the IR wavelength")
        ax.set_ylim(0, 1.5)
        f.tight_layout()
        if printing:
            f.savefig(os.path.join(save_folder,
                                   "detector_brightness_dependence_{}_harm_on_ir_wavelength.pdf".format(ii_name)),
                      dpi=300)
            plt.close(f)

# Now do a Grid Plot for the detector brightness
yy = np.arange(100, 1601, 25)
xx = np.arange(790, 806)

angle_res = 1000
max_angle = 30
harm0 = 13
harm1 = 15

lo = 15  # lower energy limit in eV
hi = 30  # upper energy limit in eV
res = 25000  # number of values to interpolated between 'lo' and 'hi'
theta = np.linspace(0, max_angle, angle_res)

detector_intensities = np.zeros([len(xx), len(yy)])
save_dict = {}
for ii, wavelength in enumerate(xx):
    print("Working on: {}nm IR".format(wavelength))
    save_folder = os.path.join(os.curdir, "tdse_projection_plots_{}".format("n_level" if nlevel else "full"),
                               str(wavelength))
    if not iap:
        save_folder += "_20fs"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    tdse_mie = TdseMie(file_path, lam=wavelength,
                       max_int=max_int,
                       save_folder=save_folder,
                       only_max_int=True,
                       nlevel=nlevel,
                       iap=iap,
                       cut_absorption=True)

    # CUT THE RIGHT SIDE OF d AND A
    w = tdse_mie.apply_envelope(on="A", optimize_window=True)
    tdse_mie.apply_envelope(on="d", width=w)

    # GET ALPHA
    lo = 15  # lower energy limit in eV
    hi = 30  # upper energy limit in eV
    res = 25000  # number of values to interpolated between 'lo' and 'hi'
    tdse_mie.get_alpha(low=lo, high=hi, res=res)
    # GET N
    tdse_mie.get_n()

    energy_ev_13 = tdse_mie.calc_harmonic(harm0)
    energy_ev_15 = tdse_mie.calc_harmonic(harm1)
    for jj, droplet_diameter in enumerate(yy):
        print("Working on: {}nm droplet radius".format(droplet_diameter))
        y1_13 = tdse_mie.get_single_rad(theta, energy_ev_13, droplet_diameter, 0)
        y2_13 = tdse_mie.get_single_rad(theta, energy_ev_13, droplet_diameter, -1)
        y1_15 = tdse_mie.get_single_rad(theta, energy_ev_15, droplet_diameter, 0)
        y2_15 = tdse_mie.get_single_rad(theta, energy_ev_15, droplet_diameter, -1)

        y1_13[:50] = 0  # This corresponds to ~ 3° detector hole
        y2_13[:50] = 0
        y1_15[:50] = 0
        y2_15[:50] = 0

        img0, ma, mi = tdse_mie.get_diffraction(y1_13 + y2_13, False)
        img1, ma_2, mi_2 = tdse_mie.get_diffraction(y1_15 + y2_15, False)

        intensity_difference_on_detector = img1.sum() / img0.sum()
        save_dict.update({"{}_{}".format(wavelength, droplet_diameter): intensity_difference_on_detector,
                          "{}_{}_y1_13".format(wavelength, droplet_diameter): y1_13,
                          "{}_{}_y2_13".format(wavelength, droplet_diameter): y2_13,
                          "{}_{}_y1_15".format(wavelength, droplet_diameter): y1_15,
                          "{}_{}_y2_15".format(wavelength, droplet_diameter): y2_15,
                          "{}_{}_img0".format(wavelength, droplet_diameter): img0,
                          "{}_{}_img1".format(wavelength, droplet_diameter): img1
                          })
        detector_intensities[ii, jj] = intensity_difference_on_detector
    del tdse_mie

save_folder = os.path.join(os.curdir, "tdse_projection_plots_{}".format("n_level" if nlevel else "full"))
np.savez(os.path.join(save_folder, "detector_brightness_dependence_on_ir_wavelength_scan.npz"),
         ints=save_dict, wavelengths=xx, radii=yy)
np.save(os.path.join(save_folder, "detector_intensities.npy"), detector_intensities)
plt.register_cmap(name='turbo', data=get_turbo(), lut=256)
with plt.rc_context({'font.size': 46,
                     "text.usetex": True,
                     "text.latex.preamble": r'\usepackage{mathpazo}'}):
    f, ax = plt.subplots(1, 1, figsize=(30, 24))
    cs = ax.imshow(detector_intensities, cmap="RdBu_r", clim=(0, 2))
    ax.set_aspect(detector_intensities.shape[1] / detector_intensities.shape[0])

    ax.set_yticks(np.arange(0, len(xx), 5))
    ax.set_yticklabels(xx[::5])
    cbar = f.colorbar(cs, ax=ax, shrink=.9)
    ax.set_xticks(np.arange(0, len(yy) + 1, 12))
    ax.set_xticklabels(yy[::12] / 2)
    ax.set_ylabel("Driving wavelength of IR laser [nm]")
    ax.set_xlabel("Nanodroplet radius [nm]")
    f.tight_layout()
    f.savefig(os.path.join(save_folder, "detector_intensities_scan.pdf"),
              dpi=300)
    plt.close(f)
