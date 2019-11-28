import os
import numpy as np
import scipy.constants as cs
from tqdm import trange
from scipy.special import expit
import miepython as mp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
from scipy import signal
from contextlib import nullcontext
from utils.ref_literature import REFERENCE, get_turbo

np.seterr(divide='ignore', invalid='ignore')  # there will be some 'divide by 0' warnings


class TdseMie(object):
    def __init__(self,
                 data_path,
                 lam=800,
                 max_int=20,
                 step_size=6.25e-4,
                 save_folder="."):
        # CONSTANTS
        self.alpha = None  # Placeholder
        self.n = None  # Placeholder
        self.n_interp = None  # Placeholder
        self.a_interp = None  # Placeholder
        self.interp_grid_ev = None  # Placeholder
        self.a_si = None  # Placeholder
        self.env = 1.  # broadcastable dummy for the sigmoidal envelope
        self.lam = lam  # in nm
        self.max_int = max_int  # in Marc's units
        self.step_size = step_size  # given by Marc
        self.c = cs.c
        self.h = cs.h
        self.hbar = cs.hbar
        self.e0 = cs.epsilon_0
        self.eV = cs.physical_constants["electron volt"][0]
        self.au_time = cs.physical_constants["atomic unit of time"][0]
        self.au_length = cs.physical_constants["atomic unit of length"][0]
        self.au_epsilon = cs.physical_constants["atomic unit of permittivity"][0]
        self.au_efield = cs.physical_constants["atomic unit of electric field"][0]
        self.au_edipole = cs.physical_constants["atomic unit of electric dipole mom."][0]
        self.au_p = cs.physical_constants["atomic unit of electric polarizability"][0]
        self.au_pot = cs.physical_constants["atomic unit of electric potential"][0]
        self.fine_struct = cs.physical_constants["fine-structure constant"][0]
        self.a0 = cs.physical_constants["Bohr radius"][0]
        self.nd = 0.022e30  # number density of liquid helium
        self.ref_res = 250  # the resolution for the interpolation of the reference values
        self.ref_values = REFERENCE(self.ref_res)
        self.save_folder = save_folder

        if not os.path.isdir(self.save_folder):
            self.save_folder = "."

        # LOAD THE DATA
        self.A_in_au = []
        self.d_in_au = []

        pargs = (self.max_int + 1, self.lam)
        desc = "Reading in {} intensities for {}nm".format(*pargs)
        for i_int in trange(pargs[0], desc=desc):
            ff = "dip{}_{}.dat".format(i_int, self.lam)
            data = np.loadtxt(os.path.join(data_path, ff))  # 0: t; 1:A(t); 2:d(t) | all in a.u.

            if i_int == 0:
                self.time_in_au = -data[:, 0]
            self.A_in_au.append(data[:, 1])
            self.d_in_au.append(data[:, 2])

        # setting up period, omega, the energy and the intensity axis
        self.t_vec = np.arange(len(self.time_in_au))
        self.period = np.diff(self.time_in_au).mean() * len(self.time_in_au)
        self.dw_au = (2 * np.pi / self.period) * self.t_vec
        self.dw_fs = (2 * np.pi / self.au_to_fs(self.period)) * self.t_vec
        self.e_ev = 1e15 * self.dw_fs * self.hbar / self.eV
        self.i_w_cm_2 = .5 / 1e4 * self.e0 * cs.c * (
                (self.au_pot / self.a0) * np.arange(pargs[0]) * self.step_size) ** 2

        # Check that A and d have dimensions (max_int, n)
        # if max_int == 1 we would end up with an array of dim n
        # we adding in that case an extra dim.
        self.A_in_au = np.array(self.A_in_au)
        try:
            assert len(self.A_in_au.shape) == 2
        except AssertionError as e:
            print("{}\n Adding an extra dimension to A".format(e))
            self.A_in_au = np.expand_dims(self.A_in_au, axis=0)

        self.d_in_au = np.array(self.d_in_au)
        try:
            assert len(self.d_in_au.shape) == 2
        except AssertionError as e:
            print("{}\n Adding an extra dimension to d".format(e))
            self.d_in_au = np.expand_dims(self.d_in_au, axis=0)

    def apply_envelope(self, on="A", width=10, center="max", sign=-1, use_hann=True):
        """
        In place multiplication of a sigmoidal envelope.
        'on' has to be 'A' or 'd' (not case-sensitive).
        'width' is the tau parameter in the sigmoid, given in fs.
        'center' is where the sigmoid should be 1/2 of the
        signal can be 'max' or a specific position in 'index'.
        """
        on = on.lower()
        assert on in ("a", "d")
        ind_width = self.fs_to_index(width)
        self.env = self.get_envelope(ind_width, center, sign, use_hann)
        if on == "a":
            self.A_in_au *= self.env
        elif on == "d":
            self.d_in_au *= self.env

    def get_alpha(self, low=15, high=30, res=25000, subtract_mean=True):
        """
        Returns complex alpha in au.
        """
        if subtract_mean:
            self.d_in_au = np.subtract(self.d_in_au.T, self.d_in_au.mean(axis=1)).T
        # The full (two-electron) dipole moment is 2*F[d](w).
        # See: Gaarde, M. B., Buth, C., Tate, J. L., & Schafer, K. J. (2011).
        # Transient absorption and reshaping of ultrafast XUV light by laser-dressed helium.
        # Physical Review A - Atomic, Molecular, and Optical Physics, 83(1).
        # https://doi.org/10.1103/PhysRevA.83.013419
        d_au_env_fft = 2 * np.fft.ifft(self.d_in_au, axis=1)
        a_au_env_fft = np.fft.ifft(self.A_in_au, axis=1)

        # ALPHA | alpha is d(w) / ( i * dw * A(w) )
        self.alpha = np.divide(d_au_env_fft, (1j * self.dw_au * a_au_env_fft))
        # INTERPOLATION
        # alpha
        self.interp_grid_ev = np.linspace(low, high, res)
        self.a_si = self.alpha * self.au_p
        self.a_interp = np.array(list(map(lambda fp: np.interp(self.interp_grid_ev, self.e_ev, fp), self.a_si)))
        return self.alpha

    def get_n(self):
        """
        Returns complex alpha in au.
        low and high are for the interpolation ... in eV
        """
        #  Check if alpha has already been calculated
        if self.alpha is None:
            self.get_alpha()

        def calc_n(a):  # Clausius-Mossotti-Relation
            frac = np.divide(3 * self.nd * a, 3 * self.e0 - self.nd * a)
            return np.sqrt(1 + frac)

        self.n = calc_n(self.a_si)

        # FOR THE INTERPOLATED ALPHA
        self.n_interp = calc_n(self.a_interp)

        return self.n, self.n_interp

    def get_envelope(self, width, center, sign=-1, signal_size=None, use_hann=True):
        """
        Construct a sigmoid that serves as an envelope.
        'width' is in 'index' and 'center' can be 'max' or in 'index'.
         Optional: 'signal_size' is the length of the signal.
        """

        if center == "max":
            c_ = int(18408)  # this was checked manually ... so be careful here
            # c_ = int(18222)  # this is what Thomas & Bjorn use
        else:
            c_ = center
        if signal_size is None:
            x = np.arange(self.time_in_au.size)
        else:
            x = np.arange(signal_size)

        if use_hann:
            env = np.zeros(self.A_in_au.shape[1])
            env[c_ - width:c_ + width] = signal.hanning(int(2 * width))
        else:
            env = expit(sign * (x - c_) / width)

        env = [env] * self.A_in_au.shape[0]
        return np.array(env)

    def fs_to_au(self, fs):
        return fs / (self.au_time * 1e15)

    def fs_to_index(self, fs, rel_to_max=False):
        """
        Convert a given time in fs to the corresponding index in the data
        if rel_to_max is True then fs is relative to the mean of all max's of A
        """
        if rel_to_max:  # get the mean of all max's of A
            max_ = 0
            for a in self.A_in_au:
                max_ += self.index_to_fs(np.argmax(a))
            max_ /= self.A_in_au.shape[0]
            fs = max_ + fs
        return int(sum(self.time_in_au < self.fs_to_au(fs)))

    def au_to_fs(self, au):
        return au * self.au_time * 1e15

    def au_to_index(self, au, rel_to_max=False):
        """
        Convert a given time in au to the corresponding index in the data
        if rel_to_max is True then au is relative to the mean of all max's of A
        """
        if rel_to_max:
            max_ = 0
            for a in self.A_in_au:
                max_ += self.index_to_au(np.argmax(a))
            max_ /= self.A_in_au.shape[0]
            au = max_ + au
        return int(sum(self.time_in_au < au))

    def index_to_fs(self, ii):
        ii = np.array(ii)
        return self.time_in_au[ii] * self.au_time * 1e15

    def index_to_au(self, ii):
        ii = np.array(ii)
        return self.time_in_au[ii]

    def calc_harmonic(self, n):
        n = np.array(n)
        return self.h * self.c / (self.lam * 1e-9 / n) / self.eV

    def ev_to_nm(self, ev):
        ev = np.array(ev)
        return self.h * self.c / ev / self.eV * 1e9

    def get_diffraction(self, rad, return_log=True, angle_res=500):

        xx = np.arange(-angle_res, angle_res)
        yy = xx.copy()
        x_grid, y_grid = np.meshgrid(xx, yy)
        z_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)

        if return_log:
            x_min = np.nanmin(rad[rad != 0])  # np.min(rad)
            global_min = np.log(x_min) if x_min > 0 else 0  # np.log(1e-3)  #
            global_max = np.log(np.max(rad))
        else:
            global_min = 1e-2  # np.min(rad)
            global_max = np.max(rad)

        if return_log:
            glob_min = np.exp(global_min)
        else:
            glob_min = global_min

        z_ = np.ones_like(z_grid) * glob_min
        for ii in range(z_grid[angle_res, :].shape[0]):
            try:
                for kk in range(angle_res):
                    try:
                        z_[kk, ii] = rad[np.round(z_grid[kk, ii]).astype(int)]
                    except IndexError:
                        pass
            except IndexError:
                pass
        z_[angle_res:, :] = np.flipud(z_[:angle_res, :])
        if return_log:
            z_ = self.inf_log(z_, global_min)
        return np.array(z_), global_max, glob_min

    @staticmethod
    def inf_log(arr, global_min):
        mask = (arr <= 0)
        notmask = ~mask
        out = arr.copy()
        out[notmask] = np.log(out[notmask])
        out[mask] = global_min  # np.log(1e-3)
        return out

    def radial_profiles(self, x, r, n=None, ev=None, normalize=False, ir_idx=-1, printing=False):
        """

        :param x: The scattering angle of interest (e.g. linspace from 0 to 30) [°]
        :param r: The radius of the droplet
        :param n: The harmonic with which we probe
        :param ev: Alternatively to 'n' you can pass your own energy of interest in eV
        :param normalize: (Bool) If True the brightest profile is set to peak at 1 and all other profiles
                          are normalized with the same value.
        :param ir_idx: Idx (In Marc's # index) to which the IR=0 profile should be compared (Default = -1)
        :return: Matplotlib figure
        """
        if printing:
            context = plt.rc_context({'font.size': 26,
                                      "text.usetex": True,
                                      "text.latex.preamble": r'\usepackage{mathpazo}'})
        else:
            context = nullcontext()

        if ev is None:
            assert n is not None
            energy_ev = self.calc_harmonic(n)
        else:
            energy_ev = ev

        y1 = self.get_single_rad(x, energy_ev, r, 0)
        y2 = self.get_single_rad(x, energy_ev, r, ir_idx)

        if normalize:
            norm = integrate.trapz(y1, x)
            y1 /= norm
            y2 /= norm

        lbl1 = r"Radial profile | Droplet Radius {} nm | " \
               r"Probing with: {:02.02f} eV | IR: 0 $W/cm^2$".format(r, energy_ev)
        lbl2 = r"Radial profile | Droplet Radius {} nm | " \
               r"Probing with: {:02.02f} eV | IR: {:02.02e} $W/cm^2$".format(r, energy_ev,
                                                                             self.i_w_cm_2[
                                                                                 ir_idx])
        if n is not None:
            lbl1 += " | {}th Harmonic".format(n)
            lbl2 += " | {}th Harmonic".format(n)
        with sns.axes_style("whitegrid"), context:
            if printing:
                f_rp, ax_rp = plt.subplots(1, 1, figsize=(32, 16))
            else:
                f_rp, ax_rp = plt.subplots(1, 1, figsize=(16, 8))

            ax_rp.semilogy(x, y1, label=lbl1, linewidth=1)
            ax_rp.semilogy(x, y2, label=lbl2, linewidth=1)
            ax_rp.set_xlabel("Scattering Angle [°]")
            ttl = "Radial profiles (Orth. pol.) | Change in brightness when IR is present: {:02.02f}%".format(
                100 * np.sum(y2) / np.sum(y1))
            if normalize:
                ttl += " | Normalized using the int. at IR=0"
            ax_rp.set_title(ttl)
            ax_rp.legend()
            f_rp.tight_layout()
            if printing:
                f_rp.savefig(os.path.join(self.save_folder,
                                          "{:02.02f}_{}_{:02.02e}_radial_profiles.pdf".format(energy_ev,
                                                                                              self.lam,
                                                                                              self.i_w_cm_2[-1])),
                             dpi=300)
                plt.close(f_rp)
            return y1, y2

    def plot_dependence_on_energy(self, on, ints=(0, -1),
                                  include_ref=False, n=(13, 15),
                                  ev=None, uncertainty=.5,
                                  printing=False):
        """
        :param on: Can be 'a' (alpha) or 'n' (refractive index). Both are case-insensitive
        :param ints: List of intensity to plot. Default is: (0, -1)
        :param include_ref: Only valid if 'on=n'. Plot also the ref values from
                    A. Lucas et al., Phys. Rev. B 28(25):2485 (1983)
        :param n: The harmonic we want to draw in the plot
        :param ev: Alternatively to 'n' you can pass your own energy of interest in eV
        :param uncertainty: Energy uncertainty (in eV) ... final output will be averaged across this energy range
        :return: Matplotlib figure
        """
        if printing:
            context = plt.rc_context({'font.size': 26,
                                      "text.usetex": True,
                                      "text.latex.preamble": r'\usepackage{mathpazo}'})
        else:
            context = nullcontext()

        on = on.lower()
        assert on in ["a", "n"]

        if on == "a":
            y = self.a_interp / self.au_p  # in a.u.
            lbl = (r"$\alpha_1$", r"$\alpha_2$")
        else:
            y = self.n_interp
            lbl = (r"$\delta$", r"$\beta$")

        if n is not None or ev is not None:  # only on arg can be not None
            assert (n is None and ev is not None) or (n is not None and ev is None)

        if n is not None:
            energy_ev = self.calc_harmonic(n)

            if isinstance(n, int):
                n = [n]
                energy_ev = [energy_ev]
            lbl_ = ["{} th Harmonic ({:02.02f} eV)".format(x, y) for x, y in zip(n, energy_ev)]
        elif ev is not None:
            energy_ev = ev
            if isinstance(energy_ev, int):
                energy_ev = [energy_ev]
            lbl_ = ["{:02.02f} eV".format(y) for x, y in energy_ev]
        else:
            energy_ev = None

        with sns.axes_style("whitegrid"), context:
            if printing:
                f, ax = plt.subplots(1, 1, figsize=(32, 16))
            else:
                f, ax = plt.subplots(1, 1, figsize=(16, 8))

            for i_ in ints:
                ax.plot(self.interp_grid_ev, y[i_].real if on == "a" else (1 - y[i_].real),
                        label=r"{} | IR: {:02.02e} $W/cm^2$".format(lbl[0], self.i_w_cm_2[i_]),
                        linewidth=1)
                ax.plot(self.interp_grid_ev, y[i_].imag,
                        label=r"{} | IR: {:02.02e} $W/cm^2$".format(lbl[1], self.i_w_cm_2[i_]),
                        linewidth=1)
            if on == "n" and include_ref:
                ax.plot(self.ref_values.e_ev, self.ref_values.delta, label=r"$\delta$ (Literature)", linewidth=1,
                        linestyle='dashed')
                ax.plot(self.ref_values.e_ev, self.ref_values.beta, label=r"$\beta$ (Literature)", linewidth=1,
                        linestyle='dashed')
            if energy_ev is not None:
                for i, (e, l) in enumerate(zip(energy_ev, lbl_)):
                    if uncertainty:
                        _idx = uncertainty / 2
                        ax.axvline(e - _idx, color=sns.color_palette()[i],
                                   linewidth=.5, linestyle="dashed")
                        ax.axvline(e + _idx, color=sns.color_palette()[i],
                                   linewidth=.5, linestyle="dashed")
                    ax.axvline(e, label=l, color=sns.color_palette()[i],
                               linewidth=1, linestyle="solid")
            ax.set_xlabel("Energy [eV]")
            ax.set_xlim([self.ref_values.e_ev.min(), self.ref_values.e_ev.max()])
            ax.set_title("Energy dependency of {}/{}".format(*lbl))
            ax.legend(loc=1)
            f.tight_layout()
            if printing:
                f.savefig(os.path.join(self.save_folder,
                                       "{}_{}_{:02.02e}_dependence_on_energy.pdf".format(on,
                                                                                         self.lam,
                                                                                         self.i_w_cm_2[-1])),
                          dpi=300)
                plt.close(f)
            else:
                return f

    def plot_dependence_on_intensity(self,
                                     on,
                                     arg="real",
                                     n=(13, 15),
                                     ev=None,
                                     ints=(0, -1),
                                     uncertainty=.5,
                                     printing=False):
        """
        :param on: Can be 'a' (alpha), 'n' (refractive index) or 'int' (Brightness on detector) (Case-insensitive)
        :param arg: Can be 'real', 'imag', 'abs'
        :param n: The harmonic with which we probe
        :param ev: Alternatively to 'n' you can pass your own energy of interest in eV
        :param ints: List of intensity to plot. Default is: (0, -1) .. This is (start, end) idx
        :param uncertainty: Energy uncertainty (in eV) ... final output will be averaged across this energy range
        :return: Matplotlib figure
        """
        if printing:
            context = plt.rc_context({'font.size': 26,
                                      "text.usetex": True,
                                      "text.latex.preamble": r'\usepackage{mathpazo}'})
        else:
            context = nullcontext()

        on = on.lower()
        arg = arg.lower()
        assert on in ["a", "n", "int"]
        assert arg in ["real", "imag", "abs"]

        if arg == "real":
            arg_fun = np.real
            if on == "a":
                lbl = r"$\alpha_1$"
            else:
                lbl = r"$\delta$"
        elif arg == "imag":
            arg_fun = np.imag
            if on == "a":
                lbl = r"$\alpha_2$"
            else:
                lbl = r"$\beta$"
        elif arg == "abs":
            arg_fun = np.abs
            if on == "a":
                lbl = r"|$\alpha$|"
            else:
                lbl = "|n|"

        if ev is None:
            assert n is not None
            energy_ev = self.calc_harmonic(n)
        else:
            energy_ev = ev

        if on == "a":
            data = self.a_interp[ints[0]:ints[1]] / self.au_p  # in a.u.
        else:
            data = self.n_interp[ints[0]:ints[1]]
        if on == "int":
            x = 1
        else:
            data = arg_fun(data)

        with sns.axes_style("whitegrid"), context:
            if printing:
                f, ax = plt.subplots(1, 1, figsize=(32, 16))
            else:
                f, ax = plt.subplots(1, 1, figsize=(16, 8))

            for ii, ev_ in enumerate(energy_ev):
                # Get the idx of the desired energy
                idx = int(sum(self.interp_grid_ev < ev_))
                if uncertainty == 0:
                    y = np.array([x[idx] for x in data])
                else:
                    lo = self.interp_grid_ev.min()
                    hi = self.interp_grid_ev.max()
                    res = len(self.interp_grid_ev)
                    _idx = int(res * (uncertainty / 2) / (np.abs(hi - lo)))
                    y = np.array([x[idx - _idx:idx + _idx].mean() for x in data])

                lbl_ = r"{} | Energy {:02.02f} eV".format(lbl, ev_)
                if n is not None:
                    lbl_ += r" | {}th Harmonic".format(n[ii])
                if uncertainty > 0:
                    lbl_ += r" | Averaged over $\pm${:01.02f} eV".format(uncertainty / 2)

                if on == "n" and arg == "real":
                    y = 1 - y
                ax.plot(self.i_w_cm_2[ints[0]:ints[1]], y,
                        label=lbl_,
                        linewidth=1)
            ax.set_xlabel(r"Intensity [$W/cm^2$]")
            ax.set_title("IR intensity dependency of {}".format(lbl))
            ax.legend(loc=1)
            f.tight_layout()
            if printing:
                f.savefig(os.path.join(self.save_folder,
                                       "{}_{}_{}_{:02.02e}_dependence_on_intensity.pdf".format(on, arg,
                                                                                               self.lam,
                                                                                               self.i_w_cm_2[-1])),
                          dpi=300)
                plt.close(f)
            else:
                return f

    def plot_real_imag_scan(self, on, cmap="jet", n=(13, 15), ev=None, uncertainty=.5, printing=False):
        """
        Highly specific method for plotting the real/imag scans

        :param on: Can be 'a' (alpha) or 'n' (refractive index). Both are case-insensitive
        :param cmap: The cmap for both axes
        :param n: The harmonic we want to draw in the contour plot
        :param ev: Alternatively to 'n' you can pass your own energy of interest in eV
        :param uncertainty: Energy uncertainty (in eV) ... final output will be averaged across this energy range
        :return: Matplotlib figure
        """
        if printing:
            context = plt.rc_context({'font.size': 26,
                                      "text.usetex": True,
                                      "text.latex.preamble": r'\usepackage{mathpazo}'})
        else:
            context = nullcontext()

        if n is not None or ev is not None:  # only on arg can be not None
            assert (n is None and ev is not None) or (n is not None and ev is None)

        if n is not None:
            energy_ev = self.calc_harmonic(n)
            if isinstance(n, int):
                n = [n]
                energy_ev = [energy_ev]
            lbl = ["{} th Harmonic ({:02.02f} eV)".format(x, y) for x, y in zip(n, energy_ev)]
        elif ev is not None:
            energy_ev = ev
            if isinstance(energy_ev, int):
                energy_ev = [energy_ev]
            lbl = ["{:02.02f} eV".format(y) for x, y in energy_ev]
        else:
            energy_ev = None

        on = on.lower()
        assert on in ["a", "n"]

        if on == "a":
            y = self.a_interp / self.au_p  # in a.u.
            ttl_ = (r"$\alpha_1$", r"$\alpha_2$")
        else:
            y = self.n_interp
            ttl_ = (r"$\delta$", r"$\beta$")
        ttl = r"{} for $\lambda$: {}nm".format("{}/{}".format(*ttl_), self.lam)

        with context:
            if printing:
                f, ax = plt.subplots(1, 2, figsize=(36, 16))
            else:
                f, ax = plt.subplots(1, 2, figsize=(18, 8))

            real_h = ax[0].imshow(y.real if on == "a" else (1 - y.real), cmap=plt.cm.get_cmap(cmap))
            imag_h = ax[1].imshow(y.imag, cmap=plt.cm.get_cmap(cmap))

            x_ticks = np.arange(int(len(self.interp_grid_ev)))[::int(len(self.interp_grid_ev) / 5)]
            x_tick_labels = self.interp_grid_ev[::int(len(self.interp_grid_ev) / 5)]

            y_ticks = np.arange(int(len(self.i_w_cm_2)))[::int(len(self.i_w_cm_2) / 5)]
            y_tick_labels = self.i_w_cm_2[::int(len(self.i_w_cm_2) / 5)] * 1e-12

            for ax_ in ax:
                ax_.set_aspect(y.shape[1] / y.shape[0])
                ax_.set_xlabel("Energy [eV]")
                ax_.set_ylabel(r"Intensity [$W/cm^{2}$] [$\cdot 10^{12}$]")
                ax_.set_xticks(x_ticks)
                ax_.set_xticklabels(["{:02.02f}".format(x) for x in x_tick_labels])
                ax_.set_yticks(y_ticks)
                ax_.set_yticklabels(["{:02.02f}".format(y) for y in y_tick_labels])
                if energy_ev is not None:
                    ee = [int(sum(self.interp_grid_ev < x)) for x in energy_ev]
                    for i, (e, l) in enumerate(zip(ee, lbl)):
                        if uncertainty:
                            lo = self.interp_grid_ev.min()
                            hi = self.interp_grid_ev.max()
                            res = len(self.interp_grid_ev)
                            _idx = int(res * (uncertainty / 2) / (np.abs(hi - lo)))
                            ax_.axvline(e - _idx, color=sns.color_palette("bright", 8)[2 + i],
                                        linewidth=1, linestyle="dashed")
                            ax_.axvline(e + _idx, color=sns.color_palette("bright", 8)[2 + i],
                                        linewidth=1, linestyle="dashed")
                        ax_.axvline(e, label=l, color=sns.color_palette("bright", 8)[2 + i],
                                    linewidth=2, linestyle="solid")
                    ax_.legend()
            ax[0].set_title(ttl_[0])
            ax[1].set_title(ttl_[1])
            f.suptitle(ttl)
            f.colorbar(real_h, ax=ax[0], shrink=.75)
            f.colorbar(imag_h, ax=ax[1], shrink=.75)
            f.tight_layout()

            if printing:
                f.savefig(os.path.join(self.save_folder,
                                       "{}_{}_{:02.02e}_real_imag_scan.pdf".format(on,
                                                                                   self.lam,
                                                                                   self.i_w_cm_2[-1])),
                          dpi=300)
                plt.close(f)
            else:
                return f

    def plot_scat_images(self,
                         rads,
                         n=None,
                         ev=None,
                         ints=(None, None),
                         use_log=True,
                         max_angle=None,
                         r=None,
                         detector_hole=True,
                         printing=False):
        """

        :param rads: The Radial profiles
        :param n: (Optional) The harmonic with which we probe
        :param ev: (Optional) Alternatively to 'n' you can pass your own energy of interest in eV
        :param ints: (Optional) IR intensities that correspond to the rad profiles
        :param use_log: (Optional) (Bool) Plot in log color scale
        :param max_angle: (Optional) For the title ... Max scattering angle
        :param r: (Optional) For the title ... Radius of the droplet
        :param detector_hole: (Bool) Simulate a detector hole
        :return: Matplotlib figure
        """
        if printing:
            context = plt.rc_context({'font.size': 26,
                                      "text.usetex": True,
                                      "text.latex.preamble": r'\usepackage{mathpazo}'})
        else:
            context = nullcontext()
        if n is not None:
            energy_ev = self.calc_harmonic(n)
        else:
            energy_ev = ev

        if ev is None and n is None:
            assert energy_ev is None

        assert len(rads) == 2  # You have to pass exactly two profiles
        assert len(rads) == len(ints)
        if detector_hole:
            rads = np.array(rads)
            rads[:, :50] = 0  # This corresponds to ~ 3°

        with sns.axes_style("white"), context:
            if printing:
                f, ax = plt.subplots(1, 2, figsize=(32, 17.5))
            else:
                f, ax = plt.subplots(1, 2, figsize=(16, 8.75))

            img0, ma, mi = self.get_diffraction(rads[0], use_log)
            img1, ma_2, mi_2 = self.get_diffraction(rads[1], use_log)

            global_max = np.max([ma, ma_2])
            global_min = np.max([mi, mi_2])

            ints_ = self.i_w_cm_2[np.array(ints)]
            for a, img_, i, ra in zip(ax, (img0, img1), ints_, rads):
                a.imshow(img_, vmin=mi, vmax=ma)
                a.set_xticks([])
                a.set_yticks([])
                if use_log:
                    int_ = np.exp(img_).sum() / np.exp(img0).sum()
                else:
                    int_ = img_.sum() / img0.sum()
                ttl = "Brightness: {:.02%}%".format(int_)
                if i is not None:
                    ttl += r" | IR Intensity: {:02.02e} $W/cm^2$".format(i)
                if energy_ev is not None:
                    ttl += " | Probing with: {:02.02f} eV".format(energy_ev)
                    if n is not None:
                        ttl += " ({}th harm.)".format(n)
                else:
                    n = (13, 15)
                    e_ = []
                    for n_ in n:
                        e_.append(self.calc_harmonic(n_))
                    ttl += " | Probing with (13th {:02.02f} and 15th {:02.02f} harm.)".format(*e_)

                a.set_title(ttl)
            ttl = "Diffraction pattern ({} cmap)".format("log." if use_log else "linear")
            if max_angle is not None or r is not None:
                ttl += " | "
            if max_angle is not None:
                ttl += r"Max scat. angle: {}°".format(max_angle)
            if r is not None:
                if max_angle is not None:
                    ttl += " | "
                ttl += "Droplet Radius: {} nm".format(r)
            f.suptitle(ttl)
            f.tight_layout()
            f.subplots_adjust(top=0.96,
                              bottom=0.0,
                              left=0.012,
                              right=0.988,
                              hspace=0.2,
                              wspace=0.029)
            if printing:
                f.savefig(os.path.join(self.save_folder,
                                       "{}_{}_{:02.02e}_scat_images.pdf".format(["{:02.02f}".format(x) for x in n],
                                                                                self.lam,
                                                                                self.i_w_cm_2[-1])),
                          dpi=300)
                plt.close(f)
            return int_

    def plot_intensity_waterfall(self, on, max_int=20, plot_energy_levels=True,
                                 min_idx=8000, max_idx=15000, cmap="turbo",
                                 printing=False):
        """

        :param max_int:
        :param plot_energy_levels:
        :param cmap:
        :return:
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.collections import PolyCollection
        if printing:
            context = plt.rc_context({'font.size': 26,
                                      "text.usetex": True,
                                      "text.latex.preamble": r'\usepackage{mathpazo}'})
        else:
            context = nullcontext()

        if cmap.lower() == "turbo":
            plt.register_cmap(name='turbo', data=get_turbo(), lut=256)

        if on == "a":
            y = self.a_interp
        else:
            y = self.n_interp

        cmap = plt.cm.get_cmap(cmap)
        color_picker = np.linspace(0, 255, max_int).astype(int)
        colors = [cmap(i) for i in color_picker]

        with context:
            if printing:
                fig = plt.figure(figsize=(32, 16))
            else:
                fig = plt.figure(figsize=(12, 8))
            ax = fig.gca(projection='3d')
            if on == "a":
                # See equation (22) in:
                # Gaarde, M. B., Buth, C., Tate, J. L., & Schafer, K. J. (2011).
                # Transient absorption and reshaping of ultrafast XUV light by laser-dressed helium.
                # Physical Review A - Atomic, Molecular, and Optical Physics, 83(1).
                ww = 2 * np.pi / self.period * self.t_vec / self.au_time
                const = 4 * np.pi * ww / (cs.c * cs.epsilon_0)  # / self.fine_struct

            xs = self.interp_grid_ev[min_idx:max_idx]
            verts = []
            y_max = 0
            for ii in range(max_int):
                if on == "a":
                    ys = const[ii] * y[ii][min_idx:max_idx].imag * 1e22
                else:
                    ys = y[ii][min_idx:max_idx].imag
                if ys.max() > y_max:
                    y_max = ys.max()
                ys[ys < 0] = 0
                ys[0], ys[-1] = (0, 0)
                verts.append(list(zip(xs, ys)))
            poly = PolyCollection(verts, facecolor=colors)

            poly.set_alpha(1)
            ax.add_collection3d(poly, zs=np.linspace(-1, max_int, max_int) - .5, zdir='y')

            ax.set_xlabel("Energy [eV]", labelpad=30)
            ax.set_xlim3d(xs.min(), xs.max())

            if on == "a":
                ax.set_zlabel("Absorption cross section [Mbarn]", labelpad=75)
            else:
                ax.set_zlabel("Absorption", labelpad=75)
            ax.set_zlim3d(0, y_max)

            ax.set_ylabel(r"IR Intensity $W/cm^2$", labelpad=90)
            ax.set_ylim3d(0, max_int)
            yticks = np.linspace(0, max_int, 5).astype(int)
            yticklabels = self.i_w_cm_2[yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(["{:02.02e}".format(x) for x in yticklabels])

            ax.view_init(elev=47, azim=-90.1)

            if plot_energy_levels:
                harm_13 = self.calc_harmonic(13)
                harm_15 = self.calc_harmonic(15)
                ax.plot([harm_13] * 2, [-.5, 19.5], [0] * 2, color="tab:red", linestyle="-",
                        label="13th Harmonic ({:02.02f} eV)".format(harm_13))
                ax.plot([harm_15] * 2, [-.5, 19.5], [0] * 2, color="tab:blue", linestyle="-",
                        label="15th Harmonic ({:02.02f} eV)".format(harm_15))
                ax.plot([20.61577498] * 2, [-.5, 19.5], [0] * 2, color="black", linestyle="--",
                        label="1s2s (20.62 eV)")
                ax.plot([21.218022851325713] * 2, [-.5, 19.5], [0] * 2, color="tab:green", linestyle="--",
                        label="1s2p (21.22 eV)")
                ax.plot([22.9203175] * 2, [-.5, 19.5], [0] * 2, color="black", linestyle="--",
                        label="1s3s (22.92 eV)")
                ax.plot([23.07407494] * 2, [-.5, 19.5], [0] * 2, color="black", linestyle="--",
                        label="1s3d (23.07 eV)")
                ax.plot([23.087018663345155] * 2, [-.5, 19.5], [0] * 2, color="tab:green", linestyle="--",
                        label="1s3p (23.09 eV)")
                ax.plot([23.742070185580754] * 2, [-.5, 19.5], [0] * 2, color="tab:green", linestyle="--",
                        label="1s4p (23.74 eV)")

            ax.tick_params(axis='y', which='major', pad=40)
            ax.tick_params(axis='z', which='major', pad=30)
            ax.tick_params(axis='x', which='major', pad=10)
            ax.legend()
            fig.tight_layout()

            if printing:
                fig.savefig(os.path.join(self.save_folder,
                                         "{}_{}_{:02.02e}_intensity_waterfall.pdf".format(on,
                                                                                          self.lam,
                                                                                          self.i_w_cm_2[-1])),
                            dpi=300)
                plt.close(fig)
            else:
                return fig

    def get_single_rad(self, x, energy_ev, r, ir_idx):
        energy_nm = self.ev_to_nm(energy_ev)
        mu = np.cos(x * np.pi / 180)
        scaling_factor = 2.1792037974346345  # make it comparable to Metzler
        idx_n = sum(self.interp_grid_ev < energy_ev)
        m_ = (self.n_interp.real[ir_idx][idx_n] - 1j * self.n_interp.imag[ir_idx][idx_n])
        size_param = (2 * np.pi * r) / energy_nm
        return mp.i_per(m_, size_param, mu) * scaling_factor
