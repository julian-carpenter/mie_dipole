import os
import numpy as np
import scipy.constants as cs
from tqdm import trange
from scipy.special import expit
import miepython as mp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
from utils.ref_literature import REFERENCE

np.seterr(divide='ignore', invalid='ignore')  # there will be some 'divide by 0' warnings


class TdseMie(object):
    def __init__(self,
                 data_path,
                 lam=800,
                 max_int=20,
                 step_size=6.25e-4):
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
        self.au_efield = cs.physical_constants["atomic unit of electric field"][0]
        self.au_p = cs.physical_constants["atomic unit of electric polarizability"][0]
        self.au_pot = cs.physical_constants["atomic unit of electric potential"][0]
        self.a0 = cs.physical_constants["Bohr radius"][0]
        self.nd = 0.022e30  # number density of liquid helium
        self.ref_res = 250  # the resolution for the interpolation of the reference values
        self.ref_values = REFERENCE(self.ref_res)

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

        # Check that A and d has dimensions (max_int, n)
        # if max_int == 1 we would end up with an array of dim n, since
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

    def apply_envelope(self, on="A", width=10, center="max", sign=-1):
        """
        In place application of a sigmoidal envelope.
        'on' has to be 'A' or 'd' (not case-sensitive).
        'width' is the tau parameter in the sigmoid, given in fs.
        'center' is where the sigmoid should be 1/2 of the
        signal can be 'max' or a specific position in 'index'.
        """
        on = on.lower()
        assert on in ("a", "d")
        ind_width = self.fs_to_index(width)
        self.env = self.get_envelope(ind_width, center, sign)
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
        d_au_env_fft = np.fft.ifft(self.d_in_au, axis=1)
        a_au_env_fft = np.fft.ifft(self.A_in_au, axis=1)

        # ALPHA | alpha is d(w) / ( i * dw * A(w) )
        # self.alpha = 2 * np.divide(d_au_env_fft, (1j * self.dw_au * a_au_env_fft))
        self.alpha = 2 * np.pi * np.divide(d_au_env_fft, (1j * self.dw_au * a_au_env_fft))
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

        if self.alpha is None:
            self.get_alpha()

        self.n = calc_n(self.a_si)

        # FOR THE INTERPOLATED ALPHA
        self.n_interp = calc_n(self.a_interp)

        return self.n, self.n_interp

    def get_envelope(self, width, center, sign=-1, signal_size=None):
        """
        Construct a sigmoid that serves as an envelope.
        'width' is in 'index' and 'center' can be 'max' or in 'index'.
         Optional: 'signal_size' is the length of the signal.
        """
        if center == "max":
            # c_ = int(18363)  # this was checked manually ... so be careful here
            c_ = int(18222)  # this is what Thomas & Bjorn use
        else:
            c_ = center
        if signal_size is None:
            x = np.arange(self.time_in_au.size)
        else:
            x = np.arange(signal_size)
        env = [expit(sign * (x - c_) / width)] * self.A_in_au.shape[0]
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

    def radial_profiles(self, x, r, n=None, ev=None, normalize=True, ir_idx=-1):
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
        if ev is None:
            assert n is not None
            energy_ev = self.calc_harmonic(n)
        else:
            energy_ev = ev

        energy_nm = self.ev_to_nm(energy_ev)
        mu = np.cos(x * np.pi / 180)
        scaling_factor = 2.1792037974346345  # make it comparable to Metzler

        idx_n = sum(self.e_ev < energy_ev)

        m_ = (self.n_interp.real[0][idx_n] - 1j * self.n_interp.imag[0][idx_n])
        size_param = (2 * np.pi * r) / energy_nm
        y1 = mp.i_per(m_, size_param, mu) * scaling_factor

        m_ = (self.n_interp.real[ir_idx][idx_n] - 1j * self.n_interp.imag[ir_idx][idx_n])
        size_param = (2 * np.pi * r) / energy_nm
        y2 = mp.i_per(m_, size_param, mu) * scaling_factor

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
        with sns.axes_style("whitegrid"):
            f_rp, ax_rp = plt.subplots(1, 1, figsize=(16, 8))
            ax_rp.semilogy(x, y1, label=lbl1, linewidth=1)
            ax_rp.semilogy(x, y2, label=lbl2, linewidth=1)
            ax_rp.set_xlabel("Scattering Angle [°]")
            ttl = "Radial profiles (Orth. pol.) | Change in brightness when IR is present: {:02.02f}%".format(
                100 * (np.sum(y2) - np.sum(y1)) / np.sum(y1))
            if normalize:
                ttl += " | Normalized using the int. at IR=0"
            ax_rp.set_title(ttl)
            ax_rp.legend()
            f_rp.tight_layout()
        return f_rp

    def plot_dependence_on_energy(self, on, ints=(0, -1), include_ref=False, n=(13, 15), ev=None, uncertainty=.5):
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

        with sns.axes_style("whitegrid"):
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
            ax.legend()
            f.tight_layout()
        return f

    def plot_dependence_on_intensity(self,
                                     on,
                                     arg="real",
                                     n=(13, 15),
                                     ev=None,
                                     ints=(0, -1),
                                     uncertainty=.5):
        """
        :param on: Can be 'a' (alpha) or 'n' (refractive index). Both are case-insensitive
        :param arg: Can be 'real', 'imag', 'abs'
        :param n: The harmonic with which we probe
        :param ev: Alternatively to 'n' you can pass your own energy of interest in eV
        :param ints: List of intensity to plot. Default is: (0, -1) .. This is (start, end) idx
        :param uncertainty: Energy uncertainty (in eV) ... final output will be averaged across this energy range
        :return: Matplotlib figure
        """
        on = on.lower()
        arg = arg.lower()
        assert on in ["a", "n"]
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

        if on == "a":
            data = self.a_interp[ints[0]:ints[1]] / self.au_p  # in a.u.
        else:
            data = self.n_interp[ints[0]:ints[1]]
        data = arg_fun(data)

        with sns.axes_style("whitegrid"):
            f, ax = plt.subplots(1, 1, figsize=(16, 8))

            if ev is None:
                assert n is not None
                energy_ev = self.calc_harmonic(n)
            else:
                energy_ev = ev

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
            ax.legend()
            f.tight_layout()
        return f

    def plot_real_imag_scan(self, on, cmap="jet", n=(13, 15), ev=None, uncertainty=.5):
        """
        Highly specific method for plotting the real/imag scans

        :param on: Can be 'a' (alpha) or 'n' (refractive index). Both are case-insensitive
        :param cmap: The cmap for both axes
        :param n: The harmonic we want to draw in the contour plot
        :param ev: Alternatively to 'n' you can pass your own energy of interest in eV
        :param uncertainty: Energy uncertainty (in eV) ... final output will be averaged across this energy range
        :return: Matplotlib figure
        """
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
                        ax_.axvline(e - _idx, color=sns.color_palette()[i],
                                    linewidth=1, linestyle="dashed")
                        ax_.axvline(e + _idx, color=sns.color_palette()[i],
                                    linewidth=1, linestyle="dashed")
                    ax_.axvline(e, label=l, color=sns.color_palette()[i],
                                linewidth=2, linestyle="solid")
                ax_.legend()
        ax[0].set_title(ttl_[0])
        ax[1].set_title(ttl_[1])
        f.suptitle(ttl)
        f.colorbar(real_h, ax=ax[0], shrink=.75)
        f.colorbar(imag_h, ax=ax[1], shrink=.75)
        f.tight_layout()
        return f
