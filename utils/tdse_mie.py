import os
import numpy as np
import scipy.constants as cs
from tqdm import trange
from scipy.special import expit
import miepython as mp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy import integrate
from scipy import signal
from scipy import optimize
from scipy import stats
from scipy.interpolate import interp1d
from contextlib import nullcontext
from utils.ref_literature import REFERENCE, get_turbo

np.seterr(divide="ignore", invalid="ignore")  # there will be some "divide by 0" warnings
# plt.register_cmap(name='turbo', cmap=get_turbo())


class TdseMie(object):
    def __init__(self,
                 data_path,
                 lam=798,
                 max_int=20,
                 step_size=6.25e-4,
                 save_folder=".",
                 only_max_int=False,
                 nlevel=False,
                 cut_absorption=False,
                 iap=True):
        # CONSTANTS
        self.alpha = None  # Placeholder
        self.n = None  # Placeholder
        self.n_interp = None  # Placeholder
        self.a_interp = None  # Placeholder
        self.interp_grid_ev = None  # Placeholder
        self.a_si = None  # Placeholder
        self.env = 1.  # broadcastable dummy for the sigmoidal envelope
        self.lam = lam  # in nm
        self.max_int = max_int  # in Marc"s units
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
        # self.nd = 0.04e30  # number density of liquid helium
        self.nd = 0.022e30  # number density of liquid helium
        self.ref_res = 250  # the resolution for the interpolation of the reference values
        self.ref_values = REFERENCE(self.ref_res)
        self.save_folder = save_folder
        self.nlevel = nlevel
        self.iap = iap
        self.cut_absorption = cut_absorption

        if not os.path.isdir(self.save_folder):
            self.save_folder = "."

        # LOAD THE DATA
        if self.nlevel:
            self.E_in_au = []
        else:
            self.A_in_au = []
        self.d_in_au = []

        if only_max_int:
            context = (0, max_int)
            pargs = (2, self.lam)
        else:
            pargs = (self.max_int + 1, self.lam)
            desc = "Reading in {} intensities for {}nm".format(*pargs)
            context = trange(pargs[0], desc=desc)

        for i_int in context:
            if self.nlevel:
                if self.iap:
                    ff = "dip{}_{}_15.npy".format(i_int, self.lam)
                else:
                    ff = "dip{}_{}_13_15_20fs_deact_states.npy".format(i_int, self.lam)
                    # ff = "dip{}_{}_13_15_20fs.npy".format(i_int, self.lam)
            else:
                ff = "dip{}_{}.dat".format(i_int, self.lam)
            if self.nlevel:
                data = np.load(os.path.join(data_path, ff))  # 0: t; 1:A(t); 2:d(t) | all in a.u.
            else:
                data = np.loadtxt(os.path.join(data_path, ff))  # 0: t; 1:A(t); 2:d(t) | all in a.u.
            data = np.nan_to_num(data)

            if self.nlevel:
                if i_int == 0:
                    self.time_in_au = data[:, 0]
                self.E_in_au.append(data[:, 1].squeeze())
            else:
                if i_int == 0:
                    self.time_in_au = -data[:, 0]
                self.A_in_au.append(data[:, 1].squeeze())
            self.d_in_au.append(data[:, 2].squeeze())

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
        if self.nlevel:
            self.E_in_au = np.array(self.E_in_au)
            try:
                assert len(self.E_in_au.shape) == 2
            except AssertionError as e:
                print("{}\n Adding an extra dimension to E".format(e))
                self.E_in_au = np.expand_dims(self.E_in_au, axis=0)
        else:
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

    def apply_envelope(self, on="A", width=10, center="max", sign=-1, use_hann=True, optimize_window=False):
        """
        In place multiplication of a sigmoidal envelope.
        "on" has to be "A" or "d" (not case-sensitive).
        "width" is the tau parameter in the sigmoid, given in fs.
        "center" is where the hann window should be centered in the
        signal can be "max" or a specific position in "index".
        """
        on = on.lower()
        assert on in ("a", "d")
        best = 0
        if optimize_window:

            def calc_n(a):  # Clausius-Mossotti-Relation
                frac = np.divide(3 * self.nd * a, 3 * self.e0 - self.nd * a)
                return np.sqrt(1 + frac)

            int_ref = np.trapz(self.ref_values.beta[50:150],
                               x=self.ref_values.e_ev[50:150])
            mu_, std_ = stats.norm.fit(self.ref_values.beta[50:150])
            ref_loss_ = std_ + int_ref + .75*np.max(self.ref_values.beta[50:150])
            argss_ = (std_, int_ref, np.max(self.ref_values.beta[50:150]),
                      ref_loss_)
            # print("{:02.02f}, {:02.02f}, {:02.02f}, {:02.02f}".format(*argss_))

            # import matplotlib
            # matplotlib.use("qt5agg")
            # f, ax1 = plt.subplots(1, 1, figsize=(12, 6))
            # ax2 = ax1.twiny()
            tmp_ = 100
            for w in np.linspace(1, 15, 50):
                ind_width = self.fs_to_index(w)
                self.env = self.get_envelope(ind_width, center, sign, use_hann=use_hann)
                low = self.ref_values.e_ev[50]  # lower energy limit in eV
                high = self.ref_values.e_ev[150]  # upper energy limit in eV
                res = 25000  # number of values to interpolated between 'lo' and 'hi'

                dd = np.copy(self.d_in_au[0] * self.env[0])
                d_au_env_fft = 2 * np.fft.ifft(dd)
                aa = np.copy(self.A_in_au[0] * self.env[0])
                a_au_env_fft = np.fft.ifft(aa)
                alpha = np.divide(2 * d_au_env_fft, (1j * self.dw_au * a_au_env_fft))
                interp_grid_ev = np.linspace(low, high, res)
                a_si = alpha * self.au_p
                a_interp = np.interp(interp_grid_ev, self.e_ev, a_si)

                n_interp = calc_n(a_interp)

                # Calculate the integral. First part of loss
                int_n = np.trapz(n_interp.imag, x=interp_grid_ev)
                # print(int_ref, int_n)
                # Calculate STD, second part of loss
                # 1) Fit a normal distribution to the data:
                mu, std = stats.norm.fit(n_interp.imag)
                cond = (std + int_n + .75*np.max(n_interp.imag)) - ref_loss_
                argss = (w, std, int_n, np.max(n_interp.imag), std + int_n + .75*np.max(n_interp.imag), cond)
                # print("{:02.02f}, {:02.02f}, {:02.02f}, {:02.02f}, {:02.02f}, {:02.02f}".format(*argss))
                if np.abs(cond) < tmp_:
                    tmp_ = np.abs(cond)
                    best = w
            print(r"Best estimate for $\tau$: {:02.04f}".format(best))

            #     ax1.plot(interp_grid_ev, n_interp.imag,
            #              label=str(w) + " {:02.02f}, {:02.02f}".format(int_n, std))
            #     x = stats.norm.ppf(np.linspace(.05, .95, len(interp_grid_ev)), mu, std)
            #     ax2.plot(x, stats.norm.pdf(x, mu, std), label=str(w) + "_fit")
            #     if w == 7:
            #         ax1.plot(self.ref_values.e_ev[50:150],
            #                  self.ref_values.beta[50:150],
            #                  label="Ref" + " {:02.02f}".format(int_ref))
            #         print(mu_, std_)
            #         x = stats.norm.ppf(np.linspace(.05, .95, len(self.ref_values.e_ev[50:150])), mu_, std_)
            #         ax2.plot(x,
            #                  stats.norm.pdf(x, mu_, std_),
            #                  label="Ref_fit {:02.02f}".format(std_))
            # f.legend()
            # plt.show()
            # matplotlib.use("agg")

            ind_width = self.fs_to_index(best)
            self.env = self.get_envelope(ind_width, center, sign, use_hann=use_hann)
        else:
            ind_width = self.fs_to_index(width)
            self.env = self.get_envelope(ind_width, center, sign, use_hann=use_hann)
        if on == "a" or on == "e":
            if self.nlevel:
                self.E_in_au *= self.env
            else:
                self.A_in_au *= self.env
        elif on == "d":
            self.d_in_au *= self.env

        return best if best > 0 else width

    def get_alpha(self, low=15, high=30, res=25000, subtract_mean=False):
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

        # d_au_env_fft = np.fft.ifft(self.d_in_au, axis=1)
        d_au_env_fft = 2 * np.fft.ifft(self.d_in_au, axis=1)
        if self.nlevel:
            # ALPHA | alpha is d(w) / E(w)
            e_au_env_fft = np.fft.ifft(self.E_in_au, axis=1)
            self.alpha = np.divide(d_au_env_fft, e_au_env_fft)
        else:
            # ALPHA | alpha is d(w) / ( i * dw * A(w) )
            a_au_env_fft = np.fft.ifft(self.A_in_au, axis=1)
            self.alpha = np.divide(2 * d_au_env_fft, (1j * self.dw_au * a_au_env_fft))

        # INTERPOLATION
        # alpha
        self.interp_grid_ev = np.linspace(low, high, res)
        self.a_si = self.alpha * self.au_p
        if self.nlevel:
            self.a_interp = np.array(
                list(map(lambda fp: interp1d(self.e_ev, fp, kind="quadratic")(self.interp_grid_ev), self.a_si)))
        else:
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
        cut = self.ref_values.beta.min()
        if self.cut_absorption:
            self.n_interp.imag[self.n_interp.imag < cut] = cut
        else:
            self.n_interp.imag[0][self.n_interp.imag[0] < cut] = cut # unperturbed is always clipped

        return self.n, self.n_interp

    def get_envelope(self, width, center, sign=-1, signal_size=None, use_hann=True):
        """
        Construct a sigmoid that serves as an envelope.
        "width" is in "index" and "center" can be "max" or in "index".
         Optional: "signal_size" is the length of the signal.
        """

        if center == "max":
            if self.nlevel:
                if self.iap:
                    c_ = int(23000)  # manually checked
                else:
                    c_ = int(44150)  # manually checked
            else:
                c_ = int(18408)  # this was checked manually ... so be careful here
            # c_ = int(18222)  # this is what Thomas & Bjorn use
        else:
            c_ = center
        if signal_size is None:
            x = np.arange(self.time_in_au.size)
        else:
            x = np.arange(signal_size)

        if self.nlevel:
            if self.iap:
                mult = .3
                denom = 5
            else:
                mult = .07125
                denom = 7
            env = np.zeros(self.d_in_au.shape[1])
            env[:c_] = signal.windows.gaussian(int(2 * c_),
                                               mult * c_)[:c_]
            res_len = self.time_in_au.size - c_
            env[c_:] = signal.windows.gaussian(int(2 * res_len),
                                               c_ / denom)[res_len:]
        else:
            if use_hann:
                env = np.zeros(self.d_in_au.shape[1])
                env[c_ - width:c_ + width] = signal.windows.hann(int(2 * width))
                # print("HANN")
            else:
                # print("EXPIT")
                env = expit(sign * (x - c_) / width)

        env = [env] * self.d_in_au.shape[0]
        return np.array(env)

    def fs_to_au(self, fs):
        return fs / (self.au_time * 1e15)

    def fs_to_index(self, fs, rel_to_max=False):
        """
        Convert a given time in fs to the corresponding index in the data
        if rel_to_max is True then fs is relative to the mean of all max"s of A
        """
        if rel_to_max:  # get the mean of all max"s of A
            max_ = 0
            for a in self.d_in_au:
                max_ += self.index_to_fs(np.argmax(a))
            max_ /= self.d_in_au.shape[0]
            fs = max_ + fs
        return int(sum(self.time_in_au < self.fs_to_au(fs)))

    def au_to_fs(self, au):
        return au * self.au_time * 1e15

    def au_to_index(self, au, rel_to_max=False):
        """
        Convert a given time in au to the corresponding index in the data
        if rel_to_max is True then au is relative to the mean of all max"s of A
        """
        if rel_to_max:
            max_ = 0
            for a in self.d_in_au:
                max_ += self.index_to_au(np.argmax(a))
            max_ /= self.d_in_au.shape[0]
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

    def radial_profiles(self, x, r, n=None, ev=None,
                        normalize=False, ir_idx=-1,
                        printing=False, just_profile=False):
        """

        :param x: The scattering angle of interest (e.g. linspace from 0 to 30) [°]
        :param r: The radius of the droplet
        :param n: The harmonic with which we probe
        :param ev: Alternatively to "n" you can pass your own energy of interest in eV
        :param normalize: (Bool) If True the brightest profile is set to peak at 1 and all other profiles
                          are normalized with the same value.
        :param ir_idx: Idx (In Marc"s # index) to which the IR=0 profile should be compared (Default = -1)
        :return: Matplotlib figure
        """
        if printing:
            context = plt.rc_context({"font.size": 36,
                                      "text.usetex": True,
                                      "text.latex.preamble": r"\usepackage{mathpazo}"})
        else:
            context = nullcontext()

        if ev is None:
            assert n is not None
            energy_ev = self.calc_harmonic(n)
        else:
            energy_ev = ev

        y1 = self.get_single_rad(x, energy_ev, r, 0)
        y2 = self.get_single_rad(x, energy_ev, r, ir_idx)

        if just_profile:
            return y1, y2

        if normalize:
            norm = integrate.trapz(y1, x)
            y1 /= norm
            y2 /= norm

        lbl1 = r"Radial profile - Probing with: {:02.02f} eV without IR".format(energy_ev)
        lbl2 = r"Radial profile - Probing with: {:02.02f} eV at IR: {:02.02e} $W/cm^2$".format(energy_ev,
                                                                                               self.i_w_cm_2[
                                                                                                   ir_idx])
        if n is not None:
            lbl1 += " - {}th. Harmonic".format(n)
            lbl2 += " - {}th. Harmonic".format(n)
        with sns.axes_style("whitegrid"), context:
            if printing:
                f_rp, ax_rp = plt.subplots(1, 1, figsize=(28, 14))
            else:
                f_rp, ax_rp = plt.subplots(1, 1, figsize=(16, 8))

            ax_rp.semilogy(x, y1, label=lbl1, linewidth=3)
            ax_rp.semilogy(x, y2, label=lbl2, linewidth=3)
            ax_rp.set_xlabel("Scattering Angle [°]")
            ttl = "Decrease in brightness with IR: {:02.02f}%".format(
                100 * np.sum(y2) / np.sum(y1))
            if normalize:
                ttl += " | Normalized using the int. at IR=0"
            # ax_rp.set_title(ttl)
            bbox = dict(boxstyle="round", fc="w", ec="w", alpha=.75)
            # ax_rp.text(15, 5, ttl, bbox=bbox)
            ax_rp.legend()
            ax_rp.set_yticklabels(["{:02.0e}".format(y) for y in ax_rp.get_yticks()])
            ax_rp.set_xticklabels(["{:01.0f}".format(y) for y in ax_rp.get_xticks()])

            # f_rp.tight_layout()
            if printing:
                f_name = "{:02.02f}_{}_{:02.02e}_radial_profiles.pdf".format(energy_ev,
                                                                             self.lam,
                                                                             self.i_w_cm_2[-1])
                f_rp.savefig(os.path.join(self.save_folder, f_name), dpi=300)
                plt.close(f_rp)
            return y1, y2

    def plot_dependence_on_energy(self, on, ints=(0, -1),
                                  include_ref=False, n=(13, 15),
                                  ev=None, uncertainty=False,
                                  printing=False):
        """
        :param on: Can be "a" (alpha) or "n" (refractive index). Both are case-insensitive
        :param ints: List of intensity to plot. Default is: (0, -1)
        :param include_ref: Only valid if "on=n". Plot also the ref values from
                    A. Lucas et al., Phys. Rev. B 28(25):2485 (1983)
        :param n: The harmonic we want to draw in the plot
        :param ev: Alternatively to "n" you can pass your own energy of interest in eV
        :param uncertainty: Energy uncertainty (in eV) ... final output will be averaged across this energy range
        :return: Matplotlib figure
        """
        if printing:
            context = plt.rc_context({"font.size": 46,
                                      "text.usetex": True,
                                      "text.latex.preamble": r"\usepackage{mathpazo}"})
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
            lbl_ = ["{}th. Harmonic ({:02.02f} eV)".format(x, y) for x, y in zip(n, energy_ev)]
        elif ev is not None:
            energy_ev = ev
            if isinstance(energy_ev, int):
                energy_ev = [energy_ev]
            lbl_ = ["{:02.02f} eV".format(y) for x, y in energy_ev]
        else:
            energy_ev = None

        with sns.axes_style("whitegrid"), context:
            if printing:
                f, ax = plt.subplots(1, 1, figsize=(32, 20))
            else:
                f, ax = plt.subplots(1, 1, figsize=(16, 10))

            for i_ in ints:
                ax.plot(self.interp_grid_ev, y[i_].real if on == "a" else (1 - y[i_].real),
                        label=r"{} | IR: {:02.02e} $W/cm^2$".format(lbl[0], self.i_w_cm_2[i_]),
                        linewidth=5, color="tab:blue", linestyle="solid" if i_ == ints[0] else "dashed")
                ax.plot(self.interp_grid_ev, y[i_].imag,
                        label=r"{} | IR: {:02.02e} $W/cm^2$".format(lbl[1], self.i_w_cm_2[i_]),
                        linewidth=5, color="tab:orange", linestyle="solid" if i_ == ints[0] else "dashed")
            if on == "n" and include_ref:
                ax.plot(self.ref_values.e_ev, self.ref_values.delta, label=r"$\delta$ (Literature)", linewidth=3,
                        linestyle="dashdot", color="black", alpha=.5)
                ax.plot(self.ref_values.e_ev, self.ref_values.beta, label=r"$\beta$ (Literature)", linewidth=3,
                        linestyle="dashdot", color="0.5", alpha=.5)
            if energy_ev is not None:
                for i, (e, l, c) in enumerate(zip(energy_ev, lbl_, ["tab:brown", "tab:red"])):
                    if uncertainty:
                        _idx = uncertainty / 2
                        ax.axvline(e - _idx, color=sns.color_palette()[i],
                                   linewidth=3, linestyle="dashdot")
                        ax.axvline(e + _idx, color=sns.color_palette()[i],
                                   linewidth=.5, linestyle="dashdot")
                    ax.axvline(e, label=l, color=c,
                               linewidth=3, linestyle="dashed")
            ax.set_xlabel("Energy [eV]")
            ax.set_xlim([self.ref_values.e_ev.min(), self.ref_values.e_ev.max()])
            ax.set_xticklabels(["{:0.01f}".format(x) for x in ax.get_xticks()])
            ax.set_yticklabels([])
            ax.set_yticklabels(["{:02.02f}".format(y) for y in ax.get_yticks()])
            # if on == "n" and include_ref:
            #     ax.set_ylim(-1, 1)
            # ax.set_title("Energy dependency of {}/{}".format(*lbl))
            # ax.legend(loc=1)
            # ax[1].legend("")
            ax.legend(bbox_to_anchor=(0.025, 1.02, 0.95, .102),
                      loc='lower left',
                      ncol=3,
                      mode="expand",
                      borderaxespad=0.)
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
                                     uncertainty=0.,
                                     printing=False):
        """
        :param on: Can be "a" (alpha), "n" (refractive index) or "int" (Brightness on detector) (Case-insensitive)
        :param arg: Can be "real", "imag", "abs"
        :param n: The harmonic with which we probe
        :param ev: Alternatively to "n" you can pass your own energy of interest in eV
        :param ints: List of intensity to plot. Default is: (0, -1) .. This is (start, end) idx
        :param uncertainty: Energy uncertainty (in eV) ... final output will be averaged across this energy range
        :return: Matplotlib figure
        """
        if printing:
            context = plt.rc_context({"font.size": 46,
                                      "text.usetex": True,
                                      "text.latex.preamble": r"\usepackage{mathpazo,nicefrac}"})
        else:
            context = nullcontext()

        on = on.lower()
        arg = arg.lower()
        assert on in ["a", "n", "int"]
        assert arg in ["real", "imag", "abs", "both"]

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
        elif arg == "both":
            def arg_fun(x):
                return np.real(x), np.imag(x)

            if on == "a":
                lbl = (r"$\alpha_1$", r"$\alpha_2$")
            else:
                lbl = (r"$\delta$", r"$\beta$")
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
                f, ax = plt.subplots(1, 1, figsize=(32, 18))
            else:
                f, ax = plt.subplots(1, 1, figsize=(16, 9))

            if arg == "both":
                ax2 = ax.twinx()
            for ii, ev_ in enumerate(energy_ev):
                # Get the idx of the desired energy
                idx = int(sum(self.interp_grid_ev < ev_))
                if uncertainty == 0:
                    if arg == "both":
                        y = np.array([x[idx] for x in data[0]])
                        yy = np.array([x[idx] for x in data[1]])
                    else:
                        y = np.array([x[idx] for x in data])
                else:
                    lo = self.interp_grid_ev.min()
                    hi = self.interp_grid_ev.max()
                    res = len(self.interp_grid_ev)
                    _idx = int(res * (uncertainty / 2) / (np.abs(hi - lo)))
                    if arg == "both":
                        y = np.array([x[idx - _idx:idx + _idx].mean() for x in data[0]])
                        yy = np.array([x[idx - _idx:idx + _idx].mean() for x in data[1]])
                    else:
                        y = np.array([x[idx - _idx:idx + _idx].mean() for x in data])

                if arg == "both":
                    lbl_1 = r"{} - Energy {:02.02f} eV".format(lbl[0], ev_)
                    lbl_2 = r"{} - Energy {:02.02f} eV".format(lbl[1], ev_)
                else:
                    lbl_ = r"{} - Energy {:02.02f} eV".format(lbl, ev_)
                if n is not None:
                    if arg == "both":
                        lbl_1 += r" - {}th. Harmonic".format(n[ii])
                        lbl_2 += r" - {}th. Harmonic".format(n[ii])
                    else:
                        lbl_ += r" - {}th. Harmonic".format(n[ii])
                if uncertainty > 0:
                    if arg == "both":
                        lbl_1 += r" - Averaged over $\pm${:01.02f} eV".format(uncertainty / 2)
                        lbl_2 += r" - Averaged over $\pm${:01.02f} eV".format(uncertainty / 2)
                    else:
                        lbl_ += r" - Averaged over $\pm${:01.02f} eV".format(uncertainty / 2)

                if on == "n" and (arg == "real" or arg == "both"):
                    y = 1 - y

                if arg == "both":
                    ax.plot(self.i_w_cm_2[ints[0]:ints[1]], y,
                            label=lbl_1, color="tab:blue",
                            linewidth=5, ls="dashed" if ii == 1 else "solid")
                    ax2.plot(self.i_w_cm_2[ints[0]:ints[1]], yy,
                             label=lbl_2, ls="dashed" if ii == 1 else "solid", color="tab:orange",
                             linewidth=5)
                    ax2.grid(False)
                    ax2.set_yticklabels(["{:02.02f}".format(y) for y in ax2.get_yticks()])
                else:
                    ax.plot(self.i_w_cm_2[ints[0]:ints[1]], y,
                            label=lbl_,
                            linewidth=3)
                ax.set_yticklabels(["{:02.02f}".format(y) for y in ax.get_yticks()])
                ax.set_xticklabels(["{:02.01f}".format(y * 1e-12) for y in ax.get_xticks()])
            ax.grid(True)
            ax.set_xlabel(r"Intensity [$\cdot 10^{12}~\nicefrac{W}{cm^{2}}$]")
            if arg == "both":
                ax.set_ylabel(r"{} [arb.]".format(lbl[0]))
                ax2.set_ylabel(r"{} [arb.]".format(lbl[1]))
            else:
                ax.set_ylabel(r"{} [arb.]".format(lbl))
            # ax.set_title("IR intensity dependency of {}".format(lbl))
            if arg == "both":
                # lines, labels = ax.get_legend_handles_labels()
                # lines2, labels2 = ax2.get_legend_handles_labels()
                # ax2.legend([lines[0], lines2[0], lines[1], lines2[1]],
                #            [labels[0], labels2[0], labels[1], labels2[1]], loc=1)
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                # ax[1].legend("")
                ax2.legend(lines + lines2, labels + labels2,
                           bbox_to_anchor=(0.025, 1.02, 0.95, .102),
                           loc='lower left',
                           ncol=2,
                           mode="expand",
                           borderaxespad=0.)
                # ax2.legend(lines + lines2, labels + labels2, loc=1)
            else:
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

    def plot_real_imag_scan(self, on, cmap="turbo", n=(13, 15), ev=None, uncertainty=False, printing=False):
        """
        Highly specific method for plotting the real/imag scans

        :param on: Can be "a" (alpha) or "n" (refractive index). Both are case-insensitive
        :param cmap: The cmap for both axes
        :param n: The harmonic we want to draw in the contour plot
        :param ev: Alternatively to "n" you can pass your own energy of interest in eV
        :param uncertainty: Energy uncertainty (in eV) ... final output will be averaged across this energy range
        :return: Matplotlib figure
        """
        if printing:
            context = plt.rc_context({"font.size": 46,
                                      "text.usetex": True,
                                      "text.latex.preamble": r"\usepackage{mathpazo, nicefrac}"})
        else:
            context = nullcontext()

        if n is not None or ev is not None:  # only on arg can be not None
            assert (n is None and ev is not None) or (n is not None and ev is None)

        if n is not None:
            energy_ev = self.calc_harmonic(n)
            if isinstance(n, int):
                n = [n]
                energy_ev = [energy_ev]
            lbl = ["{}th. Harmonic ({:02.02f} eV)".format(x, y) for x, y in zip(n, energy_ev)]
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
                f, ax = plt.subplots(1, 2, figsize=(36, 18))
            else:
                f, ax = plt.subplots(1, 2, figsize=(18, 8))

            real_data = y.real if on == "a" else (1 - y.real)
            real_h = ax[0].imshow(real_data, cmap=plt.cm.get_cmap(cmap))
            imag_h = ax[1].imshow(y.imag, cmap=plt.cm.get_cmap(cmap))

            x_ticks = np.arange(int(len(self.interp_grid_ev)))[::int(len(self.interp_grid_ev) / 5)]
            x_tick_labels = self.interp_grid_ev[::int(len(self.interp_grid_ev) / 5)]

            y_ticks = np.arange(int(len(self.i_w_cm_2)))[::int(len(self.i_w_cm_2) / 5)]
            y_tick_labels = self.i_w_cm_2[::int(len(self.i_w_cm_2) / 5)] * 1e-12
            for ax_ in ax:
                ax_.set_aspect(y.shape[1] / y.shape[0])
                ax_.grid(False)
                ax_.set_xlabel("Energy [eV]")
                ax_.set_ylabel(r"Intensity [$\cdot 10^{12}~\nicefrac{W}{cm^{2}}$]")
                ax_.set_xticks(x_ticks)
                ax_.set_xticklabels(["{:02.0f}".format(x) for x in x_tick_labels])
                ax_.set_yticks(y_ticks)
                ax_.set_yticklabels(["{:02.01f}".format(y) for y in y_tick_labels])
                if energy_ev is not None:
                    ee = [int(sum(self.interp_grid_ev < x)) for x in energy_ev]
                    for i, (e, l, c) in enumerate(zip(ee, lbl, ["tab:brown", "tab:red"])):
                        if uncertainty:
                            lo = self.interp_grid_ev.min()
                            hi = self.interp_grid_ev.max()
                            res = len(self.interp_grid_ev)
                            _idx = int(res * (uncertainty / 2) / (np.abs(hi - lo)))
                            ax_.axvline(e - _idx, color=sns.color_palette("bright", 8)[2 + i],
                                        linewidth=1, linestyle="dashed")
                            ax_.axvline(e + _idx, color=sns.color_palette("bright", 8)[2 + i],
                                        linewidth=1, linestyle="dashed")
                        ax_.axvline(e, label=l, color=c,
                                    linewidth=5, linestyle="dashed")
                    ax_.legend()
            ax[0].set_title(ttl_[0])
            ax[1].set_title(ttl_[1])
            # f.suptitle(ttl)
            cbar = f.colorbar(real_h, ax=ax[0], fraction=.05, pad=.005, shrink=.85)
            # cbar = f.colorbar(real_h, ax=ax[0], shrink=.85)

            cbar_ = f.colorbar(imag_h, ax=ax[1], fraction=.05, pad=.005, shrink=.85)
            # cbar_ = f.colorbar(imag_h, ax=ax[1], shrink=.85)
            if on == "a":
                cbar_ticks = ["{:1.0f}".format(x) for x in cbar.get_ticks()]
                cbar_2_ticks = ["{:1.0f}".format(x) for x in cbar_.get_ticks()]
            else:
                cbar_ticks = ["{:1.01f}".format(x) for x in cbar.get_ticks()]
                cbar_2_ticks = ["{:1.01f}".format(x) for x in cbar_.get_ticks()]

            cbar.ax.set_yticklabels(cbar_ticks)
            cbar_.ax.set_yticklabels(cbar_2_ticks)
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
                         cmap_norm=None,
                         use_log=True,
                         max_angle=None,
                         r=None,
                         detector_hole=True,
                         printing=False,
                         return_img=False,
                         just_img=False):
        """

        :param rads: The Radial profiles
        :param n: (Optional) The harmonic with which we probe
        :param ev: (Optional) Alternatively to "n" you can pass your own energy of interest in eV
        :param cmap_norm: (Optional) Normalization constant for color map
        :param use_log: (Optional) (Bool) Plot in log color scale
        :param max_angle: (Optional) For the title ... Max scattering angle
        :param r: (Optional) For the title ... Radius of the droplet
        :param detector_hole: (Bool) Simulate a detector hole
        :param return_img: (Optional) Return the numpy array
        :return: 1
        """
        if printing:
            context = plt.rc_context({"font.size": 72,
                                      "text.usetex": True,
                                      "text.latex.preamble": r"\usepackage{mathpazo}"})
        else:
            context = nullcontext()
        if n is not None:
            energy_ev = self.calc_harmonic(n)
        else:
            energy_ev = ev

        if ev is None and n is None:
            assert energy_ev is None

        assert len(rads) == 2  # You have to pass exactly two profiles
        if detector_hole:
            rads = np.array(rads)
            rads[:, :50] = 0  # This corresponds to ~ 1.5°

        img0, ma, mi = self.get_diffraction(rads[0], use_log)
        img1, ma_2, mi_2 = self.get_diffraction(rads[1], use_log)

        if just_img:
            return img0, img1

        ma = np.max([ma, ma_2])
        mi = np.min([mi, mi_2])
        # print(ma, mi)
        plt.close("all")
        with sns.axes_style("white"), context:
            if printing:
                f, ax = plt.subplots(1, 2, figsize=(32, 16))
            else:
                f, ax = plt.subplots(1, 2, figsize=(16, 8.75))

            for a, img_, ra, s in zip(ax, (img0, img1), rads, ("a)", "b)")):
                if cmap_norm is not None:
                    mi = 0
                    ma = cmap_norm
                a.imshow(img_, vmin=mi, vmax=ma)
                a.set_xticks([])
                a.set_yticks([])
                a.text(0.025, .925, s, transform=a.transAxes, color="w")
                if use_log:
                    int_ = np.exp(img_).sum() / np.exp(img0).sum()
                else:
                    int_ = img_.sum() / img0.sum()
                ttl = "Brightness: {:.02%}%".format(int_)
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

                # a.set_title(ttl)
            ttl = "Diffraction pattern ({} cmap)".format("log." if use_log else "linear")
            if max_angle is not None or r is not None:
                ttl += " | "
            if max_angle is not None:
                ttl += r"Max scat. angle: {}°".format(max_angle)
            if r is not None:
                if max_angle is not None:
                    ttl += " | "
                ttl += "Droplet radius: {} nm".format(r / 2)
            # f.suptitle(ttl)
            f.tight_layout()
            f.subplots_adjust(top=0.96,
                              bottom=0.0,
                              left=0.012,
                              right=0.988,
                              hspace=0.2,
                              wspace=0.029)
            if printing:
                if isinstance(n, int):
                    n = [n]
                nn = ["{:02.02f}".format(x) for x in n]
                f.savefig(os.path.join(self.save_folder,
                                       "{}_{}_{:02.02e}_scattering.pdf".format("_".join(nn),
                                                                               self.lam,
                                                                               self.i_w_cm_2[-1])),
                          dpi=300)
                plt.close(f)
            if return_img:
                return img0, img1
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
            context = plt.rc_context({"font.size": 36,
                                      "text.usetex": True,
                                      "text.latex.preamble": r"\usepackage{mathpazo, nicefrac}"})
        else:
            context = nullcontext()

        if cmap.lower() == "turbo":
            plt.register_cmap(name="turbo", data=get_turbo(), lut=256)

        if on == "a":
            y = self.a_interp
        else:
            y = self.n_interp

        cmap = plt.cm.get_cmap(cmap)
        color_picker = np.linspace(0, 255, max_int).astype(int)
        colors = [cmap(i) for i in color_picker]

        with context:
            if printing:
                fig = plt.figure(figsize=(40, 16))
            else:
                fig = plt.figure(figsize=(12, 8))
            ax = fig.gca(projection="3d")
            if on == "a":
                # See equation (22) in:
                # Gaarde, M. B., Buth, C., Tate, J. L., & Schafer, K. J. (2011).
                # Transient absorption and reshaping of ultrafast XUV light by laser-dressed helium.
                # Physical Review A - Atomic, Molecular, and Optical Physics, 83(1).
                ww = 2 * np.pi / self.period * np.arange(y.shape[-1]) / self.au_time
                const = 4 * np.pi * ww / (cs.c * cs.epsilon_0) / self.fine_struct * 1e-2

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
            ax.add_collection3d(poly, zs=np.linspace(-1, max_int, max_int) - .5, zdir="y")

            ax.set_xlabel("Energy [eV]", labelpad=35)
            ax.set_xlim3d(xs.min(), xs.max())

            if on == "a":
                ax.set_zlabel("Cross section [100 Mbarn]", labelpad=75)
            else:
                ax.set_zlabel("Absorption", labelpad=75)
            ax.set_zlim3d(0, y_max)

            ax.set_ylabel(r"Intensity [$\cdot 10^{12}~\nicefrac{W}{cm^{2}}$]", labelpad=90)
            # ax.set_ylabel(r"IR Intensity $W/cm^2$", labelpad=90)
            ax.set_ylim3d(0, max_int)
            yticks = np.linspace(0, max_int, 5).astype(int)
            yticklabels = self.i_w_cm_2[yticks] * 1e-12
            ax.set_yticks(yticks)
            ax.set_yticklabels(["{:02.01f}".format(x) for x in yticklabels])

            ax.view_init(elev=47, azim=-90.1)

            if plot_energy_levels:
                harm_13 = self.calc_harmonic(13)
                harm_15 = self.calc_harmonic(15)
                ax.plot([harm_13] * 2, [-.5, 28.5], [0] * 2, color="tab:blue", lw=3, linestyle="-",
                        label="Position of 13th. Harmonic ({:02.02f} eV)".format(harm_13))
                ax.plot([harm_15] * 2, [-.5, 28.5], [0] * 2, color="tab:red", lw=3, linestyle="-",
                        label="Position of 15th. Harmonic ({:02.02f} eV)".format(harm_15))
                ax.plot([20.61577498] * 2, [-.5, 28.5], [0] * 2, color="black", lw=3, linestyle="--",
                        label="1s2s (20.62 eV)")
                ax.plot([21.218022851325713] * 2, [-.5, 28.5], [0] * 2, color="tab:green", lw=3, linestyle="--",
                        label="1s2p (21.22 eV)")
                ax.plot([22.9203175] * 2, [-.5, 28.5], [0] * 2, color="black", lw=3, linestyle="--",
                        label="1s3s (22.92 eV)")
                ax.plot([23.07407494] * 2, [-.5, 28.5], [0] * 2, color="black", lw=3, linestyle="--",
                        label="1s3d (23.07 eV)")
                ax.plot([23.087018663345155] * 2, [-.5, 28.5], [0] * 2, color="tab:green", lw=3, linestyle="--",
                        label="1s3p (23.09 eV)")
                ax.plot([23.742070185580754] * 2, [-.5, 28.5], [0] * 2, color="tab:green", lw=3, linestyle="--",
                        label="1s4p (23.74 eV)")

            ax.tick_params(axis="y", which="major", pad=20)
            ax.tick_params(axis="z", which="major", pad=30)
            ax.tick_params(axis="x", which="major", pad=10)
            # ax.legend()
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
        # return mp.i_par(m_, size_param, mu) * scaling_factor
        return mp.i_per(m_, size_param, mu) * scaling_factor
