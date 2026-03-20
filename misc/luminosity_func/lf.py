#!/usr/bin/env python3
"""
Simple luminosity function + Schechter fit using a Parquet catalogue.

- Loads data with pandas.read_parquet()
- No config reader
- No CLI arguments
- Edit parameters in the USER SETTINGS section
"""

import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from astropy.cosmology import FlatLambdaCDM

warnings.filterwarnings("ignore")


# ============================================================
# ===================== USER SETTINGS =========================
# ============================================================

# --- Input / output ---
PARQUET_PATH = "/Users/sp624AA/Downloads/mocks/v0.4.0/cut_waves_deep_gals_peturbed_sm.parquet"
SAVE_PATH = "lf_schechter.dat"

# --- Column names in parquet ---
COL_Z = "zcos"
COL_ABS_MAG = "total_ab_dust_Z_VISTA"
COL_APP_MAG = "total_ap_dust_Z_VISTA"

# --- Survey properties ---
SURVEY_AREA_SQDEG = 50.59
Z_MIN = 0.00
Z_MAX = 0.8
MAG_LIM = 21.25

# --- LF setup ---
NBINS = 50

# --- Cosmology ---
H0 = 67.51
OM0 = 0.3
h = H0 / 100.0

# ============================================================


def schecmag(M, alpha, Mstar, lgps):
    """Schechter function in magnitude space."""
    ln10 = np.log(10.0)
    L = 10.0 ** (0.4 * (Mstar - M))
    return 0.4 * ln10 * 10.0 ** lgps * L ** (1.0 + alpha) * np.exp(-L)

def log10_schecmag(M, alpha, Mstar, lgps):
    """log10 of Schechter phi(M)."""
    return np.log10(schecmag(M, alpha, Mstar, lgps))



class LuminosityFunction:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        self.parquet_path = PARQUET_PATH
        self.savepath = SAVE_PATH

        self.col_z = COL_Z
        self.col_abs_mag = COL_ABS_MAG
        self.col_app_mag = COL_APP_MAG

        self.sq_deg_area = SURVEY_AREA_SQDEG
        self.z_min = Z_MIN
        self.z_max = Z_MAX
        self.mag_lim = MAG_LIM
        self.nbins = NBINS

        self.cosmology = FlatLambdaCDM(H0=H0, Om0=OM0)

        logging.info("LuminosityFunction initialized")
        logging.info(f"Parquet: {self.parquet_path}")
        logging.info(f"Survey area: {self.sq_deg_area} sq.deg")
        logging.info(f"Redshift range: [{self.z_min}, {self.z_max}]")
        logging.info(f"Magnitude limit: {self.mag_lim}")

    def load_data(self):
        logging.info("Loading parquet file")
        df = pd.read_parquet(
            self.parquet_path,
            columns=[self.col_z, self.col_abs_mag, self.col_app_mag],
        )

        self.zs = df[self.col_z].to_numpy(dtype="float64")
        self.abs_mag = df[self.col_abs_mag].to_numpy(dtype="float64")
        self.app_mag = df[self.col_app_mag].to_numpy(dtype="float64")

        self.abs_mag = self.abs_mag - 5.0 * np.log10(h)

        good = np.isfinite(self.zs) & np.isfinite(self.abs_mag) & np.isfinite(self.app_mag)
        self.zs = self.zs[good]
        self.abs_mag = self.abs_mag[good]
        self.app_mag = self.app_mag[good]

        self.k_corr = (
            self.app_mag
            - self.abs_mag
            - self.cosmology.distmod(self.zs).value
        )

        self.bin_min = np.min(self.abs_mag)
        self.bin_max = np.max(self.abs_mag)

        logging.info(f"Loaded {len(self.zs)} galaxies")

    def get_zlims(self):
        zlim = np.zeros(len(self.zs))

        def ddm(z, i):
            return (
                self.mag_lim
                - self.abs_mag[i]
                - self.cosmology.distmod(z).value
                - self.k_corr[i]
            )

        for i in range(len(self.zs)):
            if ddm(self.z_max, i) > 0:
                zlim[i] = self.z_max
            else:
                zlim[i] = scipy.optimize.brentq(
                    ddm, max(self.zs[i], self.z_min), 2.0, args=i
                )

        self.zlims = zlim
        logging.info("Computed z-limits")

    def get_v_vmax(self):
        area_frac = self.sq_deg_area * (np.pi / 180.0) ** 2 / (4.0 * np.pi)

        V = area_frac * self.cosmology.comoving_volume(self.zs).value
        Vmax = area_frac * self.cosmology.comoving_volume(self.zlims).value

        good = (Vmax > 0) & np.isfinite(Vmax)
        self.V = V[good]
        self.Vmax = Vmax[good]
        self.abs_mag_v = self.abs_mag[good]
        self.VVm = self.V / self.Vmax

    def compute_luminosity_function(self):
        bins = np.linspace(self.bin_min, self.bin_max, self.nbins)

        # counts and weighted LF
        Nbin, edges = np.histogram(self.abs_mag_v, bins=bins)
        phi_lin, _ = np.histogram(self.abs_mag_v, bins=bins, weights=1.0 / self.Vmax)

        Mbin = edges[:-1] + 0.5 * np.diff(edges)
        dM = edges[1] - edges[0]

        # Better error for weighted counts: sqrt(sum w^2)
        w = 1.0 / self.Vmax
        sumw2, _ = np.histogram(self.abs_mag_v, bins=bins, weights=w * w)
        phi_err_lin = np.sqrt(sumw2)

        # per-mag
        self.Mbin = Mbin
        self.phi = phi_lin / dM
        self.phi_err = phi_err_lin / dM

        # bins usable for log-fit
        ok = (
            (Nbin > 0)
            & np.isfinite(self.phi)
            & np.isfinite(self.phi_err)
            & (self.phi > 0)
            & (self.phi_err > 0)
        )

        # log10(phi) and its uncertainty
        y = np.log10(self.phi[ok])
        yerr = self.phi_err[ok] / (self.phi[ok] * np.log(10.0))

        # initial guess (alpha, M*, log10(phi*))
        p0 = (-1.0, -22.0, -2.5)

        self.popt, self.pcov = scipy.optimize.curve_fit(
            log10_schecmag,
            self.Mbin[ok],
            y,
            p0=p0,
            sigma=yerr,
            absolute_sigma=False,
            maxfev=20000,
        )

        # evaluate best-fit in linear space for plotting
        self.phi_schec = schecmag(self.Mbin, *self.popt)

        logging.info(
            f"Schechter (log-fit): alpha={self.popt[0]:.3f}, "
            f"M*={self.popt[1]:.3f}, log10(phi*)={self.popt[2]:.3f}"
        )

    def save_results(self):
        errors = np.sqrt(np.diag(self.pcov))
        values = np.array([
            10 ** self.popt[2], errors[2],
            self.popt[1], errors[1],
            self.popt[0], errors[0],
        ])

        header = "phi_star phi_star_err M_star M_star_err alpha alpha_err"
        np.savetxt(self.savepath, [values], header=header)
        logging.info(f"Saved results to {self.savepath}")

    def plot(self):
        import matplotlib.gridspec as gridspec

        # Recompute Nbin to ensure consistency with plotted bins
        bins = np.linspace(self.bin_min, self.bin_max, self.nbins)
        Nbin, edges = np.histogram(self.abs_mag_v, bins=bins)
        bin_centres = edges[:-1] + 0.5 * np.diff(edges)
        bin_width = edges[1] - edges[0]

        fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.0)

        # --- Top panel: LF + Schechter ---
        ax1 = fig.add_subplot(gs[0])
        ax1.errorbar(
            self.Mbin,
            self.phi,
            self.phi_err,
            fmt="o",
            label="Data",
        )
        ax1.plot(self.Mbin, self.phi_schec, label="Schechter fit")
        ax1.set_yscale("log")
        ax1.set_ylabel(r"$\phi(M)$ [Mpc$^{-3}$ mag$^{-1}$]")
        ax1.legend()
        ax1.set_ylim(1e-7, 1e0)
        ax1.tick_params(labelbottom=False)

        # --- Bottom panel: N per bin ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(
            bin_centres,
            Nbin,
            width=bin_width,
            align="center",
            edgecolor="black",
        )
        ax2.set_ylabel("N")
        ax2.set_xlabel(r"$M - 5\log_{10}h$")
        #ax2.set_xlabel("Absolute magnitude")
        
        # log scale on y-axis for ax2
        ax2.set_yscale("log")

        plt.tight_layout()
        plt.savefig("luminosity_function.png", dpi=150)
        plt.show()

    def run(self):
        self.load_data()
        self.get_zlims()
        self.get_v_vmax()
        self.compute_luminosity_function()
        self.save_results()
        self.plot()

        print("abs_mag percentiles:", np.percentile(self.abs_mag, [0.1, 1, 50, 99, 99.9]))
        print("k_corr percentiles:", np.percentile(self.k_corr, [0.1, 1, 50, 99, 99.9]))
        print("zlims min/med/max:", np.min(self.zlims), np.median(self.zlims), np.max(self.zlims))
        print("Vmax percentiles:", np.percentile(self.Vmax, [0.1, 1, 50, 99, 99.9]))


if __name__ == "__main__":
    lf = LuminosityFunction()
    lf.run()
