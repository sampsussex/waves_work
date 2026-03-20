import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt



sharks_wide_path = '/Users/sp624AA/Downloads/mocks/v0.4.0/waves_wide_gals.parquet'
sharks_deep_path = '/Users/sp624AA/Downloads/mocks/v0.4.0/waves_deep_gals.parquet'

sharks_deep_out_path = '/Users/sp624AA/Downloads/mocks/v0.4.0/cut_waves_deep_gals_peturbed_sm.parquet'
sharks_wide_out_path = '/Users/sp624AA/Downloads/mocks/v0.4.0/cut_waves_wide_gals_peturbed_sm.parquet'


peturbed = True
# columns to read
cols = ['ra', 'dec', 'id_galaxy_sky', 'id_group_sky', 'zcos', 'zobs', 'mstars_bulge', 'mstars_disk', 'mgas_disk', 'mgas_bulge', 'mvir_hosthalo', 'mvir_subhalo', 'id_fof', 'sfr_disk', 'sfr_burst', 'total_ab_dust_u_VST', 'total_ab_dust_g_VST', 'total_ab_dust_r_VST', 'total_ab_dust_i_VST', 'total_ab_dust_Z_VISTA','total_ap_dust_Z_VISTA', 'total_ab_dust_Y_VISTA', 'total_ab_dust_J_VISTA', 'total_ab_dust_H_VISTA', 'total_ab_dust_K_VISTA']

#sharks_wide = pd.read_parquet(sharks_wide_path, columns=cols)

sharks_deep = pd.read_parquet(sharks_deep_path, columns=cols)

# For waves wide i need to cut at zobs = 0.2 and for waves deep i need to cut at zobs = 0.8 and apparent mag < 21.25 in Z band.
sharks_deep['stellar_mass'] = (sharks_deep['mstars_disk'] + sharks_deep['mstars_bulge'])/0.67
sharks_deep['log_stellar_mass'] = np.log10((sharks_deep['mstars_disk'] + sharks_deep['mstars_bulge'])/0.67)
sharks_deep['log_sfr_total'] = np.log10((sharks_deep['sfr_disk'] + sharks_deep['sfr_burst']) * 1e-9/0.67 + 1e-20)
sharks_deep['log_sSFR'] = np.log10((sharks_deep['sfr_disk'] * 1e-9 + sharks_deep['sfr_burst'] * 1e-9) / (sharks_deep['mstars_disk'] + sharks_deep['mstars_bulge']) / 0.67 + 1e-20)
mask = (sharks_deep['log_stellar_mass'] > 8) & (sharks_deep['total_ab_dust_Z_VISTA'] > -99) & (sharks_deep['zobs'] < 0.8) & (sharks_deep['total_ap_dust_Z_VISTA'] < 21.25)
sharks_deep = sharks_deep[mask].reset_index(drop=True)


# I need to caclulate the sfr and save the is_red flag
sharks_deep['is_red'] = sharks_deep['log_sSFR'] < -11


# I need to find and calculate the survey fractional sky area. 

def rectangular_sky_area_deg2(ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg):
    """
    Area on the celestial sphere for an RA/Dec-aligned rectangle.

    Handles RA wrap-around (e.g. ra_min=350, ra_max=20).
    Inputs/outputs in degrees / deg^2.
    """
    ra_min = ra_min_deg % 360.0
    ra_max = ra_max_deg % 360.0

    # RA span in degrees, wrap-safe
    d_ra = (ra_max - ra_min) % 360.0  # in [0, 360)
    if np.isclose(d_ra, 0.0) and not np.isclose(ra_max_deg, ra_min_deg):
        d_ra = 360.0  # full wrap (rare edge case)

    # spherical rectangle area: Δλ * (sin δ2 - sin δ1)
    d_lambda = np.deg2rad(d_ra)
    sin_term = np.sin(np.deg2rad(dec_max_deg)) - np.sin(np.deg2rad(dec_min_deg))
    area_sr = d_lambda * sin_term

    # convert sr -> deg^2
    area_deg2 = area_sr * (180.0 / np.pi) ** 2
    return area_deg2

def rectangular_fraction_of_sky(ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg):
    area_deg2 = rectangular_sky_area_deg2(ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg)
    full_sky_deg2 = 4.0 * np.pi * (180.0 / np.pi) ** 2  # ~41252.96
    return area_deg2 / full_sky_deg2

# ---- example ----
ra_min, ra_max, dec_min, dec_max = 339, 351, -35, -30
area = rectangular_sky_area_deg2(ra_min, ra_max, dec_min, dec_max)
frac = rectangular_fraction_of_sky(ra_min, ra_max, dec_min, dec_max)
# print input ra dec min and max
print(f"RA: [{ra_min}, {ra_max}], Dec: [{dec_min}, {dec_max}]")

print(f"Area: {area:.2f} deg^2, Fraction of sky: {frac:.6f}")



# Check sanity by printing ra and dec min and max
print(f"WavesDeep RA min: {sharks_deep['ra'].min()}, RA max: {sharks_deep['ra'].max()}")
print(f"WavesDeep Dec min: {sharks_deep['dec'].min()}, Dec max: {sharks_deep['dec'].max()}")



# I need to caculate the k corrections back from the abs and app mags.
cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3)
sharks_deep['DM'] = cosmo.distmod(sharks_deep['zcos']).value

#data['Rpetro_abs'] = data['Rpetro'] - data['DM'] - data['k-e corr']
sharks_deep['k-e corr'] = -sharks_deep['total_ab_dust_Z_VISTA'] + sharks_deep['total_ap_dust_Z_VISTA'] - sharks_deep['DM']

# Plot histogram of k-e corr

plt.hist(sharks_deep['k-e corr'], bins=50)
plt.xlabel('k-e corr (Z band)')
plt.ylabel('Number of galaxies')
plt.show()

if peturbed:
    print("Peturbing Stellar Masses with a Gaussian noise of 0.2 dex")
    sharks_deep['log_stellar_mass'] = sharks_deep['log_stellar_mass'] + np.random.normal(0, 0.2, size=len(sharks_deep))
    sharks_deep['stellar_mass'] = 10**sharks_deep['log_stellar_mass']
# I need to save as a fits file.  
sharks_deep.to_parquet(sharks_deep_out_path, index=False)

