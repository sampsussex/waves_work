import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt

h = 0.6751

sharks_path = '/Users/sp624AA/Downloads/group_finding_mocks/galaxies_shark.parquet'

gama_like_out_path = '/Users/sp624AA/Downloads/mocks/gama_like_from_groupfinding_cat.parquet'


peturbed = False#True
# columns to read
cols = ['ra', 'dec', 'id_galaxy_sky', 'redshift_observed', 
        'redshift_cosmological', 'mass_stellar_total', 'mass_stellar_disk', 
        'mass_stellar_bulge', 'mass_virial_hosthalo', 
        'mass_virial_subhalo', 'sfr_total', 'id_group_sky', 'mag_r_SDSS',
        'id_fof', 'masked', 'mag_r_VST', 'mag_Z_VISTA', 'mag_abs_Z_VISTA', 
        'mag_abs_r_VST','mag_abs_r_SDSS']

#sharks_wide = pd.read_parquet(sharks_wide_path, columns=cols)

gama_like = pd.read_parquet(sharks_path, columns=cols)

# For waves wide i need to cut at zobs = 0.2 and for waves deep i need to cut at zobs = 0.8 and apparent mag < 21.25 in Z band.
#gama_like['stellar_mass'] = (gama_like['mstars_disk'] + gama_like['mstars_bulge'])/0.67
#gama_like['log_stellar_mass'] = np.log10((gama_like['mstars_disk'] + gama_like['mstars_bulge'])/0.67)
#gama_like['log_sfr_total'] = np.log10((gama_like['sfr_disk'] + gama_like['sfr_burst']) * 1e-9/0.67 + 1e-20)
gama_like['log_sSFR'] = gama_like['sfr_total'] - gama_like['mass_stellar_total']


print('checking hs in mass stellar total')
# I know that mass_stellar_disk and mass_stellar_bulge is in Msun/h
# So, for the first 5 rows, I will check that mass_stellar_total is equal to log10 mass_stellar_disk + mass_stellar_bulge, and that they are all in Msun/h
#for i in range(5):
#    total = gama_like['mass_stellar_total'].iloc[i]
#    disk = gama_like['mass_stellar_disk'].iloc[i]
#    bulge = gama_like['mass_stellar_bulge'].iloc[i]
#    print(f"Row {i}: total={total}, NO DIV hlog10(sum)={np.log10(disk+bulge)}, DIV h log10(sum)={np.log10((disk+bulge)/0.6751)}")
# mass_stellar_total matches the div by h version.
# as mass_stellar_disk is in Msun/h and mass_stellar_bulge is in Msun/h, when these
# are divided by h, they are in Msun. So, to convert mass_stellar_total to Msun/h, I need to times by h.
# 

plt.hist(gama_like['mass_stellar_total'])
plt.show()
plt.hist(np.log10(gama_like['mass_stellar_total']))
plt.show()

mask = (gama_like['mass_stellar_total'] > 8) & (gama_like['mag_r_VST'] > -99) & (gama_like['redshift_observed'] < 0.8) & (gama_like['mag_r_VST'] < 19.65) & (gama_like['dec'] > -3.95)
gama_like = gama_like[mask].reset_index(drop=True)

#plt.hist(gama_like['log_sSFR'], bins=50)
#plt.xlabel('log sSFR')
#plt.ylabel('Number of galaxies')
#plt.show()

# I need to caclulate the sfr and save the is_red flag
gama_like['is_red'] = gama_like['log_sSFR'] < -11


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
ra_min, ra_max, dec_min, dec_max = gama_like['ra'].min(), gama_like['ra'].max(), gama_like['dec'].min(), gama_like['dec'].max()
area = rectangular_sky_area_deg2(ra_min, ra_max, dec_min, dec_max)
frac = rectangular_fraction_of_sky(ra_min, ra_max, dec_min, dec_max)
# print input ra dec min and max
print(f"RA: [{ra_min}, {ra_max}], Dec: [{dec_min}, {dec_max}]")

print(f"Area: {area:.2f} deg^2, Fraction of sky: {frac:.6f}")



# Check sanity by printing ra and dec min and max
print(f"WavesDeep RA min: {gama_like['ra'].min()}, RA max: {gama_like['ra'].max()}")
print(f"WavesDeep Dec min: {gama_like['dec'].min()}, Dec max: {gama_like['dec'].max()}")



# I need to caculate the k corrections back from the abs and app mags.
cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3)
gama_like['DM'] = cosmo.distmod(gama_like['redshift_cosmological']).value

#data['Rpetro_abs'] = data['Rpetro'] - data['DM'] - data['k-e corr']
gama_like['k-e corr'] = -gama_like['mag_abs_r_VST'] + gama_like['mag_r_VST'] - gama_like['DM']

# Plot histogram of k-e corr

plt.hist(gama_like['k-e corr'], bins=50)
plt.xlabel('k-e corr (Z band)')
plt.ylabel('Number of galaxies')
plt.show()

plt.hist(gama_like['redshift_observed'], bins=50, log = True)
plt.xlabel('Observed Redshift')
plt.ylabel('Number of galaxies')
#plt.xscale('symlog')
plt.show()
gama_like['mass_stellar_total'] = gama_like['mass_stellar_total'] + np.log10(h)
gama_like['stellar_mass'] = 10**(gama_like['mass_stellar_total'])# * h
#gama_like['mass_stellar_total'] = gama_like['mass_stellar_total'] + np.log10(h)

plt.hist(gama_like['stellar_mass'], bins=50, log=True)
plt.xlabel('Stellar Mass (Msun)')
plt.ylabel('Number of galaxies')
plt.show()
#if peturbed:
    #print("Peturbing Stellar Masses with a Gaussian noise of 0.2 dex")
    #gama_like['log_stellar_mass'] = gama_like['log_stellar_mass'] + np.random.normal(0, 0.2, size=len(gama_like))
    #gama_like['stellar_mass'] = 10**gama_like['log_stellar_mass']
# I need to save as a fits file.  
gama_like.to_parquet(gama_like_out_path, index=False)

