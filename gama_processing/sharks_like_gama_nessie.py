import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.table import join
# Importing modules
from astropy.cosmology import FlatLambdaCDM
from nessie import FlatCosmology
from nessie.helper_funcs import create_density_function
from nessie import RedshiftCatalog
from scipy.interpolate import interp1d
from scipy.integrate import quad

cosmo = FlatCosmology(h = 0.7, omega_matter = 0.3)
astropy_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
h = 0.7
# This script processes GAMA, giving me k-corrections, and stellar masses, as well as group information. 
sharks_path = "/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet"

# Read in sharks using astropy tables
print('Reading in sharks data...')
sharks = Table.read(sharks_path)
print('sharks data read in successfully.')

#        "zobs": "redshift",
#        "total_ab_dust_r_VST": "absolute_magnitude",
#        "total_ap_dust_r_VST": "apparent_magnitude", 

print('Calculating apparent magnitudes...')
sharks['app_mag_rt'] = sharks['total_ap_dust_r_VST']
print('Apparent magnitudes calculated successfully.')
sharks['Z'] = sharks['zobs'] # Rename redshift column to Z

# make sharks base cuts
sharks['log_stellar_mass'] = np.log10((sharks['mstars_disk'] + sharks['mstars_bulge'])/0.67)
mask = (sharks['log_stellar_mass'] > 8) & (sharks["total_ab_dust_r_VST"] > -200) & (sharks["total_ap_dust_r_VST"] < 19.65) #& (df['ra'] < 20)
sharks = sharks[mask]

ra_min, ra_max = 157.00, 189.96
dec_min, dec_max = -3.00, 4.

sharks = sharks[
    (sharks['ra'] >= ra_min) & (sharks['ra'] <= ra_max) &
    (sharks['dec'] >= dec_min) & (sharks['dec'] <= dec_max)
]


sharks['RAcen'] = sharks['ra']
sharks['Deccen'] = sharks['dec']
sharks['logmstar'] = sharks['log_stellar_mass']
sharks['uberID'] = sharks['id_galaxy_sky']

# K corrections
a = [0.2085, 1.0226, 0.5237, 3.5902, 2.3843]
zref = 0
Q0 = 1.75
zp = 0.2

def kcorr(z):
    k=0
    for i in range(len(a)):
        k+= (a[i] * (z - zp)**i)

    return k - (Q0 * (z-zref))

# Calculate K corrections
print('Calculating K corrections...')
sharks['Kcorr'] = kcorr(sharks['Z'])
print('K corrections calculated successfully.')

print('Calculating absolute magnitudes...')
# Luminsoity distance
sharks['dist_mod'] = astropy_cosmo.distmod(np.array(sharks['Z'])).value
# Calculate absolute magnitude

sharks['abs_mag_rt'] = sharks['app_mag_rt']-sharks['dist_mod'] - sharks['Kcorr']
print('Absolute magnitudes calculated successfully.')

# Filter out galaxies with absolute magnitude less than -100
print('Number of galaxies before absolute magnitude cut > -100:', len(sharks))
sharks = sharks[sharks['abs_mag_rt'] > -100] # Absolute magnitude cut
print('Number of galaxies after absolute magnitude cut:', len(sharks))

# Running nessie on sharks.
print('Setting up nessie run on sharks...')
print('Creating running density function...')
sharks_area_sq_deg = 230.6 # GAMA 4 area in square degrees
sharks_frac_area = sharks_area_sq_deg / (360**2 / np.pi) 

running_density = create_density_function(sharks['Z'], total_counts = len(sharks['Z']), survey_fractional_area = sharks_frac_area, cosmology = cosmo)
print('Running density function created successfully.')

print('Creating RedshiftCatalog...')
red_cat = RedshiftCatalog(sharks['RAcen'], sharks['Deccen'], sharks['Z'], running_density, cosmo)
print('RedshiftCatalog created successfully.')

print('Calculating completeness...')
completeness = np.repeat(1, len(sharks['RAcen'])) # Assuming completeness is 1 for all galaxies

red_cat.set_completeness(completeness)

print('Completeness calculated successfully.')


print('Running Fof finder...')
red_cat.run_fof(b0 = 0.04, r0 = 36)
print('Fof finder run successfully.')

group_ids = red_cat.group_ids

vel_errors = np.repeat(50, len(sharks['RAcen'])) # in km/s


# First set up lf reading
def build_integrated_lf(path: str = "gama_lf/lf.dat", cut: float = -14) -> interp1d:
    """
    Reads the GAMA luminosity function file, filters by magnitude, builds the luminosity-weighted
    luminosity function, integrates it, and returns an interpolation function over that integral.

    Args:
        path: Path to the LF file.
        cut: Magnitude cut (default = -14)

    Returns:
        interp1d: Interpolation function over the integrated phi(L) from -30 to M.
    """
    df = pd.read_csv(path, delim_whitespace=True)
    df = df[df["mag_bin_centre"] < cut].copy()

    mags = df["mag_bin_centre"].values
    phi = df["luminosity_function"].values

    # Compute luminosity-weighted φ: phi * 10^(-0.4 * M)
    phi_lum = phi * 10 ** (-0.4 * mags)

    # Interpolation of φ(L)
    func_lum = interp1d(mags, phi_lum,bounds_error=False,fill_value=(phi_lum[0], phi_lum[-1]))

    # Integrate from -30 to each M
    integrals = np.array([
        quad(func_lum, -30, M, limit=1000)[0] if np.isfinite(M) else 0.0
        for M in mags
    ])

    # Replace zeros with min positive value
    positive = integrals[integrals > 0]
    if len(positive) == 0:
        raise ValueError("All integrated values are zero; check your LF input.")
    min_val = positive.min()
    integrals[integrals == 0] = min_val

    return interp1d(mags, integrals, bounds_error=False, fill_value=(integrals[0], integrals[-1]))



print('Calculating group properties...')
properties = Table(red_cat.calculate_group_table(sharks['abs_mag_rt'], vel_errors))
print('Group properties calculated successfully.')

print('No groups:', len(properties))

print('No groups with 5 or more members:', len(properties[properties['multiplicity'] >= 5]))
MASS_A = 10

print('Calculating luminosity correction factor...')
group_ob_limit = (19.65 - cosmo.dist_mod(properties["median_redshift"]) - kcorr(properties["median_redshift"]))
int_function = build_integrated_lf()
print('Luminosity correction factor calculated successfully.')

# -10 AB cut from tyrstans github page. 
lum_factor = int_function(-10) / int_function(group_ob_limit)
properties["lum_corrected_mass"] = properties["mass_proxy"] * lum_factor
properties["lum_corrected_flux"] = properties["flux_proxies"] * lum_factor
properties["MassA"] = properties["lum_corrected_mass"] * MASS_A

print(properties)
# join the group ids to the sharks table, and the group propertes from the group catalog as well
sharks['group_id'] = group_ids
sharks = join(sharks, properties, keys='group_id', join_type='left')
# Save the processed sharks data to a parquet file
print('Saving processed sharks data to parquet...')
print(sharks.columns)

sharks = sharks['uberID','RAcen','Deccen','Z',
              'logmstar','app_mag_rt','Kcorr','abs_mag_rt',
              'group_id','iter_ra','iter_dec','iter_redshift','iter_idx','median_redshift',
              'co_dist','r50','r100','rsigma','multiplicity','velocity_dispersion_gap',
              'velocity_dispersion_gap_err','mass_proxy','bcg_idxs','bcg_ras','bcg_decs',
              'bcg_redshifts','center_of_light_ras','center_of_light_decs','total_absolute_mag',
              'flux_proxies','lum_corrected_mass','lum_corrected_flux','MassA']


output_path = '/Users/sp624AA/Downloads/mocks/sharks_nessie_groups.parquet'
sharks.write(output_path, format='parquet', overwrite=True)

plt.hist(sharks['abs_mag_rt'], bins=50, color='blue', alpha=0.7, log =True)
plt.xlabel('Absolute Magnitude (r band)')
plt.ylabel('Number of Galaxies')
plt.title('Distribution of Absolute Magnitudes in sharks')

plt.show()

plt.hist(sharks['Z'], bins=50, color='green', alpha=0.7)
plt.xlabel('Redshift')
plt.ylabel('Number of Galaxies')
plt.title('Distribution of Redshifts in sharks')
plt.show()

plt.hist(sharks['app_mag_rt'], bins=50, color='red', alpha=0.7) 
plt.xlabel('Apparent Magnitude (r band)')
plt.ylabel('Number of Galaxies')
plt.title('Distribution of Apparent Magnitudes in sharks')
plt.show()

plt.hist(sharks['Kcorr'], bins=50, color='purple', alpha=0.7, log=True)
plt.xlabel('K Correction')
plt.ylabel('Number of Galaxies')
plt.title('Distribution of K Corrections in sharks')
plt.show()


