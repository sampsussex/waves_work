import numpy as np
from  nessie  import  FlatCosmology, RedshiftCatalog
from  nessie.helper_funcs  import  create_density_function
from astropy.table import Table
from collections import Counter
from scipy.stats import skewnorm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import logging
galform_path = '/Users/sp624AA/Downloads/mocks/GALFORM/G3CMockGalv04.fits'

# Load the data
print(f"Loading data from {galform_path}")
galform_data = Table.read(galform_path)

galform_data = galform_data[galform_data['Volume'] == 9]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your existing data loading code...
data_z_02 = galform_data[galform_data['Zspec'] < 0.2]
data_all_z = galform_data#[galform_data['Rpetro'] < 19.65]
data_all_z_1965 =  data_all_z[data_all_z['Rpetro'] < 19.65]


def score_catalog(data, name, h, omega_m, plot_z_pdfs=False):
    print('Running Cat for:', name, 'h=', h, 'omega_m=', omega_m)

    # Extract necessary columns
    ra = data['RA']
    dec = data['DEC']
    redshifts = data['Zspec']
    mock_group_ids = data['HaloID']

    # Calculate survey area
    area_sq_deg = 144
    survey_fractional_area = area_sq_deg / (360**2 / np.pi)

    # Fit skew normal to redshift distribution and plot if true
    a_hat, loc_hat, scale_hat = skewnorm.fit(redshifts)
    sampled_redshift = skewnorm.rvs(a_hat, loc=loc_hat, scale=scale_hat, size=len(data))
    x = np.linspace(min(sampled_redshift), max(sampled_redshift), 500)
    pdf = skewnorm.pdf(x, a_hat, loc=loc_hat, scale=scale_hat)

    if plot_z_pdfs == True:
        plt.clf()
        plt.hist(redshifts, bins=50, density=True, alpha=0.5, label="original zs")
        plt.hist(sampled_redshift, bins=50, density=True, alpha=0.5, label="sampled zs")
        plt.plot(x, pdf, 'r-', lw=2, label="fitted PDF")
        plt.legend()
        plt.savefig(f'pdfs{name}')




    # Set up nessie in a simple fashion. 
    cosmo = FlatCosmology(h = h, omega_matter = omega_m)
    completeness = np.repeat(1.0, len(redshifts)) 

    running_density = create_density_function(sampled_redshift, total_counts = len(sampled_redshift), 
                                              survey_fractional_area = survey_fractional_area, cosmology = cosmo)
    red_cat = RedshiftCatalog(ra, dec, redshifts, running_density, cosmo)
    red_cat.calculate_completeness(ra, dec, completeness)

    # Run fof with Trystan's parameters
    red_cat.run_fof(b0 = 0.04, r0 = 36)

    red_cat.mock_group_ids = mock_group_ids
    score = red_cat.compare_to_mock(min_group_size=5)
    print(f"Score for {name} (h={h}, Omega_m={omega_m}): {score}")
    return score

# Parameters to test. 
hs = [0.7, 1]
Omega_ms = [0.3, 0.25]
data_tables = [data_z_02, data_all_z, data_all_z_1965]
names = ['data_z_02', 'data_all_z', 'data_all_z_1965']


for i, d in enumerate(data_tables):
    for j in range(len(hs)):
        try:
            score_catalog(d, names[i], hs[j], Omega_ms[j])
        except Exception as e:
            logger.error(f"Error processing data with h={hs[j]} and Omega_m={Omega_ms[j]}: {e}")
    