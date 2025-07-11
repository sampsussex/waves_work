import numpy as np
from  nessie  import  FlatCosmology, RedshiftCatalog
from  nessie.helper_funcs  import  create_density_function
from astropy.table import Table
# Preparing redshift data
from collections import Counter

data = Table.read('/Users/sp624AA/Downloads/mocks/fs2_subselection.parquet')

ra = data['ra_gal']
dec = data['dec_gal']
redshifts = data['observed_redshift_gal']
mock_group_ids = data['halo_id']

ra_min, ra_max, dec_min, dec_max = 160, 170, 5, 10




area_total = (np.radians(ra_max) - np.radians(ra_min)) * \
    (np.sin(np.radians(dec_max)) - np.sin(np.radians(dec_min)))


area_sq_deg = area_total * (180 / np.pi)**2
survey_fractional_area = area_sq_deg / (360**2 / np.pi) 


#ra, dec, redshifts = np.loadtxt('some_redshift_survey.csv')
cosmo = FlatCosmology(h = 0.7, omega_matter = 0.3)
running_density = create_density_function(redshifts, total_counts = len(redshifts), 
                                          survey_fractional_area = survey_fractional_area, cosmology = cosmo)

# Running group catalogue
red_cat = RedshiftCatalog(ra, dec, redshifts, running_density, cosmo)
red_cat.run_fof(b0 = 0.06, r0 = 16)

#print(mock_group_ids[0:10])

# Reassign values with only 1 occurrence to -1
mock_group_id_counts = Counter(mock_group_ids)
mock_group_ids = np.array([group_id if mock_group_id_counts[group_id] > 1 else -1 for group_id in mock_group_ids])
mock_group_ids = mock_group_ids.astype(int)

# Reassign mock_group_ids with integers increasing from 1 where mock_group_ids is not -1
unique_ids = np.unique(mock_group_ids[mock_group_ids != -1])
id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
mock_group_ids = np.array([id_mapping[group_id] if group_id in id_mapping else -1 for group_id in mock_group_ids])

print(np.array(mock_group_ids[0:10]))

#group_ids = red_cat.group_ids
red_cat.mock_group_ids = mock_group_ids

print(red_cat.mock_group_ids)

#mock_group_ids = np.loadtxt("some_mock.txt", usecols=(1), unpack=True)
#red_cat.mock_groups_ids = mock_group_ids
score = red_cat.compare_to_mock(min_group_size=5)

print(score)
 # Run emcee to optiimse the b0, r0 params. 
import emcee

def log_probability(params, red_cat):
    b0, r0 = params

    red_cat.run_fof(b0=b0, r0=r0)
    score = red_cat.compare_to_mock(min_group_size=5)
    
    return score

def log_prior(params):
    b0, r0 = params
    if 0 < b0 < 1 and 0 < r0 < 100:
        return 0.0  # Uniform prior
    return -np.inf  # Outside prior bounds



def log_likelihood(params, red_cat):
    return log_probability(params, red_cat)


def log_posterior(params, red_cat):
    return log_prior(params) + log_likelihood(params, red_cat)

def run_emcee(red_cat, nwalkers=32, nsteps=1000):
    ndim = 2  # Number of parameters (b0, r0)


    #set intial position for walker as guassingly distrubited around b0 = 0.06 and r0 = 16
    initial = np.random.normal(loc=[0.06, 16], scale=[0.01, 0.1], size=(nwalkers, ndim))

    # Ensure initial positions are within prior bounds
    initial = np.clip(initial, [0, 0], [1, 100])


    print(initial)


    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[red_cat], )

    # Run the MCMC sampler
    sampler.run_mcmc(initial, nsteps, progress=True)

    return sampler

sampler = run_emcee(red_cat)

import corner
import matplotlib.pyplot as plt

samples = sampler.get_chain(flat=True, discard=100)
plt.figure(figsize=(10, 7))
corner.corner(samples, labels=["b0", "r0"])
plt.show()