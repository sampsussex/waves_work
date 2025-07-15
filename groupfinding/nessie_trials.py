import numpy as np
from  nessie  import  FlatCosmology, RedshiftCatalog
from  nessie.helper_funcs  import  create_density_function
from astropy.table import Table
# Preparing redshift data
from collections import Counter
import multiprocessing as mp
import os
import pickle 
import corner
import matplotlib.pyplot as plt
import logging

# Set up logging to track failures
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

#print(np.array(mock_group_ids[0:10]))

#group_ids = red_cat.group_ids
red_cat.mock_group_ids = mock_group_ids

#print(red_cat.mock_group_ids)

#mock_group_ids = np.loadtxt("some_mock.txt", usecols=(1), unpack=True)
#red_cat.mock_groups_ids = mock_group_ids
score = red_cat.compare_to_mock(min_group_size=5)

#print(score)
 # Run emcee to optiimse the b0, r0 params. 
import emcee
def log_probability(params, red_cat):
    """
    Robust log probability function with comprehensive error handling
    """
    b0, r0 = params
    
    # Hard bounds check first
    if b0 <= 0.06 or r0 <= 0:
        return -np.inf
    
    try:
        # Main computation that might fail
        red_cat.run_fof(b0=b0, r0=r0)
        score = red_cat.compare_to_mock(min_group_size=5)
        
        # Additional safety checks on the score
        if not np.isfinite(score):
            logger.warning(f"Non-finite score for params ({b0:.3f}, {r0:.3f})")
            return -np.inf
            
        return score
        
    except Exception as e:
        # Log the specific error for debugging
        logger.debug(f"Failed evaluation for params ({b0:.3f}, {r0:.3f}): {str(e)}")
        return -np.inf

def log_prior(params):
    """Prior distribution"""
    b0, r0 = params
    if 0.05 < b0 < 0.5 and 0 < r0 < 50:
        return 0.0  # Uniform prior
    return -np.inf  # Outside prior bounds

def log_likelihood(params, red_cat):
    """Wrapper for log_probability (keeping your naming convention)"""
    return log_probability(params, red_cat)

def log_posterior(params, red_cat):
    """Log posterior = log prior + log likelihood"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, red_cat)

def run_emcee(red_cat, nwalkers=14, nsteps=1000, initial_params=None, moves=None):
    """
    Run EMCEE with multiprocessing and error handling
    
    Parameters:
    -----------
    initial_params : array_like
        Initial walker positions (nwalkers, ndim)
    moves : emcee move object, optional
        Custom move proposal (for advanced step size control)
    """
    ndim = 2
    ncores = max(1, os.cpu_count() - 1)
    print(f"Using {ncores} cores for MCMC")
    
    if initial_params is None:
        raise ValueError("initial_params must be provided")
    
    initial = np.array(initial_params)
    
    # Try different multiprocessing context
    ctx = mp.get_context('spawn')  # or 'fork' on Unix
    
    with ctx.Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior,
            args=[red_cat],
            pool=pool,
            moves=moves  # Custom moves if provided
        )
        
        # Run with progress tracking
        sampler.run_mcmc(initial, nsteps, progress=True)
        
        # Check acceptance rate
        acceptance_rate = np.mean(sampler.acceptance_fraction)
        print(f"Mean acceptance rate: {acceptance_rate:.3f}")
        
        if acceptance_rate < 0.2:
            print("Warning: Low acceptance rate. Consider adjusting step size or priors.")
        elif acceptance_rate > 0.7:
            print("Warning: High acceptance rate. Consider wider step size.")
            
        return sampler



def analyze_chain(sampler, burnin=None):
    """
    Analyze the MCMC chain and provide diagnostics
    """
    if burnin is None:
        burnin = sampler.chain.shape[1] // 4  # Use first quarter as burn-in
    
    # Get samples after burn-in
    samples = sampler.get_chain(discard=burnin, flat=True)
    
    # Basic statistics
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)
    
    print(f"\nChain diagnostics:")
    print(f"Burn-in discarded: {burnin} steps")
    print(f"Final samples: {len(samples)}")
    print(f"Parameter means: b0={means[0]:.4f}, r0={means[1]:.4f}")
    print(f"Parameter stds: b0={stds[0]:.4f}, r0={stds[1]:.4f}")
    
    # Check for convergence (simple autocorrelation time)
    try:
        tau = sampler.get_autocorr_time()
        print(f"Autocorrelation times: b0={tau[0]:.1f}, r0={tau[1]:.1f}")
    except:
        print("Could not compute autocorrelation time")
    
    return samples


if __name__ == '__main__':

    # # Prepare your own initial positions in known good regions:
    initial = np.random.normal(loc=[0.08, 36], scale=[0.0001, 0.001], size=(14, 2))
    initial[:, 0] = np.clip(initial[:, 0], 0.061, 0.49)
    initial[:, 1] = np.clip(initial[:, 1], 0.1, 49.9)
    
        # Save the sampler state



    sampler = run_emcee(red_cat, initial_params=initial)
    # Analyze the chain

        # Save the sampler state
    with open('mcmc_sampler.pkl', 'wb') as f:
        pickle.dump(sampler, f)


    samples = analyze_chain(sampler)

    flat_samples = samples


    plot_samples  =  sampler.get_chain()

    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    labels = ["b0", "r0"]
    ndim =2 
    for i in range(ndim):
        ax = axes[i]
        ax.plot(plot_samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(plot_samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plt.savefig("mcmc_chain.png")
    fig.clf()


    #plot mcmc chain score
    plt.figure(figsize=(10, 7))
    plt.plot(sampler.get_log_prob(flat=True), color='k', alpha=0.1)
    plt.xlabel("Step")
    plt.ylabel("Log Probability")
    plt.title("MCMC Chain Log Probability")
    plt.savefig("mcmc_chain_log_prob.png")


    fig = corner.corner(flat_samples, labels=["b0", "r0"])
    plt.figure(figsize=(10, 7))
    plt.savefig("corner_plot.png")
    plt.show()

