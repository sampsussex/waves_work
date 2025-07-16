import numpy as np
from  nessie  import  FlatCosmology, RedshiftCatalog
from  nessie.helper_funcs  import  create_density_function
from astropy.table import Table
from collections import Counter
import os
import pickle 
import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import logging
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import emcee

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your existing data loading code...
data = Table.read('/its/home/sp624/waves_general/waves_work/groupfinding/fs2_subselection.parquet')

ra = data['ra_gal']
dec = data['dec_gal']
redshifts = data['observed_redshift_gal']
mock_group_ids = data['halo_id']

ra_min, ra_max, dec_min, dec_max = 160, 170, 5, 10

area_total = (np.radians(ra_max) - np.radians(ra_min)) * \
    (np.sin(np.radians(dec_max)) - np.sin(np.radians(dec_min)))

area_sq_deg = area_total * (180 / np.pi)**2
survey_fractional_area = area_sq_deg / (360**2 / np.pi) 

cosmo = FlatCosmology(h = 0.7, omega_matter = 0.3)
running_density = create_density_function(redshifts, total_counts = len(redshifts), 
                                          survey_fractional_area = survey_fractional_area, cosmology = cosmo)

# Running group catalogue
red_cat = RedshiftCatalog(ra, dec, redshifts, running_density, cosmo)
red_cat.run_fof(b0 = 0.06, r0 = 16)

# Process mock group IDs
mock_group_id_counts = Counter(mock_group_ids)
mock_group_ids = np.array([group_id if mock_group_id_counts[group_id] > 1 else -1 for group_id in mock_group_ids])
mock_group_ids = mock_group_ids.astype(int)

unique_ids = np.unique(mock_group_ids[mock_group_ids != -1])
id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
mock_group_ids = np.array([id_mapping[group_id] if group_id in id_mapping else -1 for group_id in mock_group_ids])

red_cat.mock_group_ids = mock_group_ids

# Thread-safe lock for Rust code access
rust_lock = threading.Lock()

def log_probability_threadsafe(params, red_cat, thread_id=None):
    """
    Thread-safe version of log probability function
    """
    b0, r0 = params
    
    # Hard bounds check first
    if b0 <= 0.05 or r0 <= 0:
        return -np.inf
    
    try:
        # Use lock to ensure thread-safe access to Rust code
        with rust_lock:
            red_cat.run_fof(b0=b0, r0=r0)
            score = red_cat.compare_to_mock(min_group_size=5)
        
        # Additional safety checks on the score
        if not np.isfinite(score):
            if thread_id:
                logger.warning(f"Thread {thread_id}: Non-finite score for params ({b0:.3f}, {r0:.3f})")
            return -np.inf
            
        return score
        
    except Exception as e:
        if thread_id:
            logger.debug(f"Thread {thread_id}: Failed evaluation for params ({b0:.3f}, {r0:.3f}): {str(e)}")
        else:
            logger.debug(f"Failed evaluation for params ({b0:.3f}, {r0:.3f}): {str(e)}")
        return -np.inf

def log_prior(params):
    """Prior distribution"""
    b0, r0 = params
    if 0.05 < b0 < 0.5 and 0 < r0 < 50:
        return 0.0  # Uniform prior
    return -np.inf  # Outside prior bounds

def log_likelihood(params, red_cat):
    """Thread-safe wrapper for log_probability"""
    thread_id = threading.current_thread().ident
    return log_probability_threadsafe(params, red_cat, thread_id)

def log_posterior(params, red_cat):
    """Log posterior = log prior + log likelihood"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, red_cat)

# Custom ThreadPoolExecutor for emcee
class EmceeThreadPool:
    """Custom thread pool that mimics multiprocessing.Pool interface for emcee"""
    
    def __init__(self, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
    
    def map(self, func, iterable):
        """Map function over iterable using threads"""
        return list(self.executor.map(func, iterable))
    
    def close(self):
        """Close the thread pool"""
        self.executor.shutdown(wait=False)
    
    def join(self):
        """Wait for all threads to complete"""
        self.executor.shutdown(wait=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

def run_emcee_threaded(red_cat, nwalkers=32, nsteps=1000, initial_params=None, nthreads=32):
    """
    Run EMCEE with threading instead of multiprocessing
    """
    ndim = 2
    
    # Use specified number of threads or environment variable
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        nthreads = int(os.environ['SLURM_CPUS_PER_TASK'])
    elif 'PBS_NUM_PPN' in os.environ:
        nthreads = int(os.environ['PBS_NUM_PPN'])
    
    print(f"Using {nthreads} threads for MCMC")
    logger.info(f"Starting threaded MCMC with {nwalkers} walkers, {nsteps} steps, {nthreads} threads")
    
    if initial_params is None:
        raise ValueError("initial_params must be provided")
    
    initial = np.array(initial_params)
    start_time = time.time()
    
    # Use custom thread pool
    with EmceeThreadPool(nthreads) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior,
            args=[red_cat],
            pool=pool
        )
        
        # Run with progress tracking
        sampler.run_mcmc(initial, nsteps, progress=True)
        
        # Check acceptance rate
        acceptance_rate = np.mean(sampler.acceptance_fraction)
        elapsed_time = time.time() - start_time
        
        print(f"MCMC completed in {elapsed_time:.2f} seconds")
        print(f"Mean acceptance rate: {acceptance_rate:.3f}")
        logger.info(f"Threaded MCMC completed. Acceptance rate: {acceptance_rate:.3f}, Time: {elapsed_time:.2f}s")
        
        if acceptance_rate < 0.2:
            print("Warning: Low acceptance rate. Consider adjusting step size or priors.")
            logger.warning("Low acceptance rate detected")
        elif acceptance_rate > 0.7:
            print("Warning: High acceptance rate. Consider wider step size.")
            logger.warning("High acceptance rate detected")
            
        return sampler

def run_emcee_single_threaded(red_cat, nwalkers=32, nsteps=1000, initial_params=None):
    """
    Run EMCEE single-threaded (safest option)
    """
    ndim = 2
    
    print("Running single-threaded MCMC")
    logger.info(f"Starting single-threaded MCMC with {nwalkers} walkers, {nsteps} steps")
    
    if initial_params is None:
        raise ValueError("initial_params must be provided")
    
    initial = np.array(initial_params)
    start_time = time.time()
    
    # No pool - single threaded
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=[red_cat]
    )
    
    # Run with progress tracking
    sampler.run_mcmc(initial, nsteps, progress=True)
    
    # Check acceptance rate
    acceptance_rate = np.mean(sampler.acceptance_fraction)
    elapsed_time = time.time() - start_time
    
    print(f"MCMC completed in {elapsed_time:.2f} seconds")
    print(f"Mean acceptance rate: {acceptance_rate:.3f}")
    logger.info(f"Single-threaded MCMC completed. Acceptance rate: {acceptance_rate:.3f}, Time: {elapsed_time:.2f}s")
    
    return sampler

def analyze_chain(sampler, burnin=None):
    """Analyze the MCMC chain and provide diagnostics"""
    if burnin is None:
        burnin = sampler.chain.shape[1] // 4
    
    samples = sampler.get_chain(discard=burnin, flat=True)
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)
    
    print(f"\nChain diagnostics:")
    print(f"Burn-in discarded: {burnin} steps")
    print(f"Final samples: {len(samples)}")
    print(f"Parameter means: b0={means[0]:.4f}, r0={means[1]:.4f}")
    print(f"Parameter stds: b0={stds[0]:.4f}, r0={stds[1]:.4f}")
    
    logger.info(f"Chain analysis: {len(samples)} samples, means: b0={means[0]:.4f}, r0={means[1]:.4f}")
    
    try:
        tau = sampler.get_autocorr_time()
        print(f"Autocorrelation times: b0={tau[0]:.1f}, r0={tau[1]:.1f}")
        logger.info(f"Autocorrelation times: b0={tau[0]:.1f}, r0={tau[1]:.1f}")
    except:
        print("Could not compute autocorrelation time")
        logger.warning("Could not compute autocorrelation time")
    
    return samples

def save_checkpoint(sampler, filename='mcmc_sampler.pkl'):
    """Save sampler state"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(sampler, f)
        logger.info(f"Checkpoint saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def create_plots(sampler, samples):
    """Create and save plots"""
    try:
        # Chain plot
        plot_samples = sampler.get_chain()
        fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
        labels = ["b0", "r0"]
        ndim = 2 
        for i in range(ndim):
            ax = axes[i]
            ax.plot(plot_samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(plot_samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.tight_layout()
        plt.savefig("mcmc_chain.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Chain plot saved")

        # Log probability plot
        plt.figure(figsize=(10, 7))
        plt.plot(sampler.get_log_prob(flat=True), color='k', alpha=0.1)
        plt.xlabel("Step")
        plt.ylabel("Log Probability")
        plt.title("MCMC Chain Log Probability")
        plt.tight_layout()
        plt.savefig("mcmc_chain_log_prob.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Log probability plot saved")

        # Corner plot
        fig = corner.corner(samples, labels=["b0", "r0"])
        plt.savefig("corner_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Corner plot saved")
        
    except Exception as e:
        logger.error(f"Error creating plots: {e}")

if __name__ == '__main__':
    try:
        # Set up initial positions
        np.random.seed(42)
        initial = np.random.normal(loc=[0.08, 36], scale=[0.0001, 0.001], size=(32, 2))
        initial[:, 0] = np.clip(initial[:, 0], 0.061, 0.49)
        initial[:, 1] = np.clip(initial[:, 1], 0.1, 49.9)
        
        logger.info("Starting MCMC run")
        
        # Choose your approach:
        
        # Option 1: Single-threaded (safest, start with this)
        #print("=== Testing Single-Threaded Approach ===")
        #sampler = run_emcee_single_threaded(red_cat, nwalkers=32, nsteps=100, initial_params=initial)
        
        # Option 2: Multi-threaded (if single-threaded works)
        print("=== Testing Multi-Threaded Approach ===")
        sampler = run_emcee_threaded(red_cat, nwalkers=32, nsteps=1000, initial_params=initial, nthreads=32)
        
        # Save and analyze
        save_checkpoint(sampler, 'mcmc_sampler.pkl')
        samples = analyze_chain(sampler)
        create_plots(sampler, samples)
        
        logger.info("MCMC run completed successfully")
        
    except Exception as e:
        logger.error(f"MCMC run failed: {e}")
        sys.exit(1)

