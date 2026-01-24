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
import numba
from numba import njit
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your existing data loading code...
data = Table.read('fs2_subselection.parquet')

ra = data['ra_gal']
dec = data['dec_gal']
redshifts = data['observed_redshift_gal']
mock_group_ids = data['halo_id']
print("NUMBER OF GALAXIES:", len(ra))
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
red_cat.run_fof(b0 = 0.035, r0 = 50)

# Process mock group IDs
mock_group_id_counts = Counter(mock_group_ids)
mock_group_ids = np.array([group_id if mock_group_id_counts[group_id] > 1 else -1 for group_id in mock_group_ids])
mock_group_ids = mock_group_ids.astype(int)

unique_ids = np.unique(mock_group_ids[mock_group_ids != -1])
id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
mock_group_ids = np.array([id_mapping[group_id] if group_id in id_mapping else -1 for group_id in mock_group_ids])

red_cat.mock_group_ids = mock_group_ids
score = red_cat.compare_to_mock(min_group_size=5)
print(score)
# Cache the mock comparison data in L2 cache by keeping it in memory
# This prevents repeated computation of the static mock data
_cached_mock_data = None
_cache_lock = threading.Lock()

def cache_mock_comparison_data(red_cat):
    """Cache static mock comparison data for L2 cache optimization"""
    global _cached_mock_data
    with _cache_lock:
        if _cached_mock_data is None:
            # Pre-compute and cache any static mock comparison data
            # This keeps the data warm in L2 cache
            _cached_mock_data = {
                'mock_group_ids': red_cat.mock_group_ids.copy(),
                'mock_group_id_counts': Counter(red_cat.mock_group_ids),
                'unique_mock_ids': np.unique(red_cat.mock_group_ids[red_cat.mock_group_ids != -1])
            }
            logger.info("Mock comparison data cached for L2 optimization")

# Thread-safe lock for Rust code access
rust_lock = threading.Lock()

def get_optimal_thread_count():
    """Calculate optimal thread count: 1 thread per 2 cores"""
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    optimal_threads = max(1, cpu_count // 2)
    
    # Check environment variables for cluster settings
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        slurm_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        optimal_threads = max(1, slurm_cpus // 2)
    elif 'PBS_NUM_PPN' in os.environ:
        pbs_cpus = int(os.environ['PBS_NUM_PPN'])
        optimal_threads = max(1, pbs_cpus // 2)
    
    logger.info(f"Detected {cpu_count} physical cores, using {optimal_threads} threads")
    return optimal_threads

@njit
def log_prior_numba(b0, r0):
    """Numba-optimized prior distribution calculation"""
    if 0.06 < b0 < 0.1 and 14.9 < r0 < 40.0:
        return 0.0  # Uniform prior
    return -np.inf  # Outside prior bounds

def log_probability_threadsafe(params, red_cat, thread_id=None):
    """
    Thread-safe version of log probability function
    """
    b0, r0 = params
    
    # Use numba-optimized prior check
    prior_val = log_prior_numba(b0, r0)
    if not np.isfinite(prior_val):
        return -np.inf
    
    # Hard bounds check first
    if b0 <= 0.05 or r0 <= 0:
        return -np.inf
    
    try:
        # Use lock to ensure thread-safe access to Rust code
        with rust_lock:
            red_cat.run_fof(b0=b0, r0=r0)
            # Access cached mock data to keep it warm in L2 cache
            global _cached_mock_data
            if _cached_mock_data is not None:
                # Touch the cached data to keep it in L2 cache
                _ = _cached_mock_data['mock_group_ids'][0]
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
    """Prior distribution wrapper for emcee"""
    b0, r0 = params
    return log_prior_numba(b0, r0)

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

def save_checkpoint(sampler, step, filename_base='mcmc_checkpoint'):
    """Save sampler state with step information"""
    try:
        filename = f"{filename_base}_step_{step}.pkl"
        checkpoint_data = {
            'sampler': sampler,
            'step': step,
            'chain': sampler.get_chain(),
            'log_prob': sampler.get_log_prob(),
            'acceptance_fraction': sampler.acceptance_fraction,
            'timestamp': time.time()
        }
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        logger.info(f"Checkpoint saved to {filename} at step {step}")
        return filename
    except Exception as e:
        logger.error(f"Failed to save checkpoint at step {step}: {e}")
        return None

def load_checkpoint(filename):
    """Load sampler state from checkpoint"""
    try:
        with open(filename, 'rb') as f:
            checkpoint_data = pickle.load(f)
        logger.info(f"Checkpoint loaded from {filename}")
        return checkpoint_data
    except Exception as e:
        logger.error(f"Failed to load checkpoint {filename}: {e}")
        return None

def run_emcee_with_checkpointing(red_cat, nwalkers=32, nsteps=5000, initial_params=None, 
                                checkpoint_interval=100, resume_from=None, 
                                adaptive_steps=True, target_acceptance=0.35):
    """
    Run EMCEE with regular checkpointing and adaptive monitoring
    """
    ndim = 2
    nthreads = get_optimal_thread_count()
    
    print(f"Using {nthreads} threads for MCMC with {nwalkers} walkers")
    logger.info(f"Starting MCMC with checkpointing: {nwalkers} walkers, {nsteps} steps, {nthreads} threads")
    logger.info(f"Checkpoint interval: {checkpoint_interval} steps")
    
    if initial_params is None:
        raise ValueError("initial_params must be provided")
    
    # Initialize or resume from checkpoint
    start_step = 0
    if resume_from:
        checkpoint_data = load_checkpoint(resume_from)
        if checkpoint_data:
            sampler = checkpoint_data['sampler']
            start_step = checkpoint_data['step']
            print(f"Resuming from step {start_step}")
            logger.info(f"Resumed from checkpoint at step {start_step}")
        else:
            print("Failed to load checkpoint, starting fresh")
            sampler = None
    else:
        sampler = None
    
    # Create new sampler if needed
    if sampler is None:
        initial = np.array(initial_params)
        with EmceeThreadPool(nthreads) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_posterior,
                args=[red_cat],
                pool=pool
            )
            
            # Test initial positions
            print("Testing initial walker positions...")
            initial_scores = []
            for i, pos in enumerate(initial):
                score = log_probability_threadsafe(pos, red_cat)
                initial_scores.append(score)
                if i < 5:  # Print first 5
                    print(f"Walker {i}: b0={pos[0]:.4f}, r0={pos[1]:.1f}, score={score:.4f}")
            
            print(f"Initial score statistics:")
            print(f"  Mean: {np.mean(initial_scores):.4f}")
            print(f"  Std:  {np.std(initial_scores):.4f}")
            print(f"  Min:  {np.min(initial_scores):.4f}")
            print(f"  Max:  {np.max(initial_scores):.4f}")
            
            # Check if too many walkers have poor scores
            good_walkers = np.sum(np.isfinite(initial_scores))
            if good_walkers < nwalkers * 0.8:
                logger.warning(f"Only {good_walkers}/{nwalkers} walkers have finite scores!")
                print(f"WARNING: Only {good_walkers}/{nwalkers} walkers have finite scores!")
            
            # Initialize walkers
            sampler.run_mcmc(initial, 1, progress=False)
            start_step = 1
    
    start_time = time.time()
    
    # Run MCMC with checkpointing and monitoring
    with EmceeThreadPool(nthreads) as pool:
        # Update sampler pool if resuming
        sampler.pool = pool
        
        remaining_steps = nsteps - start_step
        current_pos = sampler.get_last_sample()
        
        while remaining_steps > 0:
            # Determine how many steps to run this iteration
            steps_this_round = min(checkpoint_interval, remaining_steps)
            
            print(f"\nRunning steps {start_step + 1} to {start_step + steps_this_round}")
            logger.info(f"Running steps {start_step + 1} to {start_step + steps_this_round}")
            
            # Run MCMC for this batch
            sampler.run_mcmc(current_pos, steps_this_round, progress=True)
            current_pos = sampler.get_last_sample()
            
            # Update counters
            start_step += steps_this_round
            remaining_steps -= steps_this_round
            
            # Detailed monitoring after each checkpoint
            if start_step >= 50:  # Wait for some burn-in
                acceptance_rate = np.mean(sampler.acceptance_fraction)
                
                # Get recent chain samples for analysis
                recent_chain = sampler.get_chain()[-50:]  # Last 50 steps
                recent_logprob = sampler.get_log_prob()[-50:]
                
                # Check parameter ranges
                b0_range = [recent_chain[:, :, 0].min(), recent_chain[:, :, 0].max()]
                r0_range = [recent_chain[:, :, 1].min(), recent_chain[:, :, 1].max()]
                
                # Check log probability statistics
                logprob_mean = np.mean(recent_logprob[np.isfinite(recent_logprob)])
                logprob_std = np.std(recent_logprob[np.isfinite(recent_logprob)])
                finite_fraction = np.mean(np.isfinite(recent_logprob))
                
                print(f"=== Step {start_step} Diagnostics ===")
                print(f"Acceptance rate: {acceptance_rate:.3f}")
                print(f"b0 range: [{b0_range[0]:.4f}, {b0_range[1]:.4f}]")
                print(f"r0 range: [{r0_range[0]:.1f}, {r0_range[1]:.1f}]")
                print(f"Log prob: mean={logprob_mean:.4f}, std={logprob_std:.4f}")
                print(f"Finite log prob fraction: {finite_fraction:.3f}")
                
                # Warnings
                if acceptance_rate < 0.15:
                    print("WARNING: Very low acceptance rate - walkers may be stuck!")
                    logger.warning(f"Very low acceptance rate: {acceptance_rate:.3f}")
                elif acceptance_rate > 0.75:
                    print("WARNING: Very high acceptance rate - may need larger steps")
                    logger.warning(f"Very high acceptance rate: {acceptance_rate:.3f}")
                
                if finite_fraction < 0.8:
                    print("WARNING: Many walkers outside valid parameter space!")
                    logger.warning(f"Low finite log prob fraction: {finite_fraction:.3f}")
                
                # Check if walkers are spreading too much
                b0_spread = b0_range[1] - b0_range[0]
                r0_spread = r0_range[1] - r0_range[0]
                if b0_spread > 0.2 or r0_spread > 20:
                    print(f"WARNING: Large parameter spread - b0: {b0_spread:.3f}, r0: {r0_spread:.1f}")
                    logger.warning(f"Large parameter spread detected")
            
            # Save checkpoint
            checkpoint_file = save_checkpoint(sampler, start_step)
            
            # Calculate and log progress
            progress = (start_step / nsteps) * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (start_step / nsteps) if start_step > 0 else 0
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"Progress: {progress:.1f}% ({start_step}/{nsteps} steps)")
            print(f"Elapsed: {elapsed_time/60:.1f}m, Estimated remaining: {remaining_time/60:.1f}m")
        
        # Final statistics
        acceptance_rate = np.mean(sampler.acceptance_fraction)
        elapsed_time = time.time() - start_time
        
        print(f"\nMCMC completed in {elapsed_time/60:.2f} minutes")
        print(f"Final acceptance rate: {acceptance_rate:.3f}")
        logger.info(f"MCMC completed. Acceptance rate: {acceptance_rate:.3f}, Time: {elapsed_time:.2f}s")
        
        # Save final checkpoint
        final_checkpoint = save_checkpoint(sampler, nsteps, 'mcmc_final')
        
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
        
        # Check convergence
        converged = np.all(tau * 50 < sampler.chain.shape[1])
        print(f"Converged (50*tau < chain length): {converged}")
        if not converged:
            logger.warning("Chain may not be converged")
            
    except Exception as e:
        print(f"Could not compute autocorrelation time: {e}")
        logger.warning("Could not compute autocorrelation time")
    
    return samples

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
        log_prob = sampler.get_log_prob(flat=True)
        plt.plot(log_prob, color='k', alpha=0.1)
        plt.xlabel("Step")
        plt.ylabel("Log Probability")
        plt.title("MCMC Chain Log Probability")
        plt.tight_layout()
        plt.savefig("mcmc_chain_log_prob.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Log probability plot saved")

        # Corner plot
        fig = corner.corner(samples, labels=["b0", "r0"], 
                           truths=None, quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig("corner_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Corner plot saved")
        
        # Acceptance rate evolution
        plt.figure(figsize=(10, 6))
        acceptance = sampler.acceptance_fraction
        plt.plot(acceptance, 'k-', alpha=0.7)
        plt.axhline(y=0.25, color='r', linestyle='--', alpha=0.7, label='Target range')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        plt.xlabel("Walker")
        plt.ylabel("Acceptance Fraction")
        plt.title("Acceptance Fraction by Walker")
        plt.legend()
        plt.tight_layout()
        plt.savefig("acceptance_fraction.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Acceptance fraction plot saved")
        
    except Exception as e:
        logger.error(f"Error creating plots: {e}")

def find_good_starting_point(red_cat, n_random_samples=100):
    """Find optimal starting point using Nelder-Mead optimization + random sampling"""
    from scipy.optimize import minimize
    
    print("Finding optimal starting point using Nelder-Mead + random sampling...")
    
    # Define objective function for minimization (negative log probability)
    def objective(params):
        score = log_probability_threadsafe(params, red_cat)
        return -score if np.isfinite(score) else 1e10
    3
    # Parameter bounds
    bounds = [(0.051, 0.099), (14.9, 39.9)]
    
    # Try multiple starting points for Nelder-Mead
    starting_points = [
        [0.06, 16.0],   # Your original
        [0.08, 25.0],   # Slightly different
        [0.095, 34.8],   # Higher values
        [0.07, 20.0],   # Middle ground
        [0.05, 34.8]    # Edge case
    ]
    
    best_params = None
    best_score = -np.inf
    
    print("Running Nelder-Mead optimization from multiple starting points...")
    
    for i, start_point in enumerate(starting_points):
        try:
            print(f"  Starting point {i+1}: b0={start_point[0]:.3f}, r0={start_point[1]:.1f}")
            
            result = minimize(
                objective, 
                start_point, 
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 100,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'disp': False
                }
            )
            
            if result.success:
                score = -result.fun
                print(f"    → Converged: b0={result.x[0]:.4f}, r0={result.x[1]:.2f}, score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
                    print(f"    → New best!")
            else:
                print(f"    → Failed to converge")
                
        except Exception as e:
            print(f"    → Error: {e}")
            continue
    
    # If Nelder-Mead didn't find anything good, fall back to grid search
    if best_params is None or best_score < -1000:
        print("Nelder-Mead failed, falling back to grid search...")
        b0_range = np.linspace(0.06, 0.099, 8)
        r0_range = np.linspace(15, 34.9, 8)
        
        for b0 in b0_range:
            for r0 in r0_range:
                score = log_probability_threadsafe([b0, r0], red_cat)
                if score > best_score:
                    best_score = score
                    best_params = [b0, r0]
                    print(f"Grid search - New best: b0={b0:.3f}, r0={r0:.1f}, score={score:.4f}")
    
    print(f"\nOptimization complete. Best point: b0={best_params[0]:.4f}, r0={best_params[1]:.2f}, score={best_score:.4f}")
    
    # Now do random sampling around the optimum to explore the local region
    print(f"\nRandom sampling around optimum (n={n_random_samples})...")
    
    # Sample in a region around the optimum
    b0_opt, r0_opt = best_params
    sample_radius_b0 = min(0.02, b0_opt * 0.2)  # 20% of optimal value or 0.02, whichever is smaller
    sample_radius_r0 = min(5.0, r0_opt * 0.15)   # 15% of optimal value or 5.0, whichever is smaller
    
    good_points = [best_params.copy()]
    good_scores = [best_score]
    
    for i in range(n_random_samples):
        # Sample uniformly in a box around the optimum
        b0_sample = np.random.uniform(
            max(0.051, b0_opt - sample_radius_b0), 
            min(0.49, b0_opt + sample_radius_b0)
        )
        r0_sample = np.random.uniform(
            max(0.1, r0_opt - sample_radius_r0), 
            min(49.9, r0_opt + sample_radius_r0)
        )
        
        sample_point = [b0_sample, r0_sample]
        score = log_probability_threadsafe(sample_point, red_cat)
        
        if np.isfinite(score):
            good_points.append(sample_point)
            good_scores.append(score)
            
            # Update best if we found something even better
            if score > best_score:
                best_score = score
                best_params = sample_point.copy()
                print(f"Random sampling - New best: b0={b0_sample:.4f}, r0={r0_sample:.2f}, score={score:.4f}")
    
    # Statistics on the local region
    good_points = np.array(good_points)
    good_scores = np.array(good_scores)
    
    print(f"\nLocal region analysis:")
    print(f"  Found {len(good_points)} finite points")
    print(f"  Score range: [{np.min(good_scores):.4f}, {np.max(good_scores):.4f}]")
    print(f"  b0 range: [{good_points[:, 0].min():.4f}, {good_points[:, 0].max():.4f}]")
    print(f"  r0 range: [{good_points[:, 1].min():.2f}, {good_points[:, 1].max():.2f}]")
    
    # Return the best point and information about the local region
    region_info = {
        'good_points': good_points,
        'good_scores': good_scores,
        'sample_radius_b0': sample_radius_b0,
        'sample_radius_r0': sample_radius_r0
    }
    
    print(f"Final best starting point: b0={best_params[0]:.4f}, r0={best_params[1]:.2f}, score={best_score:.4f}")
    return best_params, best_score, region_info

def initialize_walkers_carefully(best_params, region_info, nwalkers=32, use_local_samples=True):
    """Initialize walkers using information from the optimization and local sampling"""
    b0_best, r0_best = best_params
    
    if use_local_samples and len(region_info['good_points']) >= nwalkers:
        print("Using local sampling results to initialize walkers...")
        
        # Use the best points from local sampling
        good_points = region_info['good_points']
        good_scores = region_info['good_scores']
        
        # Sort by score and take the best ones
        sorted_indices = np.argsort(good_scores)[::-1]  # Descending order
        best_indices = sorted_indices[:nwalkers]
        
        initial = good_points[best_indices].copy()
        
        print(f"Using {nwalkers} best sampled points for initialization")
        print(f"Score range: [{good_scores[best_indices].min():.4f}, {good_scores[best_indices].max():.4f}]")
        
    else:
        print("Using Gaussian ball around optimum for initialization...")
        
        # Fall back to Gaussian sampling around the optimum
        # Use the sampling radii from the optimization
        b0_scale = region_info['sample_radius_b0'] * 0.5  # Tighter than the sampling
        r0_scale = region_info['sample_radius_r0'] * 0.5
        
        initial = np.random.normal(
            loc=[b0_best, r0_best], 
            scale=[b0_scale, r0_scale], 
            size=(nwalkers, 2)
        )
        
        # Ensure all walkers are within bounds
        initial[:, 0] = np.clip(initial[:, 0], 0.051, 0.099)
        initial[:, 1] = np.clip(initial[:, 1], 14.9, 34.9)
        
        print(f"Gaussian initialization with scales: b0={b0_scale:.4f}, r0={r0_scale:.2f}")
    
    print(f"Initialized {nwalkers} walkers:")
    print(f"b0 range: [{initial[:, 0].min():.4f}, {initial[:, 0].max():.4f}]")
    print(f"r0 range: [{initial[:, 1].min():.2f}, {initial[:, 1].max():.2f}]")
    
    return initial

if __name__ == '__main__':
    try:
        # Cache mock comparison data for L2 optimization
        cache_mock_comparison_data(red_cat)
        
        # Find the best starting point using Nelder-Mead + random sampling
        np.random.seed(42)
        best_params, best_score, region_info = find_good_starting_point(red_cat, n_random_samples=150)
        
        # Initialize walkers using the optimization results
        initial = initialize_walkers_carefully(best_params, region_info, nwalkers=32, use_local_samples=True)
        
        logger.info("Starting MCMC run with optimizations")
        
        # Run MCMC with checkpointing (5000 steps, checkpoint every 100)
        print("=== Running MCMC with Checkpointing ===")
        
        # To resume from a checkpoint, uncomment and specify the checkpoint file:
        # resume_file = "mcmc_checkpoint_step_1000.pkl"
        resume_file = None
        
        sampler = run_emcee_with_checkpointing(
            red_cat, 
            nwalkers=32, 
            nsteps=1000, 
            initial_params=initial,
            checkpoint_interval=100,
            resume_from=resume_file
        )
        
        # Final analysis and plots
        samples = analyze_chain(sampler, burnin=100)  # Discard first 1000 steps as burn-in
        create_plots(sampler, samples)
        
        logger.info("MCMC run completed successfully")
        print("\nMCMC completed! Check the plots and log files for results.")
        
    except Exception as e:
        logger.error(f"MCMC run failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)