import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
import pyarrow.parquet as pq
import astropy.units as u
from astropy.constants import c
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from numba import njit

C = c.value
H = 0.6751 #100 km/s/Mpc
H0 = H*100 #km/s/Mpc
OM_M = 0.3
OM_L = 0.7


# Set up cosmology
cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3)
def virial_radius(M_vir, z=0, delta=200, ref_density='critical', cosmology=cosmo):
    """
    Compute the virial radius from virial mass.

    Parameters
    ----------
    M_vir : float or array-like
        Virial mass in solar masses (M_sun).
    z : float
        Redshift.
    delta : float
        Overdensity threshold (default: 200).
    ref_density : str
        Reference density: 'critical' or 'mean'.
    cosmology : astropy.cosmology instance
        Cosmology to use (default: Planck15).

    Returns
    -------
    R_vir : Quantity
        Virial radius in kpc.
    """
    #M = M_vir * u.Msun
    M = np.asarray(M_vir) * u.Msun

    if ref_density == 'critical':
        rho_ref = cosmology.critical_density(z)
    elif ref_density == 'mean':
        rho_ref = cosmology.critical_density(z) * cosmology.Om(z)
    else:
        raise ValueError("ref_density must be 'critical' or 'mean'")

    # Volume of a sphere with mass M and density delta * rho_ref
    R = (3 * M / (4 * np.pi * delta * rho_ref))**(1/3)
    
    return R.to(u.Mpc).value


@njit
def inverse_hubble(z):
    """ Get the inverse of the Hubble parameter at redshift z. Flat LCDM Cosmology specified in global variables. In units of Mpc/(km/s)."""
    return 1/(H0 * np.sqrt(OM_M * (1. + z)**3 + OM_L))


@njit
def simpson_integrate(func, a, b, n=1000):
    """Simple Simpson's rule integration compiled with Numba"""
    if n % 2 == 1:
        n += 1  # Ensure n is even
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    
    for i in range(n + 1):
        y[i] = func(x[i])
    
    result = y[0] + y[n]
    
    for i in range(1, n, 2):
        result += 4 * y[i]
    
    for i in range(2, n, 2):
        result += 2 * y[i]
    
    return result * h / 3

@njit
def comoving_distance(z):
    """
    Get the comoving distance at redshift z. Flat LCDM Cosmology specified in global variables.


    Parameters:
        z (float): redshift

    Returns:
        float: Comoving distance in Mpc
    """
    integral_result = simpson_integrate(inverse_hubble, 0., z)
    return C * integral_result/10**3

@njit
def get_all_comoving_distance(z_array):
    """
    Get all the comoving distance at redshift z of an array. Flat LCDM Cosmology specified in global variables.


    Parameters:
        z array(float): redshift

    Returns:
        array(float): Comoving distance in Mpc h ^-1
    """

    dms = np.zeros(len(z_array))
    for i in range(len(z_array)):
        dms[i] = comoving_distance(z_array[i])
    return dms

@njit
def angular_sep(ra1, dec1, ra2, dec2):
    """
    Get the angular seperation between 2 RA/Dec coordinate pairs.


    Parameters:
        ra1 (float): Right ascension in degrees of 1st object
        dec1 (float): Declination in degrees of 1st object
        ra2 (float): Right ascension in degrees of 2nd object
        dec2 (float): Declination in degrees of 2nd object

    Returns:
        float: Angular seperation in degrees
    """
        
    d_ra = np.radians(ra2 - ra1)
    d_dec = np.radians(dec2 - dec1)
    dec1 = np.radians(dec1)
    dec2 = np.radians(dec2)

    a = np.sin(d_dec / 2.0)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra / 2.0)**2
    c = 2. * np.arcsin(np.sqrt(a))
    return np.degrees(c)


@njit
def find_projected_separation(ra1, dec1, ra2, dec2, z1):
    """
    Get the projected seperation between 2 RA/Dec coordinate pairs at a given z. 


    Parameters:
        ra1 : float
            Right ascension in degrees of 1st object
        dec1 : float
            Declination in degrees of 1st object
        ra2 : float
            Right ascension in degrees of 2nd object
        dec2 : float
            Declination in degrees of 2nd object
        z1 : float
            Redshift of the 1st object

    Returns:
        float: Projected seperation in Mpc h ^-1
    """
    
    return np.radians(angular_sep(ra1, dec1, ra2, dec2))*comoving_distance(z1)

@njit
def find_all_projected_separation(ra1, dec1, z1, ra2, dec2):
    """
    Get the projected seperation between 2 RA/Dec coordinate pairs at a given z. 


    Parameters:
        ra1 : float
            Right ascension in degrees of 1st object
        dec1 : float
            Declination in degrees of 1st object
        ra2 : float
            Right ascension in degrees of 2nd object
        dec2 : float
            Declination in degrees of 2nd object
        z1 : float
            Redshift of the 1st object

    Returns:
        float: Projected seperation in Mpc h ^-1
    """
    pss = np.zeros(len(z1))
    for i in range(len(z1)):
        pss[i] = comoving_distance(z1[i])
    return pss

@njit
def calculate_projected_separation(ra1, dec1, z1, ra2, dec2, z2):
    """
    Calculate the projected separation between two points on the sky.
    
    Parameters:
    -----------
    ra1, dec1, z1 : float or array-like
        RA, Dec (in degrees) and redshift of first point(s)
    ra2, dec2, z2 : float or array-like
        RA, Dec (in degrees) and redshift of second point(s)
        
    Returns:
    --------
    float or array-like
        Projected separation in Mpc
    """
    # Convert to arrays for uniform handling
    ra1, dec1, z1 = np.atleast_1d(ra1), np.atleast_1d(dec1), np.atleast_1d(z1)
    ra2, dec2, z2 = np.atleast_1d(ra2), np.atleast_1d(dec2), np.atleast_1d(z2)
    
    # Use average redshift for distance calculation
    z_avg = (z1 + z2) / 2.0
    
    # Get comoving distance at average redshift
    d_avg = get_all_comoving_distance(z_avg)  # in Mpc
    
    # Convert RA/Dec to radians
    ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
    ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)
    
    # Calculate angular separation using haversine formula
    # This gives the angle in radians
    dra = ra2_rad - ra1_rad
    ddec = dec2_rad - dec1_rad
    
    a = np.sin(ddec/2)**2 + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(dra/2)**2
    angular_sep = 2 * np.arcsin(np.sqrt(a))  # in radians
    
    # Convert to physical projected separation
    r_proj = d_avg * angular_sep  # small angle approximation: arc length = radius * angle
    
    return np.squeeze(r_proj) if r_proj.size == 1 else r_proj

class HaloDensityAnalyzer:
    def __init__(self, H0=67.51, Om0=0.3):
        """
        Initialize the analyzer with cosmological parameters
        
        Parameters:
        -----------
        H0 : float
            Hubble constant in km/s/Mpc
        Om0 : float
            Matter density parameter
        """
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        
        # Mass bin edges (log10 solar masses)
        self.mass_bins = [
        #    [10.5, 11.0],
        #    [11.0, 11.5], 
        #    [11.5, 12.0],
        #    [12.0, 12.5],
        #    [12.5, 13.0],
        #    [13.0, 13.5],
        #    [13.5, 14.0], 
            [14.0, 14.5],
            [14.5, 16.0]
        ]
        
        self.mass_bin_labels = [
            f"$10^{{{low}}}$ - $10^{{{high}}}$ M$_\\odot$" 
            for low, high in self.mass_bins
        ]
    
    def load_data(self, halo_file_path, galaxy_file_path):
        """
        Load halo and galaxy parquet files and filter halos with galaxies
        
        Parameters:
        -----------
        halo_file_path : str
            Path to halo parquet file
        galaxy_file_path : str
            Path to galaxy parquet file
            
        Returns:
        --------
        pd.DataFrame
            Filtered halo catalog (only halos with galaxies)
        """
        print(f"Loading halo file: {halo_file_path}")
        halo_df = pd.read_parquet(halo_file_path)
        print(f"Loaded {len(halo_df)} halos total")
        
        print(f"Loading galaxy file: {galaxy_file_path}")
        galaxy_df = pd.read_parquet(galaxy_file_path)
        print(f"Loaded {len(galaxy_df)} galaxies total")
        
        # Get unique halo IDs that have galaxies
        halos_with_galaxies = set(galaxy_df['id_group_sky'].dropna().unique())
        print(f"Found {len(halos_with_galaxies)} unique halos with galaxies")
        
        # Filter halos to only include those with galaxies
        # Assuming halo ID column exists in halo_df - adjust column name as needed
        if 'id_group_sky' in halo_df.columns:
            halo_id_col = 'id_group_sky'
        elif 'halo_id' in halo_df.columns:
            halo_id_col = 'halo_id'
        elif 'id' in halo_df.columns:
            halo_id_col = 'id'
        else:
            # Try to find a suitable ID column
            id_cols = [col for col in halo_df.columns if 'id' in col.lower()]
            if id_cols:
                halo_id_col = id_cols[0]
                print(f"Using {halo_id_col} as halo ID column")
            else:
                raise ValueError("Could not find halo ID column. Please specify the correct column name.")
        
        initial_count = len(halo_df)
        # print halo columns

        print(f"Initial halo columns: {halo_df.columns.tolist()}")
        halo_df = halo_df[halo_df[halo_id_col].isin(halos_with_galaxies)]
        final_count = len(halo_df)
        
        print(f"Filtered halos: {initial_count} -> {final_count} ({final_count/initial_count*100:.1f}% retained)")
        
        # Add log mass column for easier binning
        halo_df['log_mass'] = np.log10(halo_df['mvir'])

        halo_df['virial_radius'] = virial_radius(halo_df['mvir'], 
                                                z=halo_df['zcos'], 
                                                delta=200, 
                                                ref_density='critical', 
                                                cosmology=self.cosmo)
        
        # Reset index and keep track of original position
        halo_df.reset_index(drop=True, inplace=True)
        halo_df['tree_index'] = halo_df.index  # Store position in KD-tree
        
        return halo_df


    def assign_mass_bins(self, df):
        """
        Assign halos to mass bins
        
        Parameters:
        -----------
        df : pd.DataFrame
            Halo catalog with log_mass column
            
        Returns:
        --------
        dict
            Dictionary with mass bin indices as keys, DataFrames as values
        """
        binned_halos = {}
        
        for i, (low, high) in enumerate(self.mass_bins):
            mask = (df['log_mass'] >= low) & (df['log_mass'] < high)
            binned_halos[i] = df[mask].copy()
            print(f"Mass bin {i} ({low}-{high}): {np.sum(mask)} halos")
        
        return binned_halos

    def calculate_density_maps(self, df, max_radius_factor=2.0):
        """
        Calculate stacked density maps for each mass bin using projected separations
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full halo catalog
        max_radius_factor : float
            Maximum radius in units of virial radius
            
        Returns:
        --------
        dict
            Results for each mass bin
        """
        print("Building spatial index for neighbor queries...")
        
        # For projected separations, we'll use RA/Dec directly
        # Create a simple index based on RA/Dec for rough spatial queries
        # We'll do a coarse filtering first, then calculate exact projected distances
        
        # Assign halos to mass bins
        binned_halos = self.assign_mass_bins(df)
        
        results = {}
        
        for bin_idx in range(len(self.mass_bins)):
            print(f"\nProcessing mass bin {bin_idx}...")
            
            query_halos = binned_halos[bin_idx]
            if len(query_halos) == 0:
                continue
                
            # Arrays to store stacked data
            all_r_scaled = []
            all_v_offset = []
            all_neighbor_masses = []  # Store log10(mvir) of neighbor halos
            
            for idx, halo in tqdm(query_halos.iterrows(), 
                                total=len(query_halos),
                                desc=f"Bin {bin_idx}"):
                
                halo_ra = halo['ra']
                halo_dec = halo['dec']
                halo_z = halo['zcos']
                halo_vel = np.array([halo['vpec_x'], halo['vpec_y'], halo['vpec_z']])
                rvir = halo['virial_radius']
                
                # Estimate maximum angular separation for filtering
                # At z ~ 0.1, 1 Mpc ~ 0.5 degrees, so be generous
                max_distance_mpc = max_radius_factor * rvir
                d_halo = self.cosmo.comoving_distance(halo_z).value
                max_angle_deg = np.degrees(max_distance_mpc / d_halo) * 2  # Be conservative
                
                # Coarse filter: select halos within a box in RA/Dec
                ra_mask = np.abs(df['ra'] - halo_ra) < max_angle_deg
                dec_mask = np.abs(df['dec'] - halo_dec) < max_angle_deg
                
                # Also filter by redshift to avoid very distant pairs
                z_mask = np.abs(df['zcos'] - halo_z) < 0.1  # Reasonable redshift window
                
                potential_neighbors = df[ra_mask & dec_mask & z_mask]
                
                # Remove the central halo itself
                potential_neighbors = potential_neighbors[potential_neighbors['tree_index'] != halo['tree_index']]
                
                if len(potential_neighbors) == 0:
                    continue
                
                # Calculate projected separations for all potential neighbors
                r_proj = find_all_projected_separation(
                    np.full(len(potential_neighbors), halo_ra),
                    np.full(len(potential_neighbors), halo_dec), 
                    np.full(len(potential_neighbors), halo_z),
                    potential_neighbors['ra'].values, 
                    potential_neighbors['dec'].values,
                )
                
                # Ensure r_proj is always an array
                r_proj = np.atleast_1d(r_proj)
                
                print(r_proj)
                print(max_distance_mpc)
                # Filter by maximum projected distance
                distance_mask = r_proj <= max_distance_mpc
                
                # Ensure distance_mask is a proper boolean array
                distance_mask = np.asarray(distance_mask, dtype=bool)
                
                if not np.any(distance_mask):
                    continue
                
                # Get properties of actual neighbors - use iloc with indices
                valid_indices = np.where(distance_mask)[0]
                actual_neighbors = potential_neighbors.iloc[valid_indices]
                actual_r_proj = r_proj[distance_mask]
                
                neighbor_vels = actual_neighbors[['vpec_x', 'vpec_y', 'vpec_z']].values
                neighbor_log_masses = actual_neighbors['log_mass'].values
                
                # Scale distances by virial radius
                r_scaled = actual_r_proj / rvir
                
                # Calculate velocity offsets (magnitude of velocity difference)
                vel_differences = neighbor_vels - halo_vel
                v_offsets = np.linalg.norm(vel_differences, axis=1)
                
                # Store data
                all_r_scaled.extend(r_scaled)
                all_v_offset.extend(v_offsets)
                all_neighbor_masses.extend(neighbor_log_masses)
            
            results[bin_idx] = {
                'r_scaled': np.array(all_r_scaled),
                'v_offset': np.array(all_v_offset),
                'neighbor_masses': np.array(all_neighbor_masses),
                'n_halos': len(query_halos),
                'n_pairs': len(all_r_scaled)
            }
            
            print(f"Bin {bin_idx}: {len(all_r_scaled)} halo pairs")
        
        return results
    
    def create_density_plots(self, results, figsize=None):
        """
        Create density plots (n_bins × 2 plot types) in 2x1 vertical arrangement
        Left: density heat map, Right: mass-weighted heat map
        
        Parameters:
        -----------
        results : dict
            Results from calculate_density_maps
        figsize : tuple, optional
            Figure size. If None, will be calculated based on number of bins
        """
        # Get number of mass bins from self.mass_bin_labels
        n_bins = len(self.mass_bin_labels)
        
        # Set default figsize based on number of bins if not provided
        if figsize is None:
            figsize = (12, 4 * n_bins)
        
        # Create figure with subplots arranged as n_bins rows, 2 columns
        fig, axes = plt.subplots(n_bins, 2, figsize=figsize)
        
        # Handle case where there's only one row (axes won't be 2D)
        if n_bins == 1:
            axes = axes.reshape(1, -1)
        
        # Color maps for density plots
        density_cmap = plt.cm.viridis
        mass_cmap = plt.cm.plasma  # Different colormap for mass
        
        # Add main title indicating projected separation
        fig.suptitle('Halo Clustering Analysis (Projected Separation)', fontsize=16, y=0.995)
        
        for bin_idx in range(n_bins):
            row = bin_idx
            
            if bin_idx not in results:
                # Empty plots for bins with no data
                axes[row, 0].text(0.5, 0.5, 'No Data', 
                                ha='center', va='center', transform=axes[row, 0].transAxes)
                axes[row, 1].text(0.5, 0.5, 'No Data', 
                                ha='center', va='center', transform=axes[row, 1].transAxes)
                continue
            
            data = results[bin_idx]
            r_scaled = data['r_scaled']
            v_offset = data['v_offset']
            neighbor_masses = data['neighbor_masses']
            
            # Left column: 2D density plots (r_scaled vs v_offset)
            ax1 = axes[row, 0]
            axes[row, 0].set_title(self.mass_bin_labels[bin_idx], fontsize=14, fontweight='bold', pad=20)
            
            if len(r_scaled) > 0:
                # Filter data to reasonable ranges
                mask = (r_scaled <= 3.0) & (v_offset <= np.percentile(v_offset, 99))
                r_filt = r_scaled[mask]
                v_filt = v_offset[mask]
                
                if len(r_filt) > 10:
                    # Create 2D histogram for density
                    hist, xedges, yedges = np.histogram2d(r_filt, v_filt, bins=50)
                    hist = np.ma.masked_where(hist == 0, hist)
                    
                    im1 = ax1.imshow(hist.T, origin='lower', aspect='auto',
                                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                cmap=density_cmap)
                    
                    # Add colorbar
                    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02, shrink=0.8)
                    cbar1.set_label('Density', fontsize=10)
            
            ax1.set_xlabel('$r_{\\perp} / R_{vir}$ (projected)', fontsize=12)
            ax1.set_ylabel('$|\\Delta v|$ [km/s]', fontsize=12)
            ax1.set_xlim(0, 2)
            ax1.text(0.05, 0.95, f'{data["n_pairs"]} pairs', 
                    transform=ax1.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
            
            # Right column: Mass-weighted heat maps
            ax2 = axes[row, 1]
            
            if len(r_scaled) > 0:
                # Filter data to reasonable ranges
                mask = (r_scaled <= 3.0) & (v_offset <= np.percentile(v_offset, 99))
                r_filt = r_scaled[mask]
                v_filt = v_offset[mask]
                mass_filt = neighbor_masses[mask]
                
                if len(r_filt) > 10:
                    # Create 2D binning for average mass calculation
                    # Define bin edges (same as left plot for consistency)
                    xbins = np.linspace(0, 3.0, 51)
                    ybins = np.linspace(0, np.percentile(v_offset[mask], 99), 51)
                    
                    # Calculate average mass in each bin
                    mass_sum = np.zeros((50, 50))
                    count = np.zeros((50, 50))
                    
                    # Digitize to find which bin each point belongs to
                    xi = np.digitize(r_filt, xbins) - 1
                    yi = np.digitize(v_filt, ybins) - 1
                    
                    # Accumulate masses
                    for i in range(len(r_filt)):
                        if 0 <= xi[i] < 50 and 0 <= yi[i] < 50:
                            mass_sum[yi[i], xi[i]] += mass_filt[i]
                            count[yi[i], xi[i]] += 1
                    
                    # Calculate average mass
                    avg_mass = np.zeros_like(mass_sum)
                    nonzero = count > 0
                    avg_mass[nonzero] = mass_sum[nonzero] / count[nonzero]
                    
                    # Mask zero bins
                    avg_mass = np.ma.masked_where(count == 0, avg_mass)
                    
                    # Plot average mass heat map
                    im2 = ax2.imshow(avg_mass, origin='lower', aspect='auto',
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                                cmap=mass_cmap, vmin=10, vmax=15)  # Set reasonable mass range
                    
                    # Add colorbar
                    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02, shrink=0.8)
                    cbar2.set_label('Mean log$_{10}$(M$_{vir}$/M$_\\odot$)', fontsize=10)
            
            ax2.set_xlabel('$r_{\\perp} / R_{vir}$ (projected)', fontsize=12)
            ax2.set_ylabel('$|\\Delta v|$ [km/s]', fontsize=12)
            ax2.set_xlim(0, 2)
            ax2.text(0.05, 0.95, f'{data["n_halos"]} central halos', 
                    transform=ax2.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.04, 
                        hspace=0.4, wspace=0.3)
        
        fig.suptitle('Sharks v0.3.0 Halo Density Analysis', fontsize=16, fontweight='bold', y=0.99)
        
        return fig
    
    def run_analysis(self, halo_file_path, galaxy_file_path, output_path='halo_density_plots_projected.png'):
        """
        Run complete analysis pipeline with projected separations
        
        Parameters:
        -----------
        halo_file_path : str
            Path to halo parquet file
        galaxy_file_path : str
            Path to galaxy parquet file
        output_path : str
            Path to save output plot
        """
        print("Starting halo density analysis with projected separations...")
        
        # Load and filter data
        df = self.load_data(halo_file_path, galaxy_file_path)
        
        # Calculate density maps
        results = self.calculate_density_maps(df)
        
        # Create plots
        fig = self.create_density_plots(results)
        
        # Save figure
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {output_path}")
        
        return results, fig

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = HaloDensityAnalyzer()
    
    # Example file paths (replace with your actual paths)
    halo_file_path = '/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_groups.parquet'
    galaxy_file_path = '/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet'
    
    # Run analysis
    results, fig = analyzer.run_analysis(halo_file_path, galaxy_file_path)
    
    # Optional: Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for bin_idx, data in results.items():
        print(f"Mass bin {bin_idx}:")
        print(f"  Central halos: {data['n_halos']}")
        print(f"  Neighbor pairs: {data['n_pairs']}")
        if len(data['r_scaled']) > 0:
            print(f"  Mean r_proj/R_vir: {np.mean(data['r_scaled']):.2f}")
            print(f"  Mean |Δv|: {np.mean(data['v_offset']):.1f} km/s")
            print(f"  Mean neighbor log10(mvir): {np.mean(data['neighbor_masses']):.2f}")
        print()
    
    plt.show()