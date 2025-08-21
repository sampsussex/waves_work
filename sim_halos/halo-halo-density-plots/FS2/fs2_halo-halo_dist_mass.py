import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
import pyarrow.parquet as pq
import astropy.units as u
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

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
            [12.0, 12.5],
            [12.5, 13.0],
            [13.0, 13.5],
            [13.5, 14.0], 
            [14.0, 14.5],
            [14.5, 16.0]
        ]
        
        self.mass_bin_labels = [
            f"$10^{{{low}}}$ - $10^{{{high}}}$ M$_\\odot$" 
            for low, high in self.mass_bins
        ]
    
    def load_data(self, galaxy_file_path):
        """
        Load galaxy parquet file and extract unique halo information
        
        Parameters:
        -----------
        galaxy_file_path : str
            Path to galaxy parquet file containing both galaxy and halo info
            
        Returns:
        --------
        pd.DataFrame
            Unique halo catalog extracted from galaxy data
        """
        print(f"Loading galaxy file: {galaxy_file_path}")
        
        # Load only the columns we need to reduce memory usage
        halo_columns = [
            'halo_id', 'vx_halo', 'vy_halo', 'vz_halo', 
            'x_halo', 'y_halo', 'z_halo', 'r_halo', 'rs_halo', 
            'rvir_halo', 'true_redshift_halo', 'lm_halo', 'lmbound_halo', 'dec_gal'
        ]
        
        # Read only the columns we need
        try:
            # Try to read only required columns to save memory and time
            galaxy_df = pd.read_parquet(galaxy_file_path, columns=halo_columns)
            print(f"Loaded {len(galaxy_df)} rows with {len(halo_columns)} columns")
            galaxy_df = galaxy_df[(galaxy_df['dec_gal'] > 10) & (galaxy_df['dec_gal'] < 20)]  # Filter based on dec_gal
        except Exception as e:
            print(f"Could not read specific columns: {e}")
            print("Loading full file...")
            galaxy_df = pd.read_parquet(galaxy_file_path)
            print(f"Loaded {len(galaxy_df)} galaxies total")
        
        # Check which columns actually exist
        available_halo_cols = [col for col in halo_columns[1:] if col in galaxy_df.columns]  # Skip halo_id
        missing_cols = [col for col in halo_columns[1:] if col not in galaxy_df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
        
        print(f"Available halo columns: {available_halo_cols}")
        
        # More efficient extraction using drop_duplicates instead of groupby
        print("Extracting unique halos...")
        required_cols = ['halo_id'] + available_halo_cols
        halo_df = galaxy_df[required_cols].drop_duplicates(subset=['halo_id']).reset_index(drop=True)
        
        print(f"Extracted {len(halo_df)} unique halos")
        
        # Rename columns to match the original code expectations
        column_mapping = {
            'lmbound_halo': 'mvir',  # Use lmbound_halo as the mass (already log10)
            'true_redshift_halo': 'zcos',
            'vx_halo': 'vpec_x',
            'vy_halo': 'vpec_y', 
            'vz_halo': 'vpec_z',
            'x_halo': 'x',
            'y_halo': 'y',
            'z_halo': 'z'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in halo_df.columns:
                halo_df[new_name] = halo_df[old_name]
        
        # Create log_mass column - lmbound_halo is already log10 of mass
        if 'lmbound_halo' in halo_df.columns:
            halo_df['log_mass'] = halo_df['lmbound_halo']
        else:
            raise ValueError("lmbound_halo column not found")
        
        ## Calculate virial radius from mass and redshift
        #if 'rvir_halo' in halo_df.columns:
        #    # Use provided virial radius (convert to Mpc if needed)
        #    halo_df['virial_radius'] = halo_df['rvir_halo'] / 1000 # Convert from kpc to Mpc
        #    print("Using provided rvir_halo column")
        #else:
        # Calculate virial radius from mass
        mass_linear = 10**halo_df['log_mass']  # Convert back to linear mass
        halo_df['virial_radius'] = virial_radius(mass_linear, 
                                                z=halo_df['zcos'], 
                                                delta=200, 
                                                ref_density='critical', 
                                                cosmology=self.cosmo)
        print("Calculated virial radius from mass")
        
        # Reset index and keep track of original position
        halo_df.reset_index(drop=True, inplace=True)
        halo_df['tree_index'] = halo_df.index  # Store position in KD-tree
        
        # Print summary statistics
        print(f"\nHalo summary:")
        print(f"  Mass range (log10): {halo_df['log_mass'].min():.2f} - {halo_df['log_mass'].max():.2f}")
        print(f"  Redshift range: {halo_df['zcos'].min():.3f} - {halo_df['zcos'].max():.3f}")
        print(f"  Virial radius range [Mpc]: {halo_df['virial_radius'].min():.4f} - {halo_df['virial_radius'].max():.4f}")
        
        return halo_df

    def ra_dec_z_to_cartesian(self, ra, dec, z):
        """
        Convert RA/Dec/z coordinates to 3D Cartesian coordinates
        
        Parameters:
        -----------
        ra : array-like
            Right ascension in degrees
        dec : array-like  
            Declination in degrees
        z : array-like
            Redshift (cosmological)
            
        Returns:
        --------
        np.ndarray
            3D Cartesian coordinates in Mpc/h
        """
        # Convert to radians
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        # Get comoving distance
        dist = self.cosmo.comoving_distance(z).value
        
        # Convert to Cartesian
        x = dist * np.cos(dec_rad) * np.cos(ra_rad)
        y = dist * np.cos(dec_rad) * np.sin(ra_rad) 
        z_cart = dist * np.sin(dec_rad)
        
        return np.column_stack([x, y, z_cart])

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
        Calculate stacked density maps for each mass bin
        
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
        print("Using provided 3D coordinates...")
        
        # Use the provided 3D coordinates directly (assuming they're already in Cartesian form)
        if all(col in df.columns for col in ['x', 'y', 'z']):
            coords_3d = df[['x', 'y', 'z']].values
            print("Using provided x, y, z coordinates")
        else:
            raise ValueError("Required coordinate columns (x, y, z) not found")
        
        # Build KD-tree for efficient neighbor queries
        print("Building KD-tree...")
        tree = cKDTree(coords_3d)
        
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
                
                # Get the tree index for this halo
                tree_idx = halo['tree_index']
                
                halo_pos = np.array([halo['x'], halo['y'], halo['z']])
                halo_vel = np.array([halo['vpec_x'], halo['vpec_y'], halo['vpec_z']])
                rvir = halo['virial_radius']
                
                # Query neighbors within max_radius_factor * virial radius
                max_distance = max_radius_factor * rvir
                neighbor_indices = tree.query_ball_point(halo_pos, max_distance)
                
                # Remove the central halo itself using tree index
                neighbor_indices = [i for i in neighbor_indices if i != tree_idx]
                
                if len(neighbor_indices) == 0:
                    continue
                
                # Get neighbor properties
                neighbors = df.iloc[neighbor_indices]
                neighbor_coords = coords_3d[neighbor_indices]
                neighbor_vels = neighbors[['vpec_x', 'vpec_y', 'vpec_z']].values
                neighbor_log_masses = neighbors['log_mass'].values  # Get log10(mvir) of neighbors
                
                # Calculate displacements
                displacements = neighbor_coords - halo_pos
                distances = np.linalg.norm(displacements, axis=1)
                
                # Scale distances by virial radius
                r_scaled = distances / rvir
                
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
                'neighbor_masses': np.array(all_neighbor_masses),  # Store neighbor masses
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
                mask = (r_scaled <= 2.0) & (v_offset <= np.percentile(v_offset, 99))
                r_filt = r_scaled[mask]
                v_filt = v_offset[mask]
                
                if len(r_filt) > 10:
                    # Create 2D histogram for density
                    xbins = np.linspace(0, 2.0, 51)
                    ybins = np.linspace(0, np.percentile(v_offset[mask], 99), 51)
                    
                    # Calculate counts in each bin
                    count = np.zeros((50, 50))
                    
                    # Digitize to find which bin each point belongs to
                    xi = np.digitize(r_filt, xbins) - 1
                    yi = np.digitize(v_filt, ybins) - 1
                    
                    # Accumulate counts
                    for i in range(len(r_filt)):
                        if 0 <= xi[i] < 50 and 0 <= yi[i] < 50:
                            count[yi[i], xi[i]] += 1
                    
                    # Mask zero bins
                    count = np.ma.masked_where(count == 0, count)
                    
                    # Plot density heat map with viridis cmap
                    im1 = ax1.imshow(count, origin='lower', aspect='auto',
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                                cmap=density_cmap)
                    
                    # Add colorbar
                    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02, shrink=0.8)
                    cbar1.set_label('Density', fontsize=10)
            
            ax1.set_xlabel('$r / R_{vir}$', fontsize=12)
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
                mask = (r_scaled <= 2.0) & (v_offset <= np.percentile(v_offset, 99))
                r_filt = r_scaled[mask]
                v_filt = v_offset[mask]
                mass_filt = neighbor_masses[mask]
                
                if len(r_filt) > 10:
                    # Create 2D binning for average mass calculation
                    xbins = np.linspace(0, 2.0, 51)
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
                                cmap=mass_cmap, vmin=10, vmax=16)  # Set reasonable mass range
                    
                    # Add colorbar
                    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02, shrink=0.8)
                    cbar2.set_label('Mean log$_{10}$(M$_{vir}$/M$_\\odot$)', fontsize=10)
            
            ax2.set_xlabel('$r / R_{vir}$', fontsize=12)
            ax2.set_ylabel('$|\\Delta v|$ [km/s]', fontsize=12)
            ax2.set_xlim(0, 2)
            ax2.text(0.05, 0.95, f'{data["n_halos"]} central halos', 
                    transform=ax2.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.04, 
                        hspace=0.4, wspace=0.3)
        fig.suptitle('Euclid FS2 Halo Density Analysis', fontsize=16, fontweight='bold', y=0.99)
        
        return fig
    
    def run_analysis(self, galaxy_file_path, output_path='halo_density_plots.png'):
        """
        Run complete analysis pipeline
        
        Parameters:
        -----------
        galaxy_file_path : str
            Path to galaxy parquet file containing both galaxy and halo info
        output_path : str
            Path to save output plot
        """
        print("Starting halo density analysis...")
        
        # Load and process data to extract unique halos
        df = self.load_data(galaxy_file_path)
        
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
    
    # Example file path (replace with your actual path)
    galaxy_file_path = '/Users/sp624AA/Downloads/mocks/FS2/waves_like_no_ra_dec_sel.parquet'
    
    # Run analysis
    results, fig = analyzer.run_analysis(galaxy_file_path)
    
    # Optional: Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for bin_idx, data in results.items():
        print(f"Mass bin {bin_idx}:")
        print(f"  Central halos: {data['n_halos']}")
        print(f"  Neighbor pairs: {data['n_pairs']}")
        if len(data['r_scaled']) > 0:
            print(f"  Mean r/R_vir: {np.mean(data['r_scaled']):.2f}")
            print(f"  Mean |Δv|: {np.mean(data['v_offset']):.1f} km/s")
            print(f"  Mean neighbor log10(mvir): {np.mean(data['neighbor_masses']):.2f}")
        print()
    
    plt.show()