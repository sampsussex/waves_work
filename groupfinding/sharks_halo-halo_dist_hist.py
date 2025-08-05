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
        print("Converting coordinates to 3D...")
        
        # Convert all halos to 3D coordinates
        coords_3d = self.ra_dec_z_to_cartesian(df['ra'], df['dec'], df['zcos'])
        df['x'] = coords_3d[:, 0]
        df['y'] = coords_3d[:, 1] 
        df['z'] = coords_3d[:, 2]
        
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
            all_displacements = []
            
            for idx, halo in tqdm(query_halos.iterrows(), 
                                total=len(query_halos),
                                desc=f"Bin {bin_idx}"):
                
                # Get the tree index for this halo
                tree_idx = halo['tree_index']
                
                halo_pos = np.array([halo['x'], halo['y'], halo['z']])
                halo_vel = np.array([halo['vpec_x'], halo['vpec_y'], halo['vpec_z']])
                rvir = halo['virial_radius']
                
                # Query neighbors within 3 * virial radius
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
                all_displacements.extend(r_scaled)
            
            results[bin_idx] = {
                'r_scaled': np.array(all_r_scaled),
                'v_offset': np.array(all_v_offset),
                'displacements': np.array(all_displacements),
                'n_halos': len(query_halos),
                'n_pairs': len(all_r_scaled)
            }
            
            print(f"Bin {bin_idx}: {len(all_r_scaled)} halo pairs")
        
        return results
    
    def create_density_plots(self, results, figsize=None):
        """
        Create density plots (n_bins × 2 plot types) in 2x1 vertical arrangement
        where n_bins is determined from self.mass_bin_labels
        
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
        
        # Color map for density plots
        cmap = plt.cm.viridis
        
        for bin_idx in range(n_bins):
            row = bin_idx
            
            # Add mass bin title spanning both columns
            # Adjust y position based on number of bins
            #y_pos = 0.95 - (bin_idx * (0.9 / n_bins))
            #fig.text(0.5, y_pos, self.mass_bin_labels[bin_idx], 
            #        ha='center', va='top', fontsize=14, fontweight='bold')
            
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
            displacements = data['displacements']
            
            # Left column: 2D density plots (r_scaled vs v_offset)
            ax1 = axes[row, 0]
            axes[row, 0].set_title(self.mass_bin_labels[bin_idx], fontsize=14, fontweight='bold', pad=20)
            if len(r_scaled) > 0:
                # Filter data to reasonable ranges
                mask = (r_scaled <= 3.0) & (v_offset <= np.percentile(v_offset, 99))
                r_filt = r_scaled[mask]
                v_filt = v_offset[mask]
                
                if len(r_filt) > 10:
                    # Create 2D histogram
                    hist, xedges, yedges = np.histogram2d(r_filt, v_filt, bins=50)
                    hist = np.ma.masked_where(hist == 0, hist)
                    
                    im = ax1.imshow(hist.T, origin='lower', aspect='auto',
                                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                cmap=cmap,)
                    
                    # Add colorbar for first plot
                   # if bin_idx == 0:
                    cbar = plt.colorbar(im, ax=ax1, pad=0.02, shrink=0.8)
                    cbar.set_label('Density', fontsize=12)
            
            ax1.set_xlabel('$r / R_{vir}$', fontsize=12)
            ax1.set_ylabel('$|\\Delta v|$ [km/s]', fontsize=12)
            ax1.set_xlim(0, 2)
            ax1.text(0.05, 0.95, f'{data["n_pairs"]} pairs', 
                    transform=ax1.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
            
            # Right column: Displacement histograms
            ax2 = axes[row, 1]
            
            if len(displacements) > 0:
                # Convert to Mpc and create histogram
                disp_mpc = displacements
                
                ax2.hist(disp_mpc, bins=30, alpha=0.7, density=True, 
                        color='steelblue', edgecolor='black', linewidth=0.5)
                
                ax2.axvline(np.median(disp_mpc), color='red', linestyle='--', 
                        label=f'Median: {np.median(disp_mpc):.2f} Mpc')
                
                ax2.legend(fontsize=10, loc='upper right')
            
            ax2.set_xlabel('Halo-Halo Distance [Mpc]', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.text(0.05, 0.95, f'{data["n_halos"]} central halos', 
                    transform=ax2.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.04, 
                        hspace=0.4, wspace=0.3)
        
        return fig
    
    def run_analysis(self, halo_file_path, galaxy_file_path, output_path='halo_density_plots.png'):
        """
        Run complete analysis pipeline
        
        Parameters:
        -----------
        halo_file_path : str
            Path to halo parquet file
        galaxy_file_path : str
            Path to galaxy parquet file
        output_path : str
            Path to save output plot
        """
        print("Starting halo density analysis...")
        
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
            print(f"  Mean r/R_vir: {np.mean(data['r_scaled']):.2f}")
            print(f"  Mean |Δv|: {np.mean(data['v_offset']):.1f} km/s")
        print()
    
    plt.show()
