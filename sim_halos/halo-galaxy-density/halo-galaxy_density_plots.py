import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from numba import njit
from astropy.cosmology import FlatLambdaCDM
import pyarrow.parquet as pq
import astropy.units as u
from astropy.constants import G
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set up cosmology
H = 0.6751  # Hubble parameter in units of 100 km/s/Mpc
cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3)
#G =  G.to((u.km/u.s)**2 * u.Mpc / u.M_sun)#.value  # in (km/s)^2 Mpc/M_sun

def virial_radius(M_vir, z, delta=200, ref_density='mean', cosmology=cosmo):
    """
    Compute virial radius in Mpc for mass M_vir (in Msun).
    Vectorised via numpy & astropy.
    """
    M = np.asarray(M_vir/H) * u.Msun
    if ref_density == 'critical':
        rho_ref = cosmology.critical_density(z)
    elif ref_density == 'mean':
        rho_ref = cosmology.critical_density(z) * cosmology.Om(z)
    else:
        raise ValueError("ref_density must be 'critical' or 'mean'")

    R = (3 * M / (4 * np.pi * delta * rho_ref))**(1/3)
    return R.to(u.Mpc).value

def vel_disp_sigma(halo_mass, z_group, delta_crit=200., cosmology=cosmo):
    """
    halo_mass : array-like
        Halo mass in units of 1e14 h^-1 M_sun
    z_group : array-like
        Redshifts of the halos
    Returns
    -------
    numpy.ndarray
        Velocity dispersion [km/s]
    """
    # convert halo_mass -> Msun
    M = (np.asarray(halo_mass) / H) * u.Msun
    
    # critical density at z, vectorized to numpy array
    rho_c = cosmology.critical_density(np.asarray(z_group))
    rho_c = rho_c.to(u.Msun/u.Mpc**3).value  # strip to plain floats
    rho_c = rho_c * (u.Msun/u.Mpc**3)        # reattach as unit Quantity
    
    # gravitational constant
    G_conv = G.to((u.km/u.s)**2 * u.Mpc / u.Msun)
    
    # prefactor
    pref = ((4*np.pi/3) * delta_crit * rho_c)**(1/3)
    
    # velocity dispersion squared
    sigma2 = G_conv * (M**(2/3)) * pref
    
    # sqrt, then convert to km/s, then return as numpy array
    sigma = np.sqrt(sigma2).to(u.km/u.s)
    return sigma.value

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

        self.delta_v_factor = 5
        self.delta_r_factor = 3
        self.min_log_halo_mass = 11.0
        
        # Mass bin edges (log10 solar masses)
        #self.mass_bins = [
        #    [12.0, 12.5],
        #    [12.5, 13.0],
        #    [13.0, 13.5],
        #    [13.5, 14.0], 
        #    [14.0, 14.5],
        #   [14.5, 16.0]
        #]
        
        #self.mass_bin_labels = [
        #    f"$10^{{{low}}}$ - $10^{{{high}}}$ M$_\\odot$" 
        #    for low, high in self.mass_bins
        #]
    
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

        galaxy_df['log_stellar_mass'] = np.log10((galaxy_df['mstars_disk'] + galaxy_df['mstars_bulge'])/0.67)
        mask = (galaxy_df['log_stellar_mass'] > 8) & (galaxy_df["total_ab_dust_r_VST"] > -200) #& (galaxy_df['zobs'] < 0.2)
        galaxy_df = galaxy_df[mask]
        
        # CRITICAL FIX: Reset index after filtering
        galaxy_df.reset_index(drop=True, inplace=True)
        print(f"After filtering: {len(galaxy_df)} galaxies retained")

        # Get unique halo IDs that have galaxies
        halos_with_galaxies = set(galaxy_df['id_group_sky'].dropna().unique())
        print(f"Found {len(halos_with_galaxies)} unique halos with galaxies")
        
        # Filter halos to only include those with galaxies
        # Assuming halo ID column exists in halo_df - adjust column name as needed
        
        initial_count = len(halo_df)
        # print halo columns

        halo_df = halo_df[halo_df['id_group_sky'].isin(halos_with_galaxies)]
        final_count = len(halo_df)
        
        print(f"Filtered halos: {initial_count} -> {final_count} ({final_count/initial_count*100:.1f}% retained)")
        
        # Add log mass column for easier binning
        halo_df['log_mass'] = np.log10(halo_df['mvir'])
        halo_df = halo_df[halo_df['log_mass'] >= self.min_log_halo_mass]

        halo_df['virial_radius'] = virial_radius(halo_df['mvir'], 
                                                z=halo_df['zcos'], 
                                                delta=200, 
                                                ref_density='mean', 
                                                cosmology=self.cosmo)
        
        halo_df['vel_disp'] = vel_disp_sigma(halo_df['mvir'], halo_df['zcos'], cosmology=self.cosmo)

        #plt.hist(halo_df['vel_disp'], bins=50, log=True)
        #plt.show()

        
        # Reset index and keep track of original position
        halo_df.reset_index(drop=True, inplace=True)
        halo_df['tree_index'] = halo_df.index  # Store position in KD-tree

        # Convert RA/Dec/z to Cartesian coordinates if needed
        halo_df['x'], halo_df['y'], halo_df['z'] = self._ra_dec_z_to_cartesian(
            halo_df['ra'], halo_df['dec'], halo_df['zcos']
        ).T

        galaxy_df['x'], galaxy_df['y'], galaxy_df['z'] = self._ra_dec_z_to_cartesian(
            galaxy_df['ra'], galaxy_df['dec'], galaxy_df['zcos']
        ).T

        self.halo_df = halo_df
        self.galaxy_df = galaxy_df

    def _ra_dec_z_to_cartesian(self, ra, dec, z):
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


    def make_kde_trees(self):
        print('Making KD Trees for fast neighbor searching...')

        self.tree_gals = cKDTree(self.galaxy_df[['x', 'y', 'z']].values)


    def populate_2d_histograms(self):
        print('Populating 2D histograms of normalized projected separation vs normalized velocity difference...')

        proj_bins, vel_bins = 50, 50
        proj_bin_edges = np.linspace(0, self.delta_r_factor, proj_bins + 1)
        vel_bin_edges  = np.linspace(0, self.delta_v_factor, vel_bins + 1)

        # Initialize herons histograms
        self.h_herons_tot        = np.zeros((proj_bins, vel_bins))
        self.h_herons_isolated   = np.zeros((proj_bins, vel_bins))
        self.h_herons_in_group   = np.zeros((proj_bins, vel_bins))
        self.h_herons_diff_group = np.zeros((proj_bins, vel_bins))

        n_groups = len(self.halo_df)
        print(f"Calculating herons halo–galaxy pairs for {n_groups} halos...")

        for i in tqdm(range(n_groups)):
            x1, y1, z1 = self.halo_df.loc[i, ['x', 'y', 'z']]
            vx1, vy1, vz1 = self.halo_df.loc[i, ['vpec_x', 'vpec_y', 'vpec_z']]
            radius1 = self.halo_df.loc[i, 'virial_radius']
            sigma1  = self.halo_df.loc[i, 'vel_disp']
            group1  = self.halo_df.loc[i, 'id_group_sky']

            if sigma1 <= 0 or radius1 <= 0:
                continue

            # Query galaxies within search radius
            indices = self.tree_gals.query_ball_point([x1, y1, z1], r=self.delta_r_factor * radius1)

            for idx in indices:
                gx, gy, gz = self.galaxy_df.loc[idx, ['x', 'y', 'z']]
                gvx, gvy, gvz = self.galaxy_df.loc[idx, ['vpec_x', 'vpec_y', 'vpec_z']]
                group2 = self.galaxy_df.loc[idx, 'id_group_sky']

                # 3D distance
                proj_sep = np.sqrt((gx - x1)**2 + (gy - y1)**2 + (gz - z1)**2)
                # velocity difference
                vel_diff = np.sqrt((gvx - vx1)**2 + (gvy - vy1)**2 + (gvz - vz1)**2)

                proj_norm, vel_norm = proj_sep / radius1, vel_diff / sigma1

                if proj_norm <= self.delta_r_factor and vel_norm <= self.delta_v_factor:
                    proj_bin = np.digitize(proj_norm, proj_bin_edges) - 1
                    vel_bin  = np.digitize(vel_norm, vel_bin_edges) - 1

                    if 0 <= proj_bin < proj_bins and 0 <= vel_bin < vel_bins:

                        self.h_herons_tot[proj_bin, vel_bin] += 1
                        if group2 == -1:
                            self.h_herons_isolated[proj_bin, vel_bin] += 1
                        elif group2 == group1:
                            self.h_herons_in_group[proj_bin, vel_bin] += 1
                        else:
                            self.h_herons_diff_group[proj_bin, vel_bin] += 1

        print("2D histograms populated.")
    
    def plot_histograms(self):
        print('Plotting 2D histograms...')
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        datasets = [
            (self.h_herons_tot,        "All Galaxies"),
            (self.h_herons_isolated,   "Isolated Galaxies"),
            (self.h_herons_in_group,   "In Group Galaxies"),
            (self.h_herons_diff_group, "Different Group Galaxies"),
        ]

        for ax, (hist, label) in zip(axes, datasets):
            # Check if histogram has any non-zero values

            min_positive = np.min(hist[hist > 0]) if np.any(hist > 0) else 1

            hist[0, 0] = min_positive 

            max_val = np.max(hist)

            # Set point at 0,0 to min_positive to avoid issues with LogNorm
            # This is a bit of a hack but helps with visualization
            
            
            if max_val > 0 and min_positive > 0:
                # Use log norm if we have valid positive values
                from matplotlib.colors import LogNorm
                im = ax.imshow(
                    hist.T, origin='lower', aspect='auto',
                    extent=[0, self.delta_r_factor, 0, self.delta_v_factor],
                    norm=LogNorm(vmin=min_positive, vmax=max_val), 
                    cmap='viridis'
                )
            else:
                # Use linear norm if histogram is empty or has problematic values
                print(f"Warning: {label} histogram is empty or has no positive values, using linear scale")
                im = ax.imshow(
                    hist.T, origin='lower', aspect='auto',
                    extent=[0, self.delta_r_factor, 0, self.delta_v_factor],
                    cmap='viridis'
                )
            
            ax.set_title(f'herons: {label}\n({int(np.sum(hist))} pairs)')
            ax.set_xlabel(r'$r / R_{\mathrm{vir}}$')
            ax.set_ylabel(r'$\Delta v / \sigma_v$')
            ax.grid(True, alpha=0.3)
            
            # Only add colorbar if the image was created successfully
            try:
                fig.colorbar(im, ax=ax, label='Number of pairs')
            except ValueError as e:
                print(f"Warning: Could not create colorbar for {label}: {e}")

        plt.suptitle(
            f'Halo–Galaxy Pairs (Herons IDs, log10(mvir) > {self.min_log_halo_mass})',  # Removed reference to undefined variable
            fontsize=16, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig('herons_filters.png', dpi=300)
        plt.show()
    
    def run_analysis(self, halo_file_path, galaxy_file_path, output_path='halo_density_plots.png'):
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
        self.load_data(halo_file_path, galaxy_file_path, )
        self.make_kde_trees()
        
        # Calculate density maps
        self.populate_2d_histograms()
        self.plot_histograms()
        
        

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = HaloDensityAnalyzer()
    
    # Example file path (replace with your actual path)
    galaxy_file_path = '/Users/sp624AA/Downloads/mocks/v0.3.0_vel_inc/waves_wide_gals.parquet'

    halo_file_path = '/Users/sp624AA/Downloads/mocks/v0.3.0_vel_inc/waves_wide_groups.parquet'
    # Run analysis

    analyzer.run_analysis(halo_file_path, galaxy_file_path)
    