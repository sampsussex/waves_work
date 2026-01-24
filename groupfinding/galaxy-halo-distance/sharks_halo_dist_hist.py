import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
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

def radec_to_cartesian(ra, dec, redshift, cosmo):
    """Convert RA/Dec/redshift to 3D Cartesian coordinates."""
    # Comoving distance in Mpc
    d_c = cosmo.comoving_distance(redshift).value
    
    # Convert to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    # Cartesian coordinates
    x = d_c * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_c * np.cos(dec_rad) * np.sin(ra_rad)
    z = d_c * np.sin(dec_rad)
    
    return x, y, z

def create_distance_histograms(galaxy_file, halo_file, output_prefix='distance_histogram_iso'):
    """
    Create distance histograms for galaxies around halos in different mass bins.
    
    Parameters:
    -----------
    galaxy_file : str
        Path to parquet file containing galaxy data
    halo_file : str
        Path to parquet file containing halo data
    output_prefix : str
        Prefix for output plot files
    """
    
    # Load data
    print("Loading galaxy and halo catalogs...")
    galaxies = pd.read_parquet(galaxy_file)
    galaxies = galaxies[galaxies['']]
    halos = pd.read_parquet(halo_file)
    
    # Convert coordinates to Cartesian
    print("Converting coordinates to 3D Cartesian...")
    gal_x, gal_y, gal_z = radec_to_cartesian(galaxies['ra'], galaxies['dec'], 
                                              galaxies['zcos'], cosmo)
    galaxies['x'] = gal_x
    galaxies['y'] = gal_y
    galaxies['z'] = gal_z
    
    halo_x, halo_y, halo_z = radec_to_cartesian(halos['ra'], halos['dec'], 
                                                 halos['zcos'], cosmo)
    halos['x'] = halo_x
    halos['y'] = halo_y
    halos['z'] = halo_z

    # Convert halo mass to virial radius
    print("Calculating virial radii for halos...")
    halos['virial_radius'] = virial_radius(halos['mvir'], z=halos['zcos'])
    
    # Build KD-tree for galaxies
    print("Building KD-tree for efficient galaxy queries...")
    galaxy_positions = np.column_stack((gal_x, gal_y, gal_z))
    galaxy_tree = cKDTree(galaxy_positions)
    
    # Define mass bins
    mass_bins = [10**10.5, 10**11, 10**11.5, 10**12, 10**12.5, 
                 10**13, 10**13.5, 10**14, 10**14.5]
    
    # Create figure with subplots (7 mass bins)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    # Remove the extra subplot
    fig.delaxes(axes[7])
    
    fig.suptitle('Galaxy Distance Distributions Around Halos', fontsize=16)
    
    # Define histogram bins for distances (in units of virial radius)
    distance_bins = np.linspace(0, 3, 31)  # 0 to 3 R_vir in 30 bins
    
    # Process each mass bin
    for i in range(len(mass_bins) - 1):
        min_mass = mass_bins[i]
        max_mass = mass_bins[i + 1]
        
        print(f"\nProcessing mass bin: {min_mass:.2e} - {max_mass:.2e} M_sun")
        
        # Select halos in this mass bin
        mask = (halos['mvir'] >= min_mass) & (halos['mvir'] < max_mass)
        bin_halos = halos[mask]
        
        if len(bin_halos) == 0:
            print(f"No halos found in mass bin {i+1}")
            axes[i].text(0.5, 0.5, f'No halos in mass range\n{min_mass:.1e}-{max_mass:.1e} M_sun', 
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_xlabel('Distance / R_virial')
            axes[i].set_ylabel('Number of galaxies')
            continue
        
        # Arrays to collect all distances
        same_halo_distances = []
        diff_halo_distances = []
        iso_gal_distance = []
        
        # Process each halo in this mass bin
        for _, halo in tqdm(bin_halos.iterrows()):
            halo_pos = np.array([halo['x'], halo['y'], halo['z']])
            search_radius = 3 * halo['virial_radius']
            
            # Find galaxies within 3 * virial radius
            indices = galaxy_tree.query_ball_point(halo_pos, search_radius)
            
            if len(indices) == 0:
                continue
            
            nearby_galaxies = galaxies.iloc[indices]
            
            # Calculate distances
            gal_positions = nearby_galaxies[['x', 'y', 'z']].values
            distances = np.sqrt(np.sum((gal_positions - halo_pos)**2, axis=1))
            
            # Scale by virial radius
            scaled_distances = distances / halo['virial_radius']
            
            # Separate galaxies by halo ID
            same_halo_mask = nearby_galaxies['id_group_sky'] == halo['id_group_sky']

            isolated_galaxy_mask = nearby_galaxies['id_group_sky'] == -1
            
            same_halo_distances.extend(scaled_distances[same_halo_mask])

            iso_gal_distance.extend(scaled_distances[isolated_galaxy_mask])

            diff_halo_distances.extend(scaled_distances[(~same_halo_mask) & (~isolated_galaxy_mask)])
            #diff_halo_distances.extend(scaled_distances[~same_halo_mask])- old version with no iso galaxies
        
        # Convert to arrays
        same_halo_distances = np.array(same_halo_distances)
        diff_halo_distances = np.array(diff_halo_distances)
        iso_gal_distance = np.array(iso_gal_distance)
        
        # Create stacked histogram
        ax = axes[i]
        
        # Plot histogram for same halo ID (bottom layer)
        n_same, bins, patches1 = ax.hist(same_halo_distances, bins=distance_bins, 
                                         alpha=0.7, label=f'Same halo ID (N={len(same_halo_distances)})',
                                         color='blue', edgecolor='black', linewidth=0.5)
        
        # Plot histogram for different halo ID (stacked on top)
        n_diff, bins, patches2 = ax.hist(diff_halo_distances, bins=distance_bins, 
                                         bottom=n_same, alpha=0.7, 
                                         label=f'Different halo ID (N={len(diff_halo_distances)})',
                                         color='red', edgecolor='black', linewidth=0.5)
        
        # Plot histogram for isolated galaxies (stacked on top)
        n_iso, bins, patches3 = ax.hist(iso_gal_distance, bins=distance_bins, 
                                         bottom=n_same + n_diff, alpha=0.7, 
                                         label=f'Isolated galaxies (N={len(iso_gal_distance)})',
                                         color='green', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Distance / R_virial')
        ax.set_ylabel('Number of galaxies')
        ax.set_title(f'Mass: {min_mass:.1e}-{max_mass:.1e} M_sun\n{len(bin_halos)} halos')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limit
        ax.set_xlim(0, 3)
        
        print(f"  Total halos in bin: {len(bin_halos)}")
        print(f"  Same halo ID: {len(same_halo_distances)} galaxies")
        print(f"  Different halo ID: {len(diff_halo_distances)} galaxies")
    
    plt.tight_layout()
    output_file = f"{output_prefix}_all_mass_bins_iso.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot to {output_file}")
    
    # Also create individual plots for each mass bin with more detail
    for i in range(len(mass_bins) - 1):
        min_mass = mass_bins[i]
        max_mass = mass_bins[i + 1]
        
        # Select halos in this mass bin
        mask = (halos['mvir'] >= min_mass) & (halos['mvir'] < max_mass)
        bin_halos = halos[mask]
        
        if len(bin_halos) == 0:
            continue
        
        # Collect distances again for individual plot
        same_halo_distances = []
        diff_halo_distances = []
        iso_gal_distances = []
        
        for _, halo in bin_halos.iterrows():
            halo_pos = np.array([halo['x'], halo['y'], halo['z']])
            search_radius = 3 * halo['virial_radius']
            indices = galaxy_tree.query_ball_point(halo_pos, search_radius)
            
            if len(indices) == 0:
                continue
            
            nearby_galaxies = galaxies.iloc[indices]
            gal_positions = nearby_galaxies[['x', 'y', 'z']].values
            distances = np.sqrt(np.sum((gal_positions - halo_pos)**2, axis=1))
            scaled_distances = distances / halo['virial_radius']
            
            same_halo_mask = nearby_galaxies['id_group_sky'] == halo['id_group_sky']
            isolated_galaxy_mask = nearby_galaxies['id_group_sky'] == -1
            diff_halo_mask = ~same_halo_mask & ~isolated_galaxy_mask

            same_halo_distances.extend(scaled_distances[same_halo_mask])
            iso_gal_distances.extend(scaled_distances[isolated_galaxy_mask])
            diff_halo_distances.extend(scaled_distances[diff_halo_mask])
        
        same_halo_distances = np.array(same_halo_distances)
        diff_halo_distances = np.array(diff_halo_distances)
        iso_gal_distances = np.array(iso_gal_distances)
        
        # Create individual figure with two subplots
        fig_individual, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left subplot: Stacked histogram
        n_same, bins, _ = ax1.hist(same_halo_distances, bins=distance_bins, 
                                   alpha=0.7, label=f'Same halo ID (N={len(same_halo_distances)})',
                                   color='blue', edgecolor='black', linewidth=0.5)
        
        n_diff, bins, _ = ax1.hist(diff_halo_distances, bins=distance_bins, 
                                   bottom=n_same, alpha=0.7, 
                                   label=f'Different halo ID (N={len(diff_halo_distances)})',
                                   color='red', edgecolor='black', linewidth=0.5)
        
        n_iso, bins, _ = ax1.hist(iso_gal_distances, bins=distance_bins, 
                                   bottom=n_same + n_diff, alpha=0.7, 
                                   label=f'Isolated galaxies (N={len(iso_gal_distances)})',
                                   color='green', edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Distance / R_virial')
        ax1.set_ylabel('Number of galaxies')
        ax1.set_title(f'Stacked Distribution\nMass: {min_mass:.1e}-{max_mass:.1e} M_sun ({len(bin_halos)} halos)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 3)
        
        # Right subplot: Normalized distributions for comparison
        if len(same_halo_distances) > 0:
            ax2.hist(same_halo_distances, bins=distance_bins, density=True,
                     alpha=0.7, label='Same halo ID', color='blue', 
                     edgecolor='black', linewidth=0.5)
        
        if len(diff_halo_distances) > 0:
            ax2.hist(diff_halo_distances, bins=distance_bins, density=True,
                     alpha=0.7, label='Different halo ID', color='red',
                     edgecolor='black', linewidth=0.5)
            
        if len(iso_gal_distances) > 0:
            ax2.hist(iso_gal_distances, bins=distance_bins, density=True,
                     alpha=0.7, label='Isolated galaxies', color='green',
                     edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Distance / R_virial')
        ax2.set_ylabel('Normalized density')
        ax2.set_title('Normalized Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 3)
        
        plt.tight_layout()
        individual_file = f"{output_prefix}_mass_bin_{i+1}_detailed_iso.png"
        plt.savefig(individual_file, dpi=300, bbox_inches='tight')
        plt.close(fig_individual)
        print(f"Saved detailed plot to {individual_file}")
    
    plt.close('all')
    
    # Create a summary statistics file
    summary_file = f"{output_prefix}_summary_iso.txt"
    with open(summary_file, 'w') as f:
        f.write("Galaxy-Halo Distance Distribution Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for i in range(len(mass_bins) - 1):
            min_mass = mass_bins[i]
            max_mass = mass_bins[i + 1]
            
            mask = (halos['mvir'] >= min_mass) & (halos['mvir'] < max_mass)
            bin_halos = halos[mask]
            
            f.write(f"Mass bin {i+1}: {min_mass:.2e} - {max_mass:.2e} M_sun\n")
            f.write(f"Number of halos: {len(bin_halos)}\n")
            
            if len(bin_halos) > 0:
                f.write(f"Average halo mass: {bin_halos['mvir'].mean():.2e} M_sun\n")
                f.write(f"Average virial radius: {bin_halos['virial_radius'].mean():.2f} Mpc\n")
            
            f.write("\n")
    
    print(f"\nSaved summary statistics to {summary_file}")
    
    return fig

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    galaxy_file = '/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet'
    halo_file = '/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_groups.parquet'
    
    # Create the distance histograms
    create_distance_histograms(galaxy_file, halo_file, output_prefix="galaxy_halo_distance")