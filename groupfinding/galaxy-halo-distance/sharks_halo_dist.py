import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from astropy.cosmology import FlatLambdaCDM
import warnings
import astropy.units as u
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
    M = M_vir * u.Msun

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

def calculate_velocity_offset(gal_v, halo_v):
    """Calculate velocity offset from halo."""
    return np.sqrt((gal_v[:, 0] - halo_v[0])**2 + 
                   (gal_v[:, 1] - halo_v[1])**2 + 
                   (gal_v[:, 2] - halo_v[2])**2)

def create_density_plots(galaxy_file, halo_file, output_prefix='density_plot'):
    """
    Create density plots for galaxies around halos in different mass bins.
    
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
    halos = pd.read_parquet(halo_file)
    
    #print column names
    print("Galaxy columns:", galaxies.columns.tolist())
    print("Halo columns:", halos.columns.tolist())


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
    
    # Build KD-tree for galaxies
    print("Building KD-tree for efficient galaxy queries...")
    galaxy_positions = np.column_stack((gal_x, gal_y, gal_z))
    galaxy_tree = cKDTree(galaxy_positions)
    
    # Define mass bins
    mass_bins = [10**10.5, 10**11, 10**11.5, 10**12, 10**12.5, 
                 10**13, 10**13.5, 10**14, 10**14.5]
    
    # Create figure with subplots
    fig, axes = plt.subplots(7, 2, figsize=(12, 28))
    fig.suptitle('Galaxy Density Plots Around Halos', fontsize=16)
    
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
            continue
        
        # Arrays to store data for plotting
        same_halo_displacements = []
        same_halo_velocities = []
        diff_halo_displacements = []
        diff_halo_velocities = []
        
        # Process each halo in this mass bin
        for _, halo in bin_halos.iterrows():
            halo_pos = np.array([halo['x'], halo['y'], halo['z']])
            halo_vel = np.array([halo['vpec_x'], halo['vpec_y'], halo['vpec_z']])
            halo['virial_radius'] = virial_radius(halo['mvir'], z=halo['zcos'])
            search_radius = 3 * halo['virial_radius']
            
            # Find galaxies within 3 * virial radius
            indices = galaxy_tree.query_ball_point(halo_pos, search_radius)
            
            if len(indices) == 0:
                continue
            
            nearby_galaxies = galaxies.iloc[indices]
            
            # Calculate displacements
            gal_positions = nearby_galaxies[['x', 'y', 'z']].values
            displacements = np.sqrt(np.sum((gal_positions - halo_pos)**2, axis=1))
            
            # Scale by virial radius
            scaled_displacements = displacements / halo['virial_radius']
            
            # Calculate velocity offsets
            gal_velocities = nearby_galaxies[['vpec_x', 'vpec_y', 'vpec_z']].values
            vel_offsets = calculate_velocity_offset(gal_velocities, halo_vel)
            
            # Separate galaxies by halo ID
            same_halo_mask = nearby_galaxies['id_group_sky'] == halo['id_group_sky']
            
            same_halo_displacements.extend(scaled_displacements[same_halo_mask])
            same_halo_velocities.extend(vel_offsets[same_halo_mask])
            
            diff_halo_displacements.extend(scaled_displacements[~same_halo_mask])
            diff_halo_velocities.extend(vel_offsets[~same_halo_mask])
        
        # Convert to arrays
        same_halo_displacements = np.array(same_halo_displacements)
        same_halo_velocities = np.array(same_halo_velocities)
        diff_halo_displacements = np.array(diff_halo_displacements)
        diff_halo_velocities = np.array(diff_halo_velocities)
        
        # Create density plots for same halo ID
        ax1 = axes[i, 0]
        if len(same_halo_displacements) > 10:
            try:
                # Create 2D histogram
                h1, xedges, yedges = np.histogram2d(same_halo_displacements, 
                                                     same_halo_velocities, 
                                                     bins=50)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im1 = ax1.imshow(h1.T, origin='lower', extent=extent, 
                                 aspect='auto', cmap='viridis')
                plt.colorbar(im1, ax=ax1, label='Count')
            except:
                ax1.text(0.5, 0.5, 'Insufficient data', 
                        transform=ax1.transAxes, ha='center')
        else:
            ax1.text(0.5, 0.5, 'No galaxies with same halo ID', 
                    transform=ax1.transAxes, ha='center')
        
        ax1.set_xlabel('Displacement / R_virial')
        ax1.set_ylabel('Velocity offset (km/s)')
        ax1.set_title(f'Same Halo ID\nMass: {min_mass:.1e}-{max_mass:.1e} M_sun')
        
        # Create density plots for different halo ID
        ax2 = axes[i, 1]
        if len(diff_halo_displacements) > 10:
            try:
                # Create 2D histogram
                h2, xedges, yedges = np.histogram2d(diff_halo_displacements, 
                                                     diff_halo_velocities, 
                                                     bins=50)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im2 = ax2.imshow(h2.T, origin='lower', extent=extent, 
                                 aspect='auto', cmap='viridis')
                plt.colorbar(im2, ax=ax2, label='Count')
            except:
                ax2.text(0.5, 0.5, 'Insufficient data', 
                        transform=ax2.transAxes, ha='center')
        else:
            ax2.text(0.5, 0.5, 'No galaxies with different halo ID', 
                    transform=ax2.transAxes, ha='center')
        
        ax2.set_xlabel('Displacement / R_virial')
        ax2.set_ylabel('Velocity offset (km/s)')
        ax2.set_title(f'Different Halo ID\nMass: {min_mass:.1e}-{max_mass:.1e} M_sun')
        
        print(f"  Same halo ID: {len(same_halo_displacements)} galaxies")
        print(f"  Different halo ID: {len(diff_halo_displacements)} galaxies")
    
    plt.tight_layout()
    output_file = f"{output_prefix}_all_mass_bins.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot to {output_file}")
    
    # Also save individual plots for each mass bin
    for i in range(len(mass_bins) - 1):
        fig_individual = plt.figure(figsize=(12, 5))
        
        # Copy the subplots from the main figure
        for j in range(2):
            ax_src = axes[i, j]
            ax_dst = fig_individual.add_subplot(1, 2, j+1)
            
            # Copy the image data if it exists
            for im in ax_src.get_images():
                new_im = ax_dst.imshow(im.get_array(), 
                                      extent=im.get_extent(),
                                      origin='lower', 
                                      aspect='auto', 
                                      cmap=im.get_cmap())
                plt.colorbar(new_im, ax=ax_dst, label='Count')
            
            # Copy labels and title
            ax_dst.set_xlabel(ax_src.get_xlabel())
            ax_dst.set_ylabel(ax_src.get_ylabel())
            ax_dst.set_title(ax_src.get_title())
        
        plt.tight_layout()
        individual_file = f"{output_prefix}_mass_bin_{i+1}.png"
        plt.savefig(individual_file, dpi=300, bbox_inches='tight')
        plt.close(fig_individual)
        print(f"Saved individual plot to {individual_file}")
    
    plt.close('all')
    return fig

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    galaxy_file = '/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet'
    halo_file = '/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_groups.parquet'
    
    # Create the density plots
    create_density_plots(galaxy_file, halo_file, output_prefix="galaxy_halo_density")