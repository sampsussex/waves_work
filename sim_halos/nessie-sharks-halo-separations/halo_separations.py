import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from nessie import FlatCosmology
from nessie.helper_funcs import create_density_function
from nessie import RedshiftCatalog
from scipy.interpolate import interp1d
from scipy.integrate import quad
from tqdm import tqdm
from numba import njit
from astropy.constants import c
from scipy.spatial import KDTree

C = c.value
H = 0.6751 #100 km/s/Mpc
H0 = H*100 #km/s/Mpc
OM_M = 0.3
OM_L = 0.7


# Columns in group properties
# 'group_id','iter_ra','iter_dec','iter_redshift','iter_idx','median_redshift',
# 'co_dist','r50','r100','rsigma','multiplicity','velocity_dispersion_gap',
# 'velocity_dispersion_gap_err','mass_proxy','bcg_idxs','bcg_ras','bcg_decs',
# 'bcg_redshifts','center_of_light_ras','center_of_light_decs','total_absolute_mag',
# 'flux_proxies','lum_corrected_mass','lum_corrected_flux',



# First set up lf reading
def build_integrated_lf(path: str = "/Users/sp624AA/Code/waves_work/gama_processing/gama_lf/lf.dat", cut: float = -14) -> interp1d:
    """
    Reads the GAMA luminosity function file, filters by magnitude, builds the luminosity-weighted
    luminosity function, integrates it, and returns an interpolation function over that integral.

    Args:
        path: Path to the LF file.
        cut: Magnitude cut (default = -14)

    Returns:
        interp1d: Interpolation function over the integrated phi(L) from -30 to M.
    """
    df = pd.read_csv(path, delim_whitespace=True)
    df = df[df["mag_bin_centre"] < cut].copy()

    mags = df["mag_bin_centre"].values
    phi = df["luminosity_function"].values

    # Compute luminosity-weighted φ: phi * 10^(-0.4 * M)
    phi_lum = phi * 10 ** (-0.4 * mags)

    # Interpolation of φ(L)
    func_lum = interp1d(mags, phi_lum,bounds_error=False,fill_value=(phi_lum[0], phi_lum[-1]))

    # Integrate from -30 to each M
    integrals = np.array([
        quad(func_lum, -30, M, limit=1000)[0] if np.isfinite(M) else 0.0
        for M in mags
    ])

    # Replace zeros with min positive value
    positive = integrals[integrals > 0]
    if len(positive) == 0:
        raise ValueError("All integrated values are zero; check your LF input.")
    min_val = positive.min()
    integrals[integrals == 0] = min_val

    return interp1d(mags, integrals, bounds_error=False, fill_value=(integrals[0], integrals[-1]))


a = [0.2085, 1.0226, 0.5237, 3.5902, 2.3843]
zref = 0
Q0 = 1.75
zp = 0.2
# K corrections

def kcorr(z):
    k=0
    for i in range(len(a)):
        k+= (a[i] * (z - zp)**i)

    return k - (Q0 * (z-zref))


@njit
def calculate_velocity_difference(z1, z2):
    """
    Calculate velocity difference from redshift difference
    Assuming small redshift differences: Δv ≈ c * Δz
    """
    c = 299792.458  # km/s
    return c * abs(z1 - z2)
# Projected separation calculations in numba


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
def find_projected_separation(ra1, dec1, ra2, dec2, co_dist):
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
        co_dist : float
            Comoving distance of the 1st object

    Returns:
        float: Projected seperation in Mpc h ^-1
    """

    return np.radians(angular_sep(ra1, dec1, ra2, dec2))*co_dist


@njit
def spherical_to_cartesian(ra, dec, comoving_distance):
    """
    Convert a single point from spherical coordinates to 3D Cartesian coordinates.
    
    Parameters:
    ----------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    comoving_distance : float
        Comoving distance
        
    Returns:
    -------
    numpy.ndarray
        1D array with x, y, z coordinates
    """
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    
    x = comoving_distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = comoving_distance * np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = comoving_distance * np.sin(dec_rad)
    
    # Return a 1D array with 3 elements for a single point
    result = np.zeros(3)
    result[0] = x
    result[1] = y
    result[2] = z_coord
    return result


@njit
def find_all_spherical_to_cartesian(ra, dec, comoving_distance):
    """
    Convert arrays of spherical coordinates to 3D Cartesian coordinates.
    
    Parameters:
    ----------
    ra : array-like
        Right ascension in degrees
    dec : array-like
        Declination in degrees
    comoving_distance : array-like
        Comoving distances
        
    Returns:
    -------
    numpy.ndarray
        Array of shape (n, 3) with x, y, z coordinates
    """
    n = len(ra)
    all_coords = np.zeros((n, 3))
    
    for i in range(n):
        all_coords[i, 0] = comoving_distance[i] * np.cos(np.deg2rad(dec[i])) * np.cos(np.deg2rad(ra[i]))
        all_coords[i, 1] = comoving_distance[i] * np.cos(np.deg2rad(dec[i])) * np.sin(np.deg2rad(ra[i]))
        all_coords[i, 2] = comoving_distance[i] * np.sin(np.deg2rad(dec[i]))
    
    return all_coords


class SharksCatalogHaloHaloSeparations:
    def __init__(self, sharks_path: str):
        print('Initializing SharksCatalogHaloHaloSeparations...')
        self.sharks_path = sharks_path
        self.cosmo = FlatCosmology(h = H, omega_matter = OM_M)
        self.astropy_cosmo = FlatLambdaCDM(H0=H*100, Om0=OM_M)
        self.h = H
        self.sharks = None
        self.red_cat_nessie = None
        self.red_cat_herons = None

        self.MASS_A = 10

        self.min_group_members = 3
        self.max_proj_r50 = 10
        self.max_vel_sigma = 5


    def load_data(self):
        print('Reading in sharks data...')
        self.sharks = Table.read(self.sharks_path)
        print('sharks data read in successfully.')


    def process_data(self):
        print('Processing sharks data...')
        # make sharks base cuts
        self.sharks['log_stellar_mass'] = np.log10((self.sharks['mstars_disk'] + self.sharks['mstars_bulge'])/0.67)
        mask = (self.sharks['log_stellar_mass'] > 8) & (self.sharks["total_ab_dust_r_VST"] > -200) & (self.sharks['zobs'] < 0.2)
        self.sharks = self.sharks[mask]

        self.sharks['Velocity_errors'] = np.repeat(0, len(self.sharks)) # in km/s

        # Reassign int64 to int32
        negative_mask = self.sharks['id_group_sky'] == -1
        unique_ids = np.unique(self.sharks['id_group_sky'][~negative_mask])
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}

        # Vectorized approach
        old_ids = self.sharks['id_group_sky'].data
        new_ids = np.full(len(old_ids), -1, dtype=np.int32)

        # Map the non-negative values
        mask = ~negative_mask
        mapped_values = np.array([id_mapping.get(old_id, -1) for old_id in old_ids[mask]], dtype=np.int32)
        new_ids[mask] = mapped_values

        self.sharks['id_group_sky'] = new_ids

        #plt.scatter(self.sharks['ra'][::1000], self.sharks['dec'][::1000], s=1, alpha=0.5)
        #plt.show()


    def find_nessie_properties(self):
        print('Finding Nessie group properties...')
        sharks_area_sq_deg = 599 + 535 # Waves wide area in square degrees
        sharks_frac_area = sharks_area_sq_deg / (360**2 / np.pi) 

        running_density = create_density_function(self.sharks['zobs'], total_counts = len(self.sharks['zobs']), survey_fractional_area = sharks_frac_area, cosmology = self.cosmo)

        self.red_cat_nessie = RedshiftCatalog(self.sharks['ra'], self.sharks['dec'], self.sharks['zobs'], running_density, self.cosmo)

        completeness = np.repeat(1, len(self.sharks['ra'])) # Assuming completeness is 1 for all galaxies

        self.red_cat_nessie.set_completeness(completeness)

        self.red_cat_nessie.run_fof(b0 = 0.05, r0 = 32) # Params ontimsied by trystan for sharks

        self.group_ids_nessie = self.red_cat_nessie.group_ids

        self.nessie_group_properties = Table(self.red_cat_nessie.calculate_group_table(np.array(self.sharks['total_ab_dust_r_VST']), np.array(self.sharks['Velocity_errors'])))

        self.nessie_group_properties['MassA'] = self.nessie_group_properties['mass_proxy'] * self.MASS_A / H # * LUM FACTOR

        print(f"Total number of Nessie groups found: {len(self.nessie_group_properties)}")

        self.nessie_group_properties = self.nessie_group_properties[self.nessie_group_properties['multiplicity'] >= self.min_group_members]

        print(f"Number of Nessie groups with at least {self.min_group_members} members: {len(self.nessie_group_properties)}")


    def find_herons_properties(self):
        print('Finding Herons group properties...')
        sharks_area_sq_deg = 599 + 535 # Waves wide area in square degrees
        sharks_frac_area = sharks_area_sq_deg / (360**2 / np.pi) 
        print('Setting up nessie')

        running_density = create_density_function(self.sharks['zobs'], total_counts = len(self.sharks['zobs']), survey_fractional_area = sharks_frac_area, cosmology = self.cosmo)

        self.red_cat_herons = RedshiftCatalog(self.sharks['ra'], self.sharks['dec'], self.sharks['zobs'], running_density, self.cosmo)

        completeness = np.repeat(1, len(self.sharks['ra'])) # Assuming completeness is 1 for all galaxies

        self.red_cat_herons.set_completeness(completeness)

        self.red_cat_herons.group_ids = self.sharks['id_group_sky']#.astype('int32')

        self.group_ids_herons = self.red_cat_herons.group_ids

        print('Calculating Herons group properties...')

        self.herons_group_properties = Table(self.red_cat_herons.calculate_group_table(np.array(self.sharks['total_ab_dust_r_VST']), np.array(self.sharks['Velocity_errors'])))

        # Add mvir_hosthalo to herons_group_properties
        # Find mvir_hosthalo for each group, by matching the group id to one in self.sharks, and taking the mvir_hosthalo from sharks for the first match

        print('Retrieving mvir_hosthalo for Herons groups...')
        sharks_groups = self.sharks['id_group_sky']
        sharks_mvir = self.sharks['mvir_hosthalo']



        # Find first occurrence of each unique group
        unique_groups, first_indices = np.unique(sharks_groups, return_index=True)
        group_mvir_mapping = sharks_mvir[first_indices]

        # Create lookup using searchsorted
        herons_groups = self.herons_group_properties['group_id']
        indices = np.searchsorted(unique_groups, herons_groups)

        # Handle out-of-bounds indices
        valid_mask = (indices < len(unique_groups)) & (unique_groups[indices] == herons_groups)
        mvir_values = np.full(len(herons_groups), -1.0)
        mvir_values[valid_mask] = group_mvir_mapping[indices[valid_mask]]

        self.herons_group_properties['mvir_hosthalo Herons'] = mvir_values

        print('MassA calculation for Herons groups...')

        self.herons_group_properties['MassA'] = self.herons_group_properties['mass_proxy'] * self.MASS_A / H # * LUM FACTOR

        print(f"Total number of Herons groups found: {len(self.herons_group_properties)}")

        self.herons_group_properties = self.herons_group_properties[self.herons_group_properties['multiplicity'] >= self.min_group_members]

        print(f"Number of Herons groups with at least {self.min_group_members} members: {len(self.herons_group_properties)}") 


    def plot_massa_mass_herons(self):
        print('Plotting Herons MassA vs Mvir Host Halo...')
        plt.scatter(np.log10(self.herons_group_properties['mvir_hosthalo Herons']), 
                    np.log10(self.herons_group_properties['MassA']), alpha=0.5, s=1)
        # Add 1-1 line
        plt.plot([10, 16], [10, 16], color='red', linestyle='--', label='1:1 Line')
        plt.xlabel('Log10(Mvir Host Halo) [M☉/h]')
        plt.ylabel('Log10(MassA) [M☉/h]')
        plt.title('Herons MassA vs Mvir Host Halo')
        plt.savefig('herons_massa_vs_mvir.png', dpi=300)
        plt.close()


    def plot_vel_disp_vs_mvir_herons(self):
        print('Plotting Herons Velocity Dispersion vs Mvir Host Halo...')
        plt.scatter(np.log10(self.herons_group_properties['mvir_hosthalo Herons']), 
                    self.herons_group_properties['velocity_dispersion_gap'], alpha=0.5, s=1)
        plt.xlabel('Log10(Mvir Host Halo) [M☉/h]')
        plt.ylabel('Velocity Dispersion Gap [km/s]')
        plt.title('Herons Velocity Dispersion Gap vs Mvir Host Halo')
        plt.yscale('log')
        plt.savefig('herons_vel_disp_vs_mvir.png', dpi=300)
        plt.close()

    def plot_hists_r50_sigma(self):
        bins_r50 = np.linspace(0, 4, 50)
        print('Plotting histograms of r50 and velocity dispersion for Nessie and Herons...')
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(self.nessie_group_properties['r50'], color='blue', alpha=0.7, log=True, label='Nessie', bins = bins_r50)
        plt.hist(self.herons_group_properties['r50'], color='orange', alpha=0.7, log=True, label='Herons', bins = bins_r50)
        plt.xlabel('r50 [Mpc/h]')
        plt.ylabel('Number of Groups')
        plt.title('Distribution of r50')
        plt.legend()

        plt.subplot(1, 2, 2)
        bins_vel = np.linspace(0, 1500, 50)
        plt.hist(self.nessie_group_properties['velocity_dispersion_gap'], color='blue', alpha=0.7, log=True, label='Nessie', bins = bins_vel)
        plt.hist(self.herons_group_properties['velocity_dispersion_gap'], color='orange', alpha=0.7, log=True, label='Herons', bins = bins_vel)
        plt.xlabel('Velocity Dispersion Gap [km/s]')
        plt.ylabel('Number of Groups')
        plt.title('Distribution of Velocity Dispersion Gap')
        plt.legend()

        plt.tight_layout()
        plt.savefig('nessie_herons_r50_vel_disp_histograms.png', dpi=300)
        plt.close()

    def plot_halo_mass_hists(self):
        print('Plotting histograms of MassA for Nessie and Herons...')
        plt.figure(figsize=(8, 6))
        bins_halo_mass = np.linspace(11, 16, 50)
        plt.hist(np.log10(self.nessie_group_properties['MassA']), color='blue', alpha=0.7, log=True, label='Nessie', bins = bins_halo_mass)
        plt.hist(np.log10(self.herons_group_properties['MassA']), color='orange', alpha=0.7, log=True, label='Herons', bins = bins_halo_mass)
        plt.xlabel('MassA [M☉/h]')
        plt.ylabel('Number of Groups')
        plt.title(f'Distribution of MassA, N >= {self.min_group_members}')
        plt.legend()

        plt.tight_layout()
        plt.savefig('nessie_herons_massa_histogram.png', dpi=300)
        plt.close()


    def make_kde_trees(self):
        print('Making KD Trees for fast neighbor searching...')
        
        cartesian_nessie = find_all_spherical_to_cartesian(self.nessie_group_properties['iter_ra'],
                                                            self.nessie_group_properties['iter_dec'],
                                                            self.nessie_group_properties['co_dist'])
        cartesian_herons = find_all_spherical_to_cartesian(self.herons_group_properties['iter_ra'],
                                                            self.herons_group_properties['iter_dec'],
                                                            self.herons_group_properties['co_dist'])
        
        self.tree_nessie = KDTree(cartesian_nessie)
        self.tree_herons = KDTree(cartesian_herons)


    def populate_2d_histograms(self):
        print('Populating 2D histograms of normalized projected separation vs normalized velocity difference...')
        # Define bins
        
        vel_bins = 50
        proj_bins = 50

        proj_bin_edges = np.linspace(0, self.max_proj_r50, proj_bins + 1)
        vel_bin_edges = np.linspace(0, self.max_vel_sigma, vel_bins + 1)

        # Nessie
        self.h_nessie = np.zeros((proj_bins, vel_bins))

        n_groups_nessie = len(self.nessie_group_properties)

        print(f"Calculating nessie pairs for {n_groups_nessie} groups...")

        for i in tqdm(range(n_groups_nessie)):
            
            ra1 = self.nessie_group_properties['iter_ra'][i]
            dec1 = self.nessie_group_properties['iter_dec'][i]
            co_dist1 = self.nessie_group_properties['co_dist'][i]
            sigma1 = self.nessie_group_properties['velocity_dispersion_gap'][i]
            if sigma1 <= 0:
                print(f"Skipping group {i} with non-positive velocity dispersion: {sigma1}")
                continue

            r50_1 = self.nessie_group_properties['r50'][i]
            if r50_1 <= 0:
                print(f"Skipping group {i} with non-positive r50: {r50_1}")
                continue

            z1 = self.nessie_group_properties['median_redshift'][i]

            # Query the tree for neighbors within the max projected separation, getting the 100 nearest neighbors
            distances, indices = self.tree_nessie.query(spherical_to_cartesian(ra1, dec1, co_dist1), k=50)

            for idx in indices:
                if idx == i:
                    continue
                ra2 = self.nessie_group_properties['iter_ra'][idx]
                dec2 = self.nessie_group_properties['iter_dec'][idx]
                z2 = self.nessie_group_properties['median_redshift'][idx]
                # Calculate projected separation
                proj_sep = find_projected_separation(ra1, dec1, ra2, dec2, co_dist1)
                # Calculate velocity difference
                vel_diff = calculate_velocity_difference(z1, z2)
                # Normalize by first halo's properties
                proj_sep_norm = proj_sep / r50_1
                vel_diff_norm = vel_diff / sigma1
                # Apply cuts
                if proj_sep_norm <= self.max_proj_r50 and vel_diff_norm <= self.max_vel_sigma:
                    # Find the bin index for proj_sep_norm and vel_diff_norm
                    proj_bin = np.digitize(proj_sep_norm, proj_bin_edges) - 1
                    vel_bin = np.digitize(vel_diff_norm, vel_bin_edges) - 1
                    # Ensure the indices are within valid range
                    if 0 <= proj_bin < proj_bins and 0 <= vel_bin < vel_bins:
                        self.h_nessie[proj_bin, vel_bin] += 1
        # Herons
        self.h_herons = np.zeros((proj_bins, vel_bins))
        n_groups_herons = len(self.herons_group_properties)
        print(f"Calculating herons pairs for {n_groups_herons} groups...")

        for i in tqdm(range(n_groups_herons)):
            ra1 = self.herons_group_properties['iter_ra'][i]
            dec1 = self.herons_group_properties['iter_dec'][i]
            co_dist1 = self.herons_group_properties['co_dist'][i]
            sigma1 = self.herons_group_properties['velocity_dispersion_gap'][i]
            mass_mvir = self.herons_group_properties['mvir_hosthalo Herons'][i]

            if sigma1 <= 0:
                print(f"Skipping group {i} with non-positive velocity dispersion: {sigma1}")
                print(f'Mass vir: {np.log10(mass_mvir)}')
                continue

            r50_1 = self.herons_group_properties['r50'][i]

            if r50_1 <= 0:
                print(f"Skipping group {i} with non-positive r50: {r50_1}")
                continue

            z1 = self.herons_group_properties['median_redshift'][i]

            # Query the tree for neighbors within the max projected separation, getting the 100 nearest neighbors
            distances, indices = self.tree_herons.query(spherical_to_cartesian(ra1, dec1, co_dist1), k=50)

            for idx in indices:
                if idx == i:
                    continue
                ra2 = self.herons_group_properties['iter_ra'][idx]
                dec2 = self.herons_group_properties['iter_dec'][idx]
                z2 = self.herons_group_properties['median_redshift'][idx]
                # Calculate projected separation
                proj_sep = find_projected_separation(ra1, dec1, ra2, dec2, co_dist1)
                # Calculate velocity difference
                vel_diff = calculate_velocity_difference(z1, z2)
                # Normalize by first halo's properties
                proj_sep_norm = proj_sep / r50_1
                vel_diff_norm = vel_diff / sigma1
                # Apply cuts
                if proj_sep_norm <= self.max_proj_r50 and vel_diff_norm <= self.max_vel_sigma:
                    # Find the bin index for proj_sep_norm and vel_diff_norm
                    proj_bin = np.digitize(proj_sep_norm, proj_bin_edges) - 1
                    vel_bin = np.digitize(vel_diff_norm, vel_bin_edges) - 1
                    # Ensure the indices are within valid range
                    if 0 <= proj_bin < proj_bins and 0 <= vel_bin < vel_bins:
                        self.h_herons[proj_bin, vel_bin] += 1

        print("2D histograms populated.")


    def plot_histograms(self):
        print('Plotting 2D histograms...')
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        proj_bins = self.h_nessie.shape[0]
        vel_bins = self.h_nessie.shape[1]

        proj_bin_edges = np.linspace(0, self.max_proj_r50, proj_bins + 1)
        vel_bin_edges = np.linspace(0, self.max_vel_sigma, vel_bins + 1)

        # Nessie
        im1 = axes[0].imshow(self.h_nessie.T, origin='lower', aspect='auto',
                           extent=[0, self.max_proj_r50, 0, self.max_vel_sigma],
                           cmap='viridis')
        axes[0].set_title(f'Nessie Groups\n({int(np.sum(self.h_nessie))} pairs - {len(self.nessie_group_properties)} groups)')
        plt.colorbar(im1, ax=axes[0], label='Number of pairs')

        # Herons
        im2 = axes[1].imshow(self.h_herons.T, origin='lower', aspect='auto',
                           extent=[0, self.max_proj_r50, 0, self.max_vel_sigma],
                           cmap='viridis')
        axes[1].set_title(f'Herons Groups\n({int(np.sum(self.h_herons))} pairs - {len(self.herons_group_properties)} groups)')
        plt.colorbar(im2, ax=axes[1], label='Number of pairs')

        # Formatting
        for ax in axes:
            ax.set_xlabel('$rp / r50$')
            ax.set_ylabel('$c\Delta z/ \sigma_v$')
            ax.grid(True, alpha=0.3)

        # set super title
        plt.suptitle(f'Sharks WAVESwide Nessie vs Herons Group IDs, N >= {self.min_group_members}', fontsize=16)
        plt.tight_layout()
        plt.savefig('sharks_halo_density_comparison.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    sharks_path = "/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet"
    halo_seps = SharksCatalogHaloHaloSeparations(sharks_path)
    halo_seps.load_data()
    halo_seps.process_data()
    halo_seps.find_nessie_properties()
    halo_seps.find_herons_properties()
    halo_seps.make_kde_trees()
    halo_seps.populate_2d_histograms()
    halo_seps.plot_histograms()
    #halo_seps.plot_massa_mass_herons()
    #halo_seps.plot_vel_disp_vs_mvir_herons()
    #halo_seps.plot_hists_r50_sigma()
    #halo_seps.plot_halo_mass_hists()
