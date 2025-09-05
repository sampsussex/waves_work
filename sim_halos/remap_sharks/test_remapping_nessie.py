import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table, join
from astropy.cosmology import FlatLambdaCDM
from nessie import FlatCosmology
from nessie.helper_funcs import create_density_function
from nessie import RedshiftCatalog


from tqdm import tqdm
from numba import njit
import pickle

H = 0.6751
OM_M = 0.3

def remap_to_int32(arr: np.ndarray) -> np.ndarray:
    # Ensure input is integer
    #if arr.dtype != np.int64:
    #    raise ValueError("Input must be int64")
    
    # Mask valid entries (excluding -1)
    mask = arr != -1
    unique_vals, inverse = np.unique(arr[mask], return_inverse=True)
    
    # Map unique values to 1..N
    mapping = np.arange(1, len(unique_vals) + 1, dtype=np.int32)
    
    # Build output array
    out = np.full(arr.shape, -1, dtype=np.int32)
    out[mask] = mapping[inverse]
    
    return out



class SharksNessieScoreRemapper:
    def __init__(self, sharks_path: str, remap_ids_path: str, remap_gal_ids_path: str): 
        print('Initializing SharksCatalogHaloHaloSeparations...')
        self.sharks_path = sharks_path
        self.remap_ids_path = remap_ids_path
        self.remap_gal_ids_path = remap_gal_ids_path
        self.cosmo = FlatCosmology(h = H, omega_matter = OM_M)
        self.astropy_cosmo = FlatLambdaCDM(H0=H*100, Om0=OM_M)
        self.h = H
        self.sharks = None

        self.min_group_members = 5


    def load_data(self):
        print('Reading in sharks data...')
        self.sharks = Table.read(self.sharks_path)

        print('sharks data read in successfully.')
        self.remap_ids = Table.read(self.remap_ids_path)

        self.remap_gal_ids = Table.read(self.remap_gal_ids_path)






    def process_data(self):
        print('Processing sharks data...')
        # make sharks base cuts
        self.sharks['log_stellar_mass'] = np.log10((self.sharks['mstars_disk'] + self.sharks['mstars_bulge'])/0.67)
        mask = (self.sharks['log_stellar_mass'] > 8) & (self.sharks["total_ab_dust_r_VST"] > -200)
        self.sharks = self.sharks[mask]

        # remap sharks ids
        # Rename the column to match your table's column name for joining
        self.remap_ids.rename_column('old_halo_id', 'id_group_sky')

        # Join the tables
        self.sharks = join(self.sharks, self.remap_ids, keys='id_group_sky', join_type='left')

        negative_mask = self.sharks['id_group_sky'] == -1
        self.sharks['new_halo_id'][negative_mask] = -1


        #self.sharks['Velocity_errors'] = np.repeat(0, len(self.sharks)) # in km/s

        self.remap_gal_ids.rename_column('new_group_id', 'new_halo_and_gal_id')

        self.sharks = join(self.sharks, self.remap_gal_ids, keys='id_galaxy_sky', join_type='left')

        self.sharks['id_group_sky'] = remap_to_int32(self.sharks['id_group_sky'])#.astype(np.int32)
        self.sharks['new_halo_id'] = remap_to_int32(self.sharks['new_halo_id'])#.astype(np.int32)
        self.sharks['new_halo_and_gal_id'] = remap_to_int32(self.sharks['new_halo_and_gal_id'])#.astype(np.int32)

        """
        # Reassign int64 to int32
        #negative_mask = self.sharks['id_group_sky'] == -1
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


        negative_mask = self.sharks['new_halo_id'] == -1
        unique_ids = np.unique(self.sharks['new_halo_id'][~negative_mask])
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}

        # Vectorized approach
        old_ids = self.sharks['new_halo_id'].data
        new_ids = np.full(len(old_ids), -1, dtype=np.int32)

        # Map the non-negative values
        mask = ~negative_mask
        mapped_values = np.array([id_mapping.get(old_id, -1) for old_id in old_ids[mask]], dtype=np.int32)
        new_ids[mask] = mapped_values

        self.sharks['new_halo_id'] = new_ids


        negative_mask = self.sharks['new_halo_and_gal_id'] == -1
        unique_ids = np.unique(self.sharks['new_halo_and_gal_id'][~negative_mask])
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
        # Vectorized approach
        old_ids = self.sharks['new_halo_and_gal_id'].data
        new_ids = np.full(len(old_ids), -1, dtype=np.int32)
        # Map the non-negative values
        mask = ~negative_mask
        mapped_values = np.array([id_mapping.get(old_id, -1) for old_id in old_ids[mask]], dtype=np.int32)
        new_ids[mask] = mapped_values
        self.sharks['new_halo_and_gal_id'] = new_ids


        #print('Removing isolated galaxies...')
        # Remove isolated galaxies
        #self.sharks = self.sharks[~negative_mask]

        #plt.scatter(self.sharks['ra'][::1000], self.sharks['dec'][::1000], s=1, alpha=0.5)
        #plt.show()
        """


    def find_nessie_groups(self):
        print('Finding Nessie group properties...')
        sharks_area_sq_deg = 599 + 535 # Waves wide area in square degrees
        sharks_frac_area = sharks_area_sq_deg / (360**2 / np.pi) 

        running_density = create_density_function(self.sharks['zobs'], total_counts = len(self.sharks['zobs']), survey_fractional_area = sharks_frac_area, cosmology = self.cosmo)

        self.red_cat_nessie = RedshiftCatalog(self.sharks['ra'], self.sharks['dec'], self.sharks['zobs'], running_density, self.cosmo)

        completeness = np.repeat(1, len(self.sharks['ra'])) # Assuming completeness is 1 for all galaxies

        self.red_cat_nessie.set_completeness(completeness)
        print(self.sharks['id_group_sky'].dtype, self.sharks['new_halo_id'].dtype, self.sharks['new_halo_and_gal_id'].dtype)

        self.red_cat_nessie.run_fof(b0 = 0.05, r0 = 32) # Params ontimsied by trystan for sharks

        # Print the dtypes of the group ids
        print(self.sharks['id_group_sky'].dtype, self.sharks['new_halo_id'].dtype, self.sharks['new_halo_and_gal_id'].dtype)




        self.group_ids_nessie = self.red_cat_nessie.group_ids

        self.red_cat_nessie.mock_group_ids = self.sharks['id_group_sky']
        self.score_old_ids = self.red_cat_nessie.compare_to_mock(min_group_size=5)
        print("Score against original sharks ids:", self.score_old_ids)

        self.red_cat_nessie.mock_group_ids = self.sharks['new_halo_id']
        self.score_new_ids = self.red_cat_nessie.compare_to_mock(min_group_size=5)

        print("Score against remapped sharks ids:", self.score_new_ids)

        self.red_cat_nessie.mock_group_ids = self.sharks['new_halo_and_gal_id']
        self.score_new_inc_gal_ids = self.red_cat_nessie.compare_to_mock(min_group_size=5)

        print("Score against remapped sharks ids including galaxy id:", self.score_new_inc_gal_ids)



if __name__ == "__main__":
    sharks_path = "/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet"
    remap_ids_path = '/Users/sp624AA/Code/waves_work/sim_halos/remap_sharks/halo_id_mapping.csv'
    remap_gal_ids_path = 'galaxy_group_remap.csv'

    sharks_nessie = SharksNessieScoreRemapper(sharks_path, remap_ids_path, remap_gal_ids_path)
    sharks_nessie.load_data()
    sharks_nessie.process_data()
    sharks_nessie.find_nessie_groups()
