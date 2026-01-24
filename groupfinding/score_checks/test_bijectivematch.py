import numpy as np
from  nessie  import  FlatCosmology, RedshiftCatalog
from  nessie.helper_funcs  import  create_density_function
from astropy.table import Table
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your existing data loading code...
data = Table.read('/Users/sp624AA/Downloads/mocks/GALFORM/G3CMockGalv04all_z_processed.parquet')

print(data.columns)
ra = data['RA']
dec = data['DEC']
redshifts = data['Zspec']
mock_group_ids = data['HaloID']
print("NUMBER OF GALAXIES:", len(ra))
#ra_min, ra_max, dec_min, dec_max = 160, 170, 5, 10

#area_total = (np.radians(ra_max) - np.radians(ra_min)) * \
#    (np.sin(np.radians(dec_max)) - np.sin(np.radians(dec_min)))

area_sq_deg = 144 #area_total * (180 / np.pi)**2
survey_fractional_area = area_sq_deg / (360**2 / np.pi)

cosmo = FlatCosmology(h = 1, omega_matter = 0.25)
running_density = create_density_function(redshifts, total_counts = len(redshifts), 
                                          survey_fractional_area = survey_fractional_area, cosmology = cosmo)

completeness = np.repeat(1.0, len(redshifts))  # Assuming completeness is 100% for all galaxies
# Running group catalogue
red_cat = RedshiftCatalog(ra, dec, redshifts, running_density, cosmo)

plt.scatter(ra, dec, s=1)
plt.savefig('input_galaxies.png')
red_cat.calculate_completeness(ra, dec, completeness)
red_cat.run_fof(b0 = 0.04, r0 = 36)

# Process mock group IDs
mock_group_id_counts = Counter(mock_group_ids)
mock_group_ids = np.array([group_id if mock_group_id_counts[group_id] > 1 else -1 for group_id in mock_group_ids])
mock_group_ids = mock_group_ids.astype(int)

unique_ids = np.unique(mock_group_ids[mock_group_ids != -1])
id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
mock_group_ids = np.array([id_mapping[group_id] if group_id in id_mapping else -1 for group_id in mock_group_ids])

red_cat.mock_group_ids = mock_group_ids
score = red_cat.compare_to_mock(min_group_size=5)


print('Nessie Score', score)

fid_groups = red_cat.mock_group_ids
asg_groups = red_cat.group_ids
gal_ids = np.arange(len(fid_groups))


# Handle -1 group assignments (isolated galaxies)
isolated_mask = fid_groups == -1
n_isolated = np.sum(isolated_mask)

# Assign unique negative IDs starting from -2, -3, -4, etc.
unique_negative_ids = np.arange(-2, -2 - n_isolated, -1)
fid_groups[isolated_mask] = unique_negative_ids

isolated_mask = asg_groups == -1
n_isolated = np.sum(isolated_mask)

# Assign unique negative IDs starting from -2, -3, -4, etc.
unique_negative_ids = np.arange(-2, -2 - n_isolated, -1)
asg_groups[isolated_mask] = unique_negative_ids

asg_df = Table({'id_galaxy_sky': gal_ids, 'asg_id_group_sky': asg_groups}).to_pandas()
fid_df = Table({'id_galaxy_sky': gal_ids, 'fid_id_group_sky': fid_groups}).to_pandas()

import pandas as pd
import numpy as np
from astropy.table import Table
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def count_bijective_matches(df, min_group_size):
    # Count members per group
    fof_sizes = df.groupby('asg_id_group_sky').size().rename('assigned_size')
    fid_sizes = df.groupby('fid_id_group_sky').size().rename('fid_size')
    
    # Count shared members between each pair of (fof, fiducial)
    overlap = df.groupby(['asg_id_group_sky', 'fid_id_group_sky']).size().rename('shared').reset_index()
    
    # Merge in group sizes (only for valid groups)
    overlap = overlap.merge(fof_sizes, on='asg_id_group_sky')
    overlap = overlap.merge(fid_sizes, on='fid_id_group_sky')
    
    # Compute fractions
    overlap['frac_assigned'] = overlap['shared'] / overlap['assigned_size']
    overlap['frac_fid'] = overlap['shared'] / overlap['fid_size']
    
    # Apply >50% threshold to both AND minimum group size filter
    bijective_matches = overlap[
        (overlap['frac_assigned'] > 0.5) & 
        (overlap['frac_fid'] > 0.5) &
        (overlap['assigned_size'] >= min_group_size) &
        (overlap['fid_size'] >= min_group_size)
    ]
    
    # Count the number of groups that meet the minimum size requirement
    num_valid_asg_groups = len(fof_sizes[fof_sizes >= min_group_size])
    num_valid_fid_groups = len(fid_sizes[fid_sizes >= min_group_size])
    
    # Each bijective match is a unique (assigned_group_id, fiducial_group_id) pair
    return len(bijective_matches), bijective_matches, num_valid_asg_groups, num_valid_fid_groups


def calculate_purity_product(bijective_matches):
    """
    Calculate the purity product for each bijectively matched group and compute Q_asg and Q_fid.
    
    The purity product is defined as the product of:
    - frac_assigned: fraction of assigned group members in the fiducial group
    - frac_fid: fraction of fiducial group members in the assigned group
    
    Q_asg is the sum of (asg_group_size * asg_purity) / sum of all asg group sizes
    Q_fid is the sum of (fid_group_size * fid_purity) / sum of all fid group sizes
    
    Parameters:
    -----------
    bijective_matches : pd.DataFrame
        DataFrame containing bijective matches with frac_assigned and frac_fid columns
    
    Returns:
    --------
    tuple: (bijective_matches_df, Q_asg, Q_fid)
        - bijective_matches_df: DataFrame with added purity_product column
        - Q_asg: Weighted purity for assigned groups
        - Q_fid: Weighted purity for fiducial groups
    """
    logger.info("Calculating purity product and Q statistics for bijective matches...")
    
    # Calculate purity product
    bijective_matches = bijective_matches.copy()
    bijective_matches['purity_product'] = bijective_matches['frac_assigned'] * bijective_matches['frac_fid']
    
    # Calculate Q_asg: weighted purity for assigned groups
    # Q_asg = sum(asg_group_size * asg_purity) / sum(all asg group sizes)
    numerator_asg = (bijective_matches['assigned_size'] * bijective_matches['frac_assigned']).sum()
    denominator_asg = bijective_matches['assigned_size'].sum()
    Q_asg = numerator_asg / denominator_asg
    
    # Calculate Q_fid: weighted purity for fiducial groups  
    # Q_fid = sum(fid_group_size * fid_purity) / sum(all fid group sizes)
    numerator_fid = (bijective_matches['fid_size'] * bijective_matches['frac_fid']).sum()
    denominator_fid = bijective_matches['fid_size'].sum()
    Q_fid = numerator_fid / denominator_fid

    Q_tot = Q_asg*Q_fid
    logger.info(f"Weighted purity statistics:")
    logger.info(f"  Q_asg (assigned group weighted purity): {Q_asg:.4f}")
    logger.info(f"  Q_fid (fiducial group weighted purity): {Q_fid:.4f}")
    logger.info(f"  Q_tot (total group weighted purity): {Q_tot:.4f}")
    
    return Q_asg, Q_fid, Q_tot




class BijectiveMatching:
    def __init__(self):
        """
        Initialize bijective matching class with configuration from ConfigReader.
        
        Args:
            config_reader (ConfigReader): An instance of ConfigReader with loaded configuration
        """
        
        # Initialize data containers
        self.asg_groups = asg_groups
        self.fid_groups = fid_groups
        
        # Get bijective matching options
        self.min_group_size = 5

        self.fid_df = fid_df
        self.asg_df = asg_df

        self.n_fid_groups = len(np.unique(self.fid_groups))
        self.n_asg_groups = len(np.unique(self.asg_groups))


    def merge_group_ids(self):
        """
        Merge fiducial and assigned group catalogues.
        """
        logger.info("Merging fiducial and assigned group catalogues...")
        
        if self.fid_df is None:
            logger.error("Fiducial data not loaded. Call load_fiducial_data() first.")
            return
            
        if self.asg_df is None:
            logger.error("Assigned data not loaded. Call load_assigned_data() first.")
            return
        
        try:
            # Check if the number of galaxies match
            if len(self.fid_df) == len(self.asg_df):
                self.groups_df = pd.merge(
                    self.fid_df, 
                    self.asg_df, 
                    how='outer', 
                    on=['id_galaxy_sky'], 
                    validate='1:1'
                )
                logger.info('Fiducial and assigned groups successfully merged')
                logger.info(f"Merged dataset contains {len(self.groups_df)} galaxies")
                
                # Check for any missing matches
                missing_fid = self.groups_df['fid_id_group_sky'].isna().sum()
                missing_asg = self.groups_df['asg_id_group_sky'].isna().sum()
                
                if missing_fid > 0:
                    logger.warning(f"{missing_fid} galaxies missing from fiducial data")
                if missing_asg > 0:
                    logger.warning(f"{missing_asg} galaxies missing from assigned data")
                    
            else:
                logger.error(f"!!! Inconsistent number of galaxies in each input!!!")
                logger.error(f"Fiducial: {len(self.fid_df)}, Assigned: {len(self.asg_df)}")
                
                # Still attempt merge but with warnings
                self.groups_df = pd.merge(
                    self.fid_df, 
                    self.asg_df, 
                    how='outer', 
                    on=['id_galaxy_sky']
                )
                logger.warning(f"Performed outer merge anyway, result has {len(self.groups_df)} galaxies")
                
        except Exception as e:
            logger.error(f"Error merging group catalogues: {e}")
            raise


    def bijective_matching(self):
        logger.info("Finding Bijective matches...")
        self.n_bij_m, self.bijective_matches, self.n_asg_groups, self.n_fid_groups = count_bijective_matches(self.groups_df, self.min_group_size)
        logger.info(f'Bijective matches found, N assigned: {self.n_asg_groups}, N fiducial: {self.n_fid_groups}, N bijective: {self.n_bij_m}')
        self.e_asg = self.n_bij_m/self.n_asg_groups
        self.e_fid =self.n_bij_m/self.n_fid_groups
        self.e_tot = self.e_asg*self.e_fid
        logger.info(f"Efficency statistics; E_asg: {self.e_asg}, E_fid: {self.e_fid}, E_tot: {self.e_tot}")
        
    def calculate_purity_products(self):
        """
        Calculate purity products for all bijectively matched groups and compute Q statistics.
        Must be called after bijective_matching().
        """
        if not hasattr(self, 'bijective_matches'):
            raise ValueError("Must run bijective_matching() before calculating purity products")
        
        self.Q_asg, self.Q_fid, self.Q_tot = calculate_purity_product(self.bijective_matches)

    def get_summary_statistic(self):
        self.S_tot = self.e_tot * self.Q_tot
        logger.info(f"Final summary statistic S_tot = {self.S_tot}")

        
    def run(self):
        """
        Run the full bijective matching process.
        """
        logger.info("Running full bijective matching process...")
        
        self.merge_group_ids()
        self.bijective_matching()
        self.calculate_purity_products()
        self.get_summary_statistic()
        
        logger.info("Bijective matching process completed.")
    


if __name__ == "__main__":

    matching = BijectiveMatching()

    matching.merge_group_ids()
    matching.bijective_matching()
    matching.calculate_purity_products()
    matching.get_summary_statistic()



import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BijResults:
    """Results from bijective comparison between two group catalogues."""
    e_num: int
    e_den: int
    q_num: float
    q_den: float


def bijcheck(group_ids_1: List[int], group_ids_2: List[int], min_group_size: int) -> BijResults:
    """
    Bijective comparison between two group catalogues as per Robotham+2011.
    This calculates Equations (9, 10, 12, and 13) from Robotham+2011.
    
    Args:
        group_ids_1: First group catalog (list of group IDs)
        group_ids_2: Second group catalog (list of group IDs)
        min_group_size: Minimum group size threshold
        
    Returns:
        BijResults containing e_num, e_den, q_num, q_den
    """
    assert len(group_ids_1) == len(group_ids_2), "Group catalogs must have same length"
    
    # Convert to numpy arrays for easier manipulation
    group_ids_1 = np.array(group_ids_1)
    group_ids_2 = np.array(group_ids_2)
    
    # Frequency tables excluding -1
    count_table_1 = Counter(group_ids_1[group_ids_1 != -1])
    count_table_2 = Counter(group_ids_2[group_ids_2 != -1])
    
    # Filter groups in tab1 with size >= min_group_size
    valid_groups_1 = [group for group, count in count_table_1.items() 
                     if count >= min_group_size]
    
    # Find indices of valid group members
    valid_mask = np.isin(group_ids_1, valid_groups_1)
    valid_indices_1 = np.where(valid_mask)[0]
    
    # Create group_list maintaining order (first occurrence of each group)
    group_list = []
    seen = set()
    for idx in valid_indices_1:
        group_id = group_ids_1[idx]
        if group_id not in seen:
            group_list.append(group_id)
            seen.add(group_id)
    
    # Process each group
    q1_values = []
    q2_values = []
    n1_values = []
    
    for group_id in group_list:
        # Find all galaxies in this group
        group_galaxies = np.where(group_ids_1 == group_id)[0]
        
        # Get corresponding groups in catalog 2
        overlap_groups = group_ids_2[group_galaxies]
        overlap_valid = overlap_groups[overlap_groups != -1]
        
        n1_current = count_table_1.get(group_id, 1)
        
        if len(overlap_valid) > 0:
            # Count overlaps
            temptab = Counter(overlap_valid)
            
            frac_1 = []
            frac_2 = []
            
            for group2, count in temptab.items():
                if group2 in count_table_2:
                    n2_val = count_table_2[group2]
                    frac_1.append(count / n1_current)
                    frac_2.append(count / n2_val)
            
            # Handle isolated galaxies (group_id = -1)
            num_isolated = np.sum(overlap_groups == -1)
            if num_isolated > 0:
                iso_frac1 = 1.0 / n1_current
                for _ in range(num_isolated):
                    frac_1.append(iso_frac1)
                    frac_2.append(1.0)
            
            # Find best match (first occurrence of maximum product)
            if frac_1:
                products = [f1 * f2 for f1, f2 in zip(frac_1, frac_2)]
                best_match = np.argmax(products)  # argmax returns first occurrence
                q1 = frac_1[best_match]
                q2 = frac_2[best_match]
            else:
                q1 = 1.0 / n1_current
                q2 = 1.0
        else:
            # All isolated
            q1 = 1.0 / n1_current
            q2 = 1.0
        
        q1_values.append(q1)
        q2_values.append(q2)
        n1_values.append(float(n1_current))
    
    # Calculate final results
    e_num = sum(1 for q1, q2 in zip(q1_values, q2_values) if q1 > 0.5 and q2 > 0.5)
    e_den = len(n1_values)
    q_num = sum(q1 * n1 for q1, n1 in zip(q1_values, n1_values))
    q_den = sum(n1_values)
    
    return BijResults(e_num=e_num, e_den=e_den, q_num=q_num, q_den=q_den)


def s_score(measured_groups: List[int], mock_groups: List[int], groupcut: int) -> float:
    """
    The final S-score measurement for comparisons between two group catalogues.
    Equation 15 of Robotham+2011.
    
    Args:
        measured_groups: Measured group catalog
        mock_groups: Mock/reference group catalog
        groupcut: Minimum group size threshold
        
    Returns:
        S-score value
    """
    mock_vals = bijcheck(mock_groups, measured_groups, groupcut)
    measured_vals = bijcheck(measured_groups, mock_groups, groupcut)
    
    mock_e = mock_vals.e_num / mock_vals.e_den
    fof_e = measured_vals.e_num / measured_vals.e_den
    mock_q = mock_vals.q_num / mock_vals.q_den
    fof_q = measured_vals.q_num / measured_vals.q_den
    
    return mock_e * fof_e * mock_q * fof_q


# Example usage
if __name__ == "__main__":
    # Example group catalogs
    group_ids_1 = asg_groups
    group_ids_2 = fid_groups
    min_group_size = 5
    
    # Perform bijective comparison
    #results = bijcheck(group_ids_1, group_ids_2, min_group_size)
    #print(f"Bijective comparison results:")
    #print(f"e_num: {results.e_num}, e_den: {results.e_den}")
    #print(f"q_num: {results.q_num:.3f}, q_den: {results.q_den:.3f}")
    
    # Calculate S-score
    #print(f"S-score: {score:.6f}")
    score = s_score(group_ids_1, group_ids_2, min_group_size)