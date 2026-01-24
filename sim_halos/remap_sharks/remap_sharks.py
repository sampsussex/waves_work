import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import warnings
from tqdm import tqdm
from numba import njit
warnings.filterwarnings('ignore')

# Cosmology (kept same defaults as you used)
cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3)
H = 0.6751

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



#@njit
def select_candidates_within_rvir(neigh_idxs, neigh_dists, central_idx, central_Rvir, halo_ids_arr, available_mask):
    """
    Numba routine used when assigning halos/galaxies: pick candidate objects that
    are within central_Rvir and still available. Updates available_mask in-place.
    Skips selecting the central itself and any object that belongs to the same
    original halo id as the central.
    Returns boolean mask per neighbor selection.
    """
    n = len(neigh_idxs)
    chosen = np.zeros(n, dtype=np.bool_)
    central_hid = halo_ids_arr[central_idx]
    for i in range(n):
        gi = neigh_idxs[i]
        if not available_mask[gi]:
            continue
        # skip central itself
        if gi == central_idx:
            continue
        # skip if same halo id already
        if halo_ids_arr[gi] == central_hid:
            continue
        if neigh_dists[i] <= central_Rvir:
            chosen[i] = True
            available_mask[gi] = False
    return chosen

@njit
def select_galaxies_within_rvir(cand_idxs, dist_arr, R, gal_available):
    chosen = []
    for idx, d in zip(cand_idxs, dist_arr):
        if gal_available[idx] and d <= R:
            chosen.append(idx)
    return np.array(chosen, dtype=np.int64)



def build_halo_mapping_numba(df, mass_threshold=11, n_neighbors=500):
    """
    Halo -> new host mapping updated to reassign a halo to a more massive
    halo if it lies within the *virial radius* of that more massive halo.
    Returns mapping {old_halo_id: new_halo_id}.
    """
    halos = df.copy().reset_index(drop=True)
    positions = halos[['x', 'y', 'z']].to_numpy()
    velocities = halos[['vpec_x', 'vpec_y', 'vpec_z']].to_numpy()
    halo_ids = halos['id_group_sky'].to_numpy()
    masses = halos['log_mass'].to_numpy()
    virial_radii = halos['virial_radius'].to_numpy()

    tree = cKDTree(positions)

    # central candidates: mass > threshold, sorted descending
    mask_massive = masses > mass_threshold
    massive_idx = np.where(mask_massive)[0]
    if massive_idx.size == 0:
        return {hid: hid for hid in halo_ids}

    order = np.argsort(masses[massive_idx])[::-1]
    massive_idx = massive_idx[order]

    mapping = {hid: hid for hid in halo_ids}
    available = np.ones(len(halos), dtype=np.bool_)

    for central_idx in tqdm(massive_idx, desc="Remapping halos"):
        if not available[central_idx]:
            continue
        central_pos = positions[central_idx]
        # get nearest neighbors (a safety cap n_neighbors)
        k = min(n_neighbors, len(halos))
        dists, idxs = tree.query(central_pos, k=k)
        # ensure arrays
        if np.isscalar(idxs):
            idxs = np.array([idxs], dtype=np.int64)
            dists = np.array([dists], dtype=float)

        central_Rvir = virial_radii[central_idx]

        # Prevent the central from being selected as a neighbor by marking
        # it unavailable before selection. This preserves the same "claiming"
        # logic used previously (an object can be claimed once).
        available[central_idx] = False

        chosen_mask = select_candidates_within_rvir(
            idxs.astype(np.int64), dists.astype(float), int(central_idx), central_Rvir,
            halo_ids.astype(np.int64), available
        )

        # update mapping: assign any chosen neighbor halo to central's halo id
        central_hid = int(halo_ids[int(central_idx)])
        for j in range(len(idxs)):
            if chosen_mask[j]:
                nb_idx = int(idxs[j])
                mapping[int(halo_ids[nb_idx])] = central_hid

    return mapping


def remap_galaxies(halo_df, galaxy_df, mass_threshold=11):
    """
    Reassign galaxies to the most massive halo (largest -> smallest ordering)
    if galaxy lies within halo's virial radius.
    Only reassign *isolated* galaxies (id_group_sky = -1).
    
    Returns:
      new_galaxy_df (copy with updated 'id_group_sky'),
      galaxy_to_halo (dict: original_galaxy_index -> new_halo_id)
    """
    halos = halo_df.copy().reset_index(drop=True)
    gals = galaxy_df.copy().reset_index(drop=True)

    # halo properties
    halo_pos = halos[['x', 'y', 'z']].to_numpy()
    halo_Rvir = halos['virial_radius'].to_numpy()
    halo_ids = halos['id_group_sky'].to_numpy()
    halo_masses = halos['log_mass'].to_numpy()

    # galaxy properties
    gal_pos = gals[['x', 'y', 'z']].to_numpy()
    gal_ids = gals['id_group_sky'].to_numpy()

    # only isolated galaxies are allowed to be reassigned
    isolated_mask = (gal_ids == -1)

    # KDTree of galaxies for fast queries
    gal_tree = cKDTree(gal_pos)

    # massive halos only, sorted descending by mass
    mask_massive = halo_masses > mass_threshold
    massive_idx = np.where(mask_massive)[0]
    if massive_idx.size == 0:
        return gals, {i: gal_ids[i] for i in range(len(gals))}

    order = np.argsort(halo_masses[massive_idx])[::-1]
    massive_idx = massive_idx[order]

    # track which isolated galaxies are still unclaimed
    gal_available = isolated_mask.copy()
    galaxy_to_halo = {i: int(gal_ids[i]) for i in range(len(gals))}  # default = keep original

    # loop over halos
    for central_idx in tqdm(massive_idx, desc="Assigning galaxies to halos"):
        central_pos = halo_pos[central_idx]
        R = halo_Rvir[central_idx]
        central_hid = int(halo_ids[central_idx])

        # find candidate galaxies within virial radius
        cand_list = gal_tree.query_ball_point(central_pos, r=R)
        if len(cand_list) == 0:
            continue
        cand_idxs = np.array(cand_list, dtype=np.int64)

        # compute distances
        diffs = gal_pos[cand_idxs] - central_pos[np.newaxis, :]
        dists = np.sqrt((diffs**2).sum(axis=1))

        # filter to those within Rvir and still available
        chosen_idxs = []
        for gi, dist in zip(cand_idxs, dists):
            if gal_available[gi] and dist <= R:
                chosen_idxs.append(gi)

        # assign galaxies to halo
        for gi in chosen_idxs:
            galaxy_to_halo[gi] = central_hid
            gal_available[gi] = False  # mark as claimed

    # update dataframe with final mapping
    new_gals = gals.copy()
    new_ids = np.array([galaxy_to_halo[i] for i in range(len(new_gals))],
                       dtype=new_gals['id_group_sky'].dtype)
    new_gals['id_group_sky'] = new_ids

    return new_gals, galaxy_to_halo



class HaloRemapper:
    def __init__(self, H0=67.51, Om0=0.3):
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        self.mass_lim_min = 11
        self.save_path_mapping = "halo_id_mapping.csv"

    def load_data(self, halo_file_path, galaxy_file_path):
        """
        Load halo & galaxy files, filter galaxies reasonably, compute log mass &
        virial radius, and compute Cartesian coords. Returns halo_df and galaxy_df.
        """
        print(f"Loading halo file: {halo_file_path}")
        halo_df = pd.read_parquet(halo_file_path)
        print(f"Loaded {len(halo_df)} halos")

        print(f"Loading galaxy file: {galaxy_file_path}")
        galaxy_df = pd.read_parquet(galaxy_file_path)
        print(f"Loaded {len(galaxy_df)} galaxies")

        # simple galaxy filter (kept your logic)
        galaxy_df['log_stellar_mass'] = np.log10((galaxy_df['mstars_disk'] + galaxy_df['mstars_bulge'])/0.67)
        mask = (galaxy_df['log_stellar_mass'] > 8) & (galaxy_df["total_ab_dust_r_VST"] > -200)
        galaxy_df = galaxy_df[mask].reset_index(drop=True)
        print(f"Filtered galaxies -> {len(galaxy_df)}")

        # which halos have galaxies
        halos_with_gals = set(galaxy_df['id_group_sky'].dropna().unique())
        # find the halo id column in halo_df
        if 'id_group_sky' in halo_df.columns:
            hid_col = 'id_group_sky'
        elif 'halo_id' in halo_df.columns:
            hid_col = 'halo_id'
        elif 'id' in halo_df.columns:
            hid_col = 'id'
        else:
            raise ValueError("Could not find halo ID column in halo file.")

        # filter halos to those that have galaxies (reduces work)
        halo_df = halo_df[halo_df[hid_col].isin(halos_with_gals)].reset_index(drop=True)
        print(f"Filtered halos -> {len(halo_df)}")

        # add log mass & virial radius (vectorised)
        halo_df['log_mass'] = np.log10(halo_df['mvir'])
        halo_df['virial_radius'] = virial_radius(halo_df['mvir'], z=halo_df['zcos'], delta=200, ref_density='mean', cosmology=self.cosmo)

        # cartesian coordinates
        halo_df[['x', 'y', 'z']] = self._ra_dec_z_to_cartesian(halo_df['ra'], halo_df['dec'], halo_df['zcos'])
        galaxy_df[['x', 'y', 'z']] = self._ra_dec_z_to_cartesian(galaxy_df['ra'], galaxy_df['dec'], galaxy_df['zcos'])

        return halo_df, galaxy_df

    def _ra_dec_z_to_cartesian(self, ra, dec, z):
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        dist = self.cosmo.comoving_distance(z).value  # Mpc
        x = dist * np.cos(dec_rad) * np.cos(ra_rad)
        y = dist * np.cos(dec_rad) * np.sin(ra_rad)
        zc = dist * np.sin(dec_rad)
        return np.column_stack([x, y, zc])

    def remap_halos(self, halo_df):
        print("Building halo ID mapping...")
        mapping = build_halo_mapping_numba(halo_df, mass_threshold=self.mass_lim_min, n_neighbors=500)
        self.mapping = mapping
        pd.DataFrame(list(mapping.items()), columns=['old_halo_id', 'new_halo_id']).to_csv(self.save_path_mapping, index=False)
        print("Halo mapping saved to", self.save_path_mapping)
        return mapping

    def remap_galaxy_membership(self, halo_df, galaxy_df):
        print("Remapping galaxy membership (assign *isolated* galaxies to enclosing halos)...")
        new_gals, mapping = remap_galaxies(halo_df, galaxy_df, mass_threshold=self.mass_lim_min)
        print("Galaxy remapping complete.")
        return new_gals, mapping


if __name__ == "__main__":
    remapper = HaloRemapper(H0=67.51, Om0=0.3)

    halo_file = "/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_groups.parquet"
    galaxy_file = "/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet"

    halo_df, galaxy_df = remapper.load_data(halo_file, galaxy_file)

    # Step 1. Remap halos (reassign halos that fall within the virial radius of a more massive halo)
    halo_mapping = remapper.remap_halos(halo_df)

    # Load mapping from CSV
    #df_mapping = pd.read_csv("halo_id_mapping.csv")
    # Convert to dictionary
    #halo_mapping = dict(zip(df_mapping["old_halo_id"], df_mapping["new_halo_id"]))

    # Step 2. Apply halo mapping to galaxy catalog (preserve original ids where mapping not found)
    galaxy_df = galaxy_df.copy()
    galaxy_df["id_group_sky"] = galaxy_df["id_group_sky"].map(halo_mapping).fillna(galaxy_df["id_group_sky"]) 

    # Step 3. Remap only *isolated* galaxies (id_group_sky == -1) by virial radius membership
    new_galaxy_df, galaxy_to_halo_map = remapper.remap_galaxy_membership(halo_df, galaxy_df)

    # Step 4. Save down galaxy id + final remapped group id
    out_df = new_galaxy_df[["id_galaxy_sky", "id_group_sky"]].rename(columns={"id_group_sky": "new_group_id"})
    out_file = "galaxy_group_remap.csv"
    out_df.to_csv(out_file, index=False)

    print(f"Galaxy → halo remapping saved to {out_file}")
    print(out_df.head())
