# A bit of code for remapping sharks halos. In Psudeo code this works as follows; 
Overall Pipeline

1. Load halo_df and galaxy_df from files.
2. Compute halo virial radii and Cartesian coordinates.
3. Run halo remapping to build halo_id mapping.
4. Apply halo_id mapping to galaxy_df (preserve original where no mapping).
5. Run galaxy remapping → reassign isolated galaxies within halo virial radii.
6. Save final galaxy_id → new_group_id table.



Halo Remapping (halos → halos)

INPUT: halo_df with positions (x,y,z), velocities, masses, virial radii, ids

1. Build KDTree of halo positions.

2. Select halos with log_mass > threshold, sort them descending by mass.

3. Initialize mapping: each halo_id → itself.
   Mark all halos as available.

4. FOR each central halo in sorted order:
      IF central is not available → skip
      Mark central as unavailable (it cannot be claimed by others)
      
      Query KDTree for up to N nearest neighbor halos.
      
      FOR each neighbor halo:
          IF neighbor is available AND neighbor_id ≠ central_id:
              IF distance(central, neighbor) ≤ central_virial_radius:
                  Reassign neighbor’s halo_id → central’s halo_id
                  Mark neighbor as unavailable (so it can’t be claimed again)

5. Return mapping {old_halo_id → new_halo_id}.



Galaxy Remapping (galaxies → halos)

INPUT: halo_df (with virial radii, masses), galaxy_df (with positions, ids)

1. Build KDTree of galaxy positions.

2. Select halos with log_mass > threshold, sort them descending by mass.

3. Mark galaxies as available ONLY if galaxy.id_group_sky == -1 (isolated).
   Initialize galaxy_to_halo map = keep current id for all galaxies.

4. FOR each central halo in sorted order:
      Query KDTree for galaxies within central_virial_radius.
      
      FOR each candidate galaxy:
          IF galaxy is available (isolated, not yet reassigned):
              Reassign galaxy’s id_group_sky → central’s halo_id
              Mark galaxy as unavailable

5. Return updated galaxy_df and mapping {galaxy_index → new_halo_id}.
