# Construct mangle-compatible Waves wide mask.    
# Files in apollo /sp624/waves_mask

alias stilts="/research/astro/gama/loveday/sw/stilts"

# Define waves N region as 1 rectangles in waves_n_rect.dat
# Convert to polygon format (default weight=1)
echo "# WavesWide North 
# ra_min ra_max dec_min dec_max 
157.25 225.0 -3.95 3.95" > waves_n_rect.dat

# Point to mangle directory
export MANGLE_DIR="/research/astro/gama/loveday/sw/mangle2.2"

# Create waves N region ply file
$MANGLE_DIR/bin/poly2poly -ir1 waves_n_rect.dat waves_n_rect.ply


echo 0 > zero #zero referenced in weighting later on.

# Python script ran to create list of Sabines's regions as vertex points. 
# Run script saved in this directory, then continue. 
# Convert to polygon format with weight=0
$MANGLE_DIR/bin/weight -iv -zzero Sabines_NGCS/sabines_ngcs.dat ngc_reg.ply

# Set mask radius limits for bright Gaia stars. GAIA DR2 Queried with python code in folder. 
# Radius set where r=10^(1.6-0.15g) for objects with g (mag) < 16.0
stilts tpipe ifmt=fits in=/mnt/lustre/projects/astro/general/sp624/waves_masks/gaia_stars/GAIA_WAVES_N_MASK.fits \
  cmd='keepcols "ra dec radius"' \
  ofmt=ascii out=gaia_mask_n.dat
# Convert to polygon format with weight=0
$MANGLE_DIR/bin/weight -ic1 -zzero gaia_mask_n.dat gaia_n_reg.ply


# Combine (zero-weight) NGC masks with (unit-weight) waves region
$MANGLE_DIR/bin/pixelize waves_n_rect.ply ngc_reg.ply gaia_n_reg.ply temp1

# Snap & balkanize
$MANGLE_DIR/bin/snap temp1 temp2
$MANGLE_DIR/bin/balkanize temp2 waves_wide_N_window_ngc_star_mask.ply

#Generate 1000000 randoms. 
$MANGLE_DIR/bin/ransack -r10000000 waves_wide_N_window_ngc_star_mask.ply waves_wide_N_window_ngc_star_mask_randoms.dat

