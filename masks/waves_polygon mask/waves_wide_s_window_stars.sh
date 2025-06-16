# Construct mangle-compatible Waves wide mask.    
# Files in apollo /sp624/waves_mask

alias stilts="/research/astro/gama/loveday/sw/stilts"

# Define waves S region as 1 rectangles in waves_n_rect.dat
# Convert to polygon format (default weight=1)
echo "# WavesWide South 
# ra_min ra_max dec_min dec_max 
330.0 51.6 -35.6 -27.0" > waves_s_rect.dat

# Point to mangle directory
export MANGLE_DIR="/research/astro/gama/loveday/sw/mangle2.2"

# Create waves N region ply file
$MANGLE_DIR/bin/poly2poly -ir1 waves_s_rect.dat waves_s_rect.ply


echo 0 > zero #zero referenced in weighting later on.

# Python script ran to create list of Sabines's regions as vertex points. 
# Run script saved in this directory, then continue. 

# Convert to polygon format with weight=0
$MANGLE_DIR/bin/weight -iv -zzero Sabines_NGCS/sabines_ngcs.dat ngc_reg.ply

# Set mask radius limits for bright Gaia stars. GAIA DR3 Queried with python code in this folder. 
# Radius set where r=10^(1.6-0.15g) for objects with g (mag) < 16.0
stilts tpipe ifmt=fits in=/mnt/lustre/projects/astro/general/sp624/waves_masks/gaia_stars/GAIA_WAVES_S_MASK.fits \
  cmd='keepcols "ra dec radius"' \
  ofmt=ascii out=gaia_mask_s.dat
  
# Convert to polygon format with weight=0
$MANGLE_DIR/bin/weight -ic1 -zzero gaia_mask_s.dat gaia_s_reg.ply

# Convert list of additional GCs and bright star to ply format
$MANGLE_DIR/bin/weight -ic1 -zzero Extra_waves_s_sources.dat extra_s_sources_reg.ply

# Combine (zero-weight) NGC masks with (unit-weight) waves region
$MANGLE_DIR/bin/pixelize waves_s_rect.ply ngc_reg.ply extra_s_sources_reg.ply gaia_s_reg.ply temp1

# Snap & balkanize
$MANGLE_DIR/bin/snap temp1 temp2
$MANGLE_DIR/bin/balkanize temp2 waves_wide_S_window_ngc_star_mask.ply

# Generate 10000000 randoms. 
$MANGLE_DIR/bin/ransack -r10000000 waves_wide_S_window_ngc_star_mask.ply waves_wide_S_window_ngc_star_mask_randoms.dat

