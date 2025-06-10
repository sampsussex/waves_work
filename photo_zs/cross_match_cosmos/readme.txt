# Quick script to cross match the cosmos catalog with WAVES WD10 DDF field. 
# Sam Philipsborn 10-06-25
# https://github.com/sampsussex/waves_work/tree/main/photo_zs/cross_match_cosmos
#
# Here we cross match COSMOS DR1 (https://arxiv.org/abs/2503.00120) with WAVES DDF v2p4. 
# We choose a 1 arc sec matching radius on RAmax, Decmax from WAVES, and ra_corrected,
# dec_corrected from COSMOS. See plot for scatter on this.
#
# We filter on mask, duplicate == 0 from WAVES, as well as class != artefect. 
# We choose the specz_compilation_COSMOS_DR1.00_unique.fits (i,e no duplicates, best quailty
# spectra kept) file from cosmos, filtering on Confidence_level >= 95. 
#
# There are stars in this sample, where the Confidence_level is high but the redshift
# is very low or indeed 0. I have kept the class flag from WAVES if needed.
