#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Import and plot styles
from astroquery.gaia import Gaia
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

# Query for waves N region. 0.5 degrees extra around region is added, to aviod any boundary masking issues.
# g band mag cut listed from Bellstedt 2020. GAIA DR2 is consistent with the existing starmask



# orginial N window - 157.25 225.0 -3.95 3.95
query_waves_n = """
SELECT phot_g_mean_mag, ra, dec
FROM gaiadr2.gaia_source
WHERE phot_g_mean_mag < 16
AND ra > 156.75
AND ra < 225.5
AND dec > -4.45
and dec < 4.45"""

# Perfom Query
job_waves_n = Gaia.launch_job_async(query_waves_n)

total_r = job_waves_n.get_results()

print('Total length before cuts:', len(total_r))

total_r['radius'] = (10**(1.6-0.15*total_r['phot_g_mean_mag'])) / 60 #Radius-g mag relation from Bellstedt 2020.

#This cut is listed in Bellstedt 2020. Interpretation not 100% clear
for j in range(len(total_r)):
    if total_r['radius'][j] > 5/60:
        total_r['radius'][j] = 5/60 


print('Total length after cuts:', len(total_r))

#Save down results
total_r.write('GAIA_WAVES_N_MASK.fits', format='fits', overwrite=True) #Save as fits


# In[16]:


# orginial S window - 330.0 51.6 -35.6 -27.0
query_waves_s = """
SELECT phot_g_mean_mag, ra, dec
FROM gaiadr2.gaia_source
WHERE phot_g_mean_mag < 16
AND (
    (ra > 329.5 OR ra < 52.1)
)
AND dec > -36.1
and dec < -26.5"""

# Perfom Query
job_waves_s = Gaia.launch_job_async(query_waves_s)

total_r = job_waves_s.get_results()

print('Total length before cuts:', len(total_r))

total_r['radius'] = (10**(1.6-0.15*total_r['phot_g_mean_mag'])) / 60 #Radius-g mag relation from Bellstedt 2020.

#This cut is listed in Bellstedt 2020. Interpretation not 100% clear
for j in range(len(total_r)):
    if total_r['radius'][j] > 5/60:
        total_r['radius'][j] = 5/60 

        
print('Total length after cuts:', len(total_r))

total_r.write('GAIA_WAVES_S_MASK.fits', format='fits', overwrite=True) #Save as fits



