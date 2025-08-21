import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck13
cosmo=Planck13
# Importing modules
from astropy.io import fits
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

# This script processes the GALFORM mock galaxy catalog from GAMA
#gama4_path = '/Users/sp624AA/Downloads/gama3/gkvScienceCatv02.fits'
#group_gal_path = '/Users/sp624AA/Downloads/gama3/G3CGal.fits'
#group_info_path = '/Users/sp624AA/Downloads/gama3/G3CFoFGroup.fits'

# Reading in GAMA input catalogue data from the equatorial regions
df=pd.read_csv('/Users/sp624AA/Downloads/G09_G12_G15_r21.csv')


df=df.set_index('uberID')
df['mag_rt']=8.9-2.5*np.log10(df['flux_rt'])

# Importing galaxies from the GAMA group catalogue
hdul = fits.open('/Users/sp624AA/Downloads/gama3/G3CGal.fits')
data = hdul[1].data
t=Table(data)
names = [name for name in t.colnames if len(t[name].shape) <= 1]
df_gals=t[names].to_pandas()

df_gals=df_gals.set_index('CATAID')

# Removing the G02 region of GAMA 
df_gals=df_gals[df_gals['RA']>100]


# Crossmatching GAMA input catalogue with spec galaxies
from astropy.coordinates import SkyCoord
import astropy.units as u

max_sep=1*u.arcsec
parent = SkyCoord(ra=df['RAmax'].values*u.degree, dec=df['Decmax'].values*u.degree)
gama = SkyCoord(ra=df_gals['RA'].values*u.degree, dec=df_gals['Dec'].values*u.degree)
idx, d2d, d3d = parent.match_to_catalog_sky(gama)
sep_constraint = d2d < max_sep
parent_matches=df.iloc[sep_constraint]
gama_matches=df_gals.iloc[idx[sep_constraint]]

# Removing the spectroscopic galaxies from the photometric catalogue
df=df.drop(parent_matches.index)

# Reading in GAMA groups
dfHalo=pd.read_csv('/Users/sp624AA/Downloads/GAMA_groups.csv')

dfHalo=dfHalo.set_index('GroupID')


#dfHalo['Z']=dfHalo['IterCenZ']
#dfHalo['RA']=dfHalo['IterCenRA']
#dfHalo['Dec']=dfHalo['IterCenDec']

H0 = cosmo.H(0)
h=H0.value/100

# Removing G02
#dfHalo=dfHalo[(dfHalo['RA']>100)]
df_gals=df_gals[(df_gals['RA']>100)]

dfHalo['LumDistance'] = cosmo.luminosity_distance(np.array(dfHalo['Z'])).value * h * 1e6
df_gals['LumDistance'] = cosmo.luminosity_distance(np.array(df_gals['Z'])).value  * h * 1e6


# Magnitude to luminosity conversion
def lumconv(mag):
    return  10**(0.4 *(4.67- mag))

# Luminosity to halo mass conversion 
def massconv(lum):
    return 1e14 * h**-1 * (0.81) * (lum/(10**11.5))**(1.01)


# K corrections
a = [0.2085, 1.0226, 0.5237, 3.5902, 2.3843]
zref = 0
Q0 = 1.75
zp = 0.2


def kcorr(z):
    k=0
    for i in range(len(a)):
        k+= (a[i] * (z - zp)**i)

    return k - (Q0 * (z-zref))


df_gals['Kcorr'] = df_gals['Z'].apply(kcorr)

# Finding luminosity of GAMA galaxies
df_gals['AbMag'] = df_gals['Rpetro']-5*np.log10(df_gals['LumDistance'])+5 - df_gals['Kcorr']
df_gals['lum']=lumconv(df_gals['AbMag'])


print(df_gals.columns)

print(dfHalo.columns)

print(dfHalo.index)

#mix 
result = df_gals.merge(dfHalo, left_on='GroupID', right_index=True, how='left')
print(result.columns)

print(result.head())

print(np.unique(result['MassA']), len(np.unique(result['MassA'])))


result.to_parquet('/Users/sp624AA/Downloads/gama3/G3CGal_processed.parquet', index=False)
# Filter gama 4
#data = data[data['SC'] >= 7]

# Convert flux_rt to apparent magnitude


#
# Columns names=('GalID','RA','DEC','Zspec','Rpetro','DM_100_25_75','SelID','HaloID','MillSimID','GroupID','GroupIDDeep','Volume','k-e corr','Rpetro_abs')
# Area is the same as the OG(not final) GAMA Eq regions; namely
# - [129.0, 141.0, −1, 3]
# - [174.0, 186.0, −2, 2]
# - [211.5, 223.5, −2, 2]

# Select Volume = 0


# Cut at Zspec = 0.2

#data = data[data['Zspec'] < 0.2]



#def k_corr(zspec):
#    """ k-e correction from g3cv1 paper"""
#    z_ref = 0
#    Q_z_ref = 1.75
#    z_p = 0.2
#    N = 4
#    a = [0.2085, 1.0226, 0.5237, 3.5902, 2.3843]
#    k_e = Q_z_ref*(zspec-z_ref)
#    for i in range(N+1):
#        #print(i)
#        k_e += a[i]*((zspec-z_p)**i) 
#    return k_e


#data['k-e corr'] = k_corr(data['Zspec'])
#data['Rpetro_abs'] = data['Rpetro'] - data['DM_100_25_75'] - data['k-e corr']

# Save data as a parquet file in same directory as fits file
# print len (data) and columns
#print(f"Processed data contains {len(data)} rows and {len(data.columns)} columns")

#output_path = path.replace('.fits', '_processed.parquet')
#data.write(output_path, format='parquet', overwrite=True)
#print(f"Processed data saved to {output_path}")