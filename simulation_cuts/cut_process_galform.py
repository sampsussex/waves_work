import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# This script processes the GALFORM mock galaxy catalog from GAMA

path = '/Users/sp624AA/Downloads/mocks/GALFORM/G3CMockGalv04.fits'

# Load the data
print(f"Loading data from {path}")
data = Table.read(path)
# Print the first few rows of the data
print(f"Data loaded with {len(data)} rows and {len(data.columns)} columns")

#
#Columns names=('GalID','RA','DEC','Zspec','Rpetro','DM_100_25_75','SelID','HaloID','MillSimID','GroupID','GroupIDDeep','Volume','k-e corr','Rpetro_abs')
# Area is the same as the OG(not final) GAMA Eq regions; namely
# - [129.0, 141.0, −1, 3]
# - [174.0, 186.0, −2, 2]
# - [211.5, 223.5, −2, 2]

# Select Volume = 0

data = data[data['Volume'] == 0]

# Cut at Zspec = 0.2

data = data[data['Zspec'] < 0.2]

def k_corr(zspec):
    """ k-e correction from g3cv1 paper"""
    z_ref = 0
    Q_z_ref = 1.75
    z_p = 0.2
    N = 4
    a = [0.2085, 1.0226, 0.5237, 3.5902, 2.3843]
    k_e = Q_z_ref*(zspec-z_ref)
    for i in range(N+1):
        #print(i)
        k_e += a[i]*((zspec-z_p)**i) 
    return k_e

data['k-e corr'] = k_corr(data['Zspec'])

data['Rpetro_abs'] = data['Rpetro'] - data['DM_100_25_75'] - data['k-e corr']

# Save data as a parquet file in same directory as fits file
output_path = path.replace('.fits', '_processed.parquet')
data.write(output_path, format='parquet', overwrite=True)
print(f"Processed data saved to {output_path}")