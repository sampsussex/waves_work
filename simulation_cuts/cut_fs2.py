import numpy as np
from astropy.table import Table

H = 67.51 / 100  # Hubble constant in km/s/Mpc
fs2_sel_path = '/Users/sp624AA/Downloads/mocks/21500.parquet'
data = Table.read(fs2_sel_path)


data = data[data['observed_redshift_gal'] < 0.2]

def ab_flux_to_mag_lsst_z(flux):
    """
    Convert flux to LSST z-band magnitude.
    """
    return -2.5 * np.log10(flux) - 48.6

def abs_flux_to_mag_lsst_z(flux):
    """
    Convert absolute flux to LSST z-band magnitude.
    """

    return -2.5 * np.log10(flux) - 48.6 + 5 * np.log10(H)

data['total_abs_lsst_z'] = abs_flux_to_mag_lsst_z(data['lsst_z_abs'])


data['total_ap_lsst_z'] = ab_flux_to_mag_lsst_z(data['lsst_z_el_model3_ext'])


data = data[data['total_ap_lsst_z'] < 21.25]  # Apply the magnitude cut

data['unique_gal_id'] = np.arange(len(data))  # Create a unique ID for each row
# Save the filtered data to a new Parquet file
output_path = '/Users/sp624AA/Downloads/mocks/fs2_subselection.parquet'
data.write(output_path, format='parquet', overwrite=True)
print(f"Filtered data saved to {output_path}")
