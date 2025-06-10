import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from astropy.io import fits

cosmos_path = '/Users/sp624AA/Downloads/waves_photo_zs/speczcompilation/specz_compilation/specz_compilation_COSMOS_DR1.00_unique.fits'
wd_10_path = '/Users/sp624AA/Downloads/waves_photo_zs/WD10_2p4.parquet'

def load_pq_with_filters_pyarrow(file_path, columns, filters=None):
    """Load specific columns with filters using PyArrow."""
    
    # PyArrow filters format: [('column', 'operator', value)]
    # Example: [('magnitude', '<', 20), ('ra', '>', 150)]
    
    table = pq.read_table(file_path, columns=columns, filters=filters)
    df = table.to_pandas()

    return Table.from_pandas(df)

def load_fits_with_filters(file_path, columns=None, filters=None, hdu=1):
    """
    Load specific columns with filters from FITS files.
    
    Parameters:
    -----------
    file_path : str
        Path to FITS file
    columns : list, optional
        List of columns to load. If None, loads all columns.
    filters : list, optional
        List of filter tuples: [('column', 'operator', value)]
        Operators: '<', '>', '<=', '>=', '==', '!='
        Example: [('magnitude', '<', 20), ('ra', '>', 150)]
    hdu : int, optional
        HDU number to read (default: 1 for first extension)
        
    Returns:
    --------
    pandas.DataFrame
        Filtered data with selected columns
    """
    
    # Open FITS file
    with fits.open(file_path) as hdul:
        
        # Get the data HDU (usually extension 1 for tables)
        data = hdul[hdu].data
        
        if data is None:
            raise ValueError(f"No data found in HDU {hdu}")
        
        # Convert to astropy Table for easier manipulation
        table = Table(data)
        
        # Apply filters first (before column selection for efficiency)
        if filters is not None:
            mask = np.ones(len(table), dtype=bool)
            
            for column, operator, value in filters:
                if column not in table.colnames:
                    raise ValueError(f"Column '{column}' not found in FITS file")
                
                col_data = table[column]
                
                if operator == '<':
                    mask &= (col_data < value)
                elif operator == '>':
                    mask &= (col_data > value)
                elif operator == '<=':
                    mask &= (col_data <= value)
                elif operator == '>=':
                    mask &= (col_data >= value)
                elif operator == '==':
                    mask &= (col_data == value)
                elif operator == '!=':
                    mask &= (col_data != value)
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
            
            # Apply the mask
            table = table[mask]
        
        # Select specific columns if requested
        if columns is not None:
            # Check that all requested columns exist
            missing_cols = [col for col in columns if col not in table.colnames]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            
            table = table[columns]
        
        # Convert to pandas DataFrame
        df = table.to_pandas()
        
        return Table.from_pandas(df)


def match_catalogs(ra1, dec1, ra2, dec2, max_sep=1.0*u.arcsec, 
                   catalog1_data=None, catalog2_data=None):
    """
    Match two catalogs based on RA/Dec coordinates using astropy.
    
    Parameters:
    -----------
    ra1, dec1 : array-like
        RA and Dec coordinates of first catalog (in degrees)
    ra2, dec2 : array-like  
        RA and Dec coordinates of second catalog (in degrees)
    max_sep : astropy Quantity
        Maximum separation for a match (default: 1 arcsec)
    catalog1_data, catalog2_data : dict, optional
        Additional data columns for each catalog
        
    Returns:
    --------
    matches : astropy.table.Table
        Table containing matched sources with separation info
    """

    # Create SkyCoord objects
    coord1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree, frame='icrs')
    coord2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree, frame='icrs')
    
    # Find matches using astropy's match_coordinates_sky
    idx, d2d, d3d = match_coordinates_sky(coord1, coord2)
    
    # Apply separation cut
    good_matches = d2d < max_sep
    
    # Create results table
    n_matches = np.sum(good_matches)
    print(f"Found {n_matches} matches within {max_sep}")
    
    if n_matches == 0:
        return Table()
    
    # Build the matched catalog table
    matched_table = Table()
    
    # Add coordinate info
    #matched_table['cat1_idx'] = np.arange(len(ra1))[good_matches]
    #matched_table['cat2_idx'] = idx[good_matches]
    matched_table['ra1'] = ra1[good_matches]
    matched_table['dec1'] = dec1[good_matches]
    matched_table['ra2'] = ra2[idx[good_matches]]
    matched_table['dec2'] = dec2[idx[good_matches]]
    matched_table['separation_arcsec'] = d2d[good_matches].to(u.arcsec).value
    
    # Add additional data if provided
    if catalog1_data is not None:
        for key, values in catalog1_data.items():
            matched_table[f'{key}'] = values[good_matches]
            
    if catalog2_data is not None:
        for key, values in catalog2_data.items():
            matched_table[f'{key}'] = values[idx[good_matches]]
    
    return matched_table


def plot_matches(ra1, dec1, ra2, dec2, matches):
    """Plot separation histogram and RA/Dec separation plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if len(matches) == 0:
        ax1.text(0.5, 0.5, 'No matches found', ha='center', va='center',
                transform=ax1.transAxes, fontsize=14)
        ax1.set_title('No Matches')
        ax2.text(0.5, 0.5, 'No matches found', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('No Matches')
        plt.tight_layout()
        plt.show()
        return
    
    # Plot 1: Separation histogram
    ax1.hist(matches['separation_arcsec'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax1.set_xlabel('Separation (arcsec)')
    ax1.set_ylabel('Number of matches')
    ax1.set_title('Match Separation Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    ax1.axvline(1, color='red', linestyle='--', alpha=0.8, label=f'Max sep: {1:.3f}"')
    ax1.legend()
    
    # Plot 2: RA/Dec separation plot
    # Calculate RA and Dec differences (in arcsec)
    ra_diff = (matches['ra2'] - matches['ra1']) * 3600 * np.cos(np.radians(matches['dec1']))  # Account for Dec projection
    dec_diff = (matches['dec2'] - matches['dec1']) * 3600
    
    scatter = ax2.scatter(ra_diff, dec_diff, alpha=0.05, s=2)
    ax2.set_xlabel('ΔRA (arcsec)')
    ax2.set_ylabel('ΔDec (arcsec)')
    ax2.set_title('Position Differences (WAVES RA/Dec max - COSMOS ra/dec _corrected)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='black', linestyle='-', alpha=0.3)
    

    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_aspect('equal')
    

    plt.tight_layout()
    plt.savefig('cosmos_wd10_crossmatch')
    plt.show()


def save_matches(matches):
    """ Save matches as a parquet file"""
    #print(matches)
    matches.remove_columns(['ra1', 'dec1', 'ra2', 'dec2', 'separation_arcsec'])
    print('Top 10 from table:')
    print(matches[0:10])
    matches.write('cosmos_waves_crossmatch.parquet', format='parquet', overwrite =True)


# Example usage
if __name__ == "__main__":
    """
    # Create sample catalogs
    ra1, dec1, ra2, dec2, cat1_data, cat2_data = create_sample_catalogs()
    
    print("Matching catalogs...")
    print(f"Catalog 1: {len(ra1)} sources")
    print(f"Catalog 2: {len(ra2)} sources")
    
    # Match the catalogs
    matches = match_catalogs(ra1, dec1, ra2, dec2, 
                           max_sep=2.0*u.arcsec,  # 2 arcsec matching radius
                           catalog1_data=cat1_data,
                           catalog2_data=cat2_data)
    
    
    # Plot results
    plot_matches(ra1, dec1, ra2, dec2, matches)
    
    # Example: Load from files (commented out)
    """

    # Load files
    cat1 = load_fits_with_filters(cosmos_path, ['Id_specz', 'ra_corrected','dec_corrected','specz', 'Confidence_level'], 
                                     [('Confidence_level', '>=', 95)])
    

    cat2 = load_pq_with_filters_pyarrow(wd_10_path, ['uberID', 'RAmax', 'Decmax', 'class', 'duplicate', 'mask'], 
                                     [('class', '!=', 'artefact'), ('mask', '==', 0), ('duplicate', '==', 0)])
    
    
    # Extract coordinates
    ra1, dec1 = cat1['ra_corrected'], cat1['dec_corrected'] 
    ra2, dec2 = cat2['RAmax'], cat2['Decmax']
    
    # Additional data
    cat1_data = {'Id_specz': cat1['Id_specz'], 'specz': cat1['specz'], 'Confidence_level': cat1['Confidence_level']}
    cat2_data = {'uberID': cat2['uberID']}
    
    matches = match_catalogs(ra1, dec1, ra2, dec2, 
                           max_sep=10.0*u.arcsec,
                           catalog1_data=cat1_data, 
                           catalog2_data=cat2_data)
    

        # Analyze results
    #analyze_matches(matches)
    
    # Plot results
    plot_matches(ra1, dec1, ra2, dec2, matches)


    matches = match_catalogs(ra1, dec1, ra2, dec2, 
                        max_sep=1.0*u.arcsec,
                        catalog1_data=cat1_data, 
                        catalog2_data=cat2_data)
    save_matches(matches)

    

