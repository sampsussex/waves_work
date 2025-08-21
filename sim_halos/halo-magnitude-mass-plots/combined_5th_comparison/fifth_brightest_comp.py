import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def process_dataset_1(parquet_file_path):
    """
    Process the first dataset (WAVES wide).
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_file_path)
    
    # Rename columns for easier handling
    column_mapping = {
        "zobs": "redshift",
        "total_ab_dust_r_VST": "absolute_magnitude",
        "total_ap_dust_r_VST": "apparent_magnitude", 
        'id_group_sky': "group_id",
    }
    
    # Only rename columns that exist in the dataframe
    existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_columns)

    df['log_stellar_mass'] = np.log10((df['mstars_disk'] + df['mstars_bulge'])/0.67)
    mask = (df['log_stellar_mass'] > 8) & (df["absolute_magnitude"] > -200) & (df['apparent_magnitude'] < 19.65) & (df['redshift'] < 0.2) #(df['ra'] < 20) #(df['redshift'] < 0.5) #&  (df['apparent_magnitude'] < 19.8)
    df = df[mask]

    ra_min, ra_max = 157.00, 189.96
    dec_min, dec_max = -3.00, 4.

    df = df[
        (df['ra'] >= ra_min) & (df['ra'] <= ra_max) &
        (df['dec'] >= dec_min) & (df['dec'] <= dec_max)
]
    
    print(f"Dataset 1 - Total galaxies in catalog: {len(df)}")
    print(f"Dataset 1 - Total unique halos: {df['group_id'].nunique()}")
    
    # Group by halo and count members
    halo_counts = df.groupby('group_id').size()
    halos_with_5_plus = halo_counts[halo_counts >= 5].index
    print(f"Dataset 1 - Halos with 5+ members: {len(halos_with_5_plus)}")
    
    # Filter to only halos with 5+ members
    df_filtered = df[df['group_id'].isin(halos_with_5_plus)]
    
    # Sort by apparent magnitude for this dataset
    df_sorted = df_filtered.sort_values(['group_id', 'apparent_magnitude'])
    
    # Get the 5th brightest galaxy in each halo
    fifth_brightest = df_sorted.groupby('group_id').nth(4).reset_index()
    
    # Find mass columns
    mass_columns = [col for col in df.columns if 'mass' in col.lower() or 'mvir' in col.lower() or 'mhalo' in col.lower()]
    
    if not mass_columns:
        print("Dataset 1 - Warning: No halo mass column found.")
        return None
    
    mass_column = mass_columns[0]
    print(f"Dataset 1 - Using mass column: {mass_column}")
    
    # Get halo masses
    halo_masses = df_filtered.groupby('group_id')[mass_column].first()
    fifth_brightest = fifth_brightest.merge(
        halo_masses.reset_index().rename(columns={mass_column: 'halo_mass'}), 
        on='group_id'
    )
    
    # Remove NaN values
    fifth_brightest = fifth_brightest.dropna(subset=['absolute_magnitude', 'halo_mass'])

    # remove NaN halo masses
    fifth_brightest = fifth_brightest.dropna(subset=['halo_mass'])

    # Remove galaxies with NaN absolute magnitudes
    fifth_brightest = fifth_brightest.dropna(subset=['absolute_magnitude'])

    # now remove those with inf of -inf magnitudes

    fifth_brightest = fifth_brightest[~fifth_brightest['absolute_magnitude'].isin([np.inf, -np.inf])]
    
    print(f"Dataset 1 - Final data points: {len(fifth_brightest)}")
    
    return fifth_brightest

def process_dataset_2(parquet_file_path):
    """
    Process the second dataset (GAMA4).
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_file_path)
    
    # Rename columns for easier handling
    column_mapping = {
        'abs_mag_rt': "absolute_magnitude",
        'app_mag_rt': "apparent_magnitude",
        'group_id': "group_id"
    }


    
    
    # Only rename columns that exist in the dataframe
    existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_columns)
    
    print(f"Dataset 2 - Total galaxies in catalog: {len(df)}")
    print(f"Dataset 2 - Total unique halos: {df['group_id'].nunique()}")
    print(df.columns)
    df = df[df['median_redshift'] < 0.2]
    
    # Group by halo and count members
    halo_counts = df.groupby('group_id').size()
    halos_with_5_plus = halo_counts[halo_counts >= 5].index
    print(f"Dataset 2 - Halos with 5+ members: {len(halos_with_5_plus)}")
    
    # Filter to only halos with 5+ members
    df_filtered = df[df['group_id'].isin(halos_with_5_plus)]
    
    # Sort by absolute magnitude for this dataset
    df_sorted = df_filtered.sort_values(['group_id', 'absolute_magnitude'])
    
    # Get the 5th brightest galaxy in each halo
    fifth_brightest = df_sorted.groupby('group_id').nth(4).reset_index()
    
    # Find mass columns
    mass_columns = [col for col in df.columns if 'massa' in col.lower()]
    
    if not mass_columns:
        print("Dataset 2 - Warning: No halo mass column found.")
        return None
    
    mass_column = mass_columns[0]
    print(f"Dataset 2 - Using mass column: {mass_column}")
    
    # Get halo masses and convert to log scale
    halo_masses = df_filtered.groupby('group_id')[mass_column].first()
    halo_masses = np.log10(halo_masses)
    fifth_brightest = fifth_brightest.merge(
        halo_masses.reset_index().rename(columns={mass_column: 'halo_mass'}), 
        on='group_id'
    )
    
    # Remove NaN values
    fifth_brightest = fifth_brightest.dropna(subset=['absolute_magnitude', 'halo_mass'])

    # remove NaN halo masses
    fifth_brightest = fifth_brightest.dropna(subset=['halo_mass'])

    # Remove galaxies with NaN absolute magnitudes
    fifth_brightest = fifth_brightest.dropna(subset=['absolute_magnitude'])

    # now remove those with inf of -inf magnitudes
    fifth_brightest = fifth_brightest[~fifth_brightest['absolute_magnitude'].isin([np.inf, -np.inf])]
    # now on halo mass
    fifth_brightest = fifth_brightest[~fifth_brightest['halo_mass'].isin([np.inf, -np.inf])]
    
    print(f"Dataset 2 - Final data points: {len(fifth_brightest)}")
    
    return fifth_brightest


def process_dataset_2(parquet_file_path):
    """
    Process the third dataset (Sharks processed like GAMA).
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_file_path)
    
    # Rename columns for easier handling
    column_mapping = {
        'abs_mag_rt': "absolute_magnitude",
        'app_mag_rt': "apparent_magnitude",
        'group_id': "group_id"
    }


    # Only rename columns that exist in the dataframe
    existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_columns)
    
    print(f"Dataset 2 - Total galaxies in catalog: {len(df)}")
    print(f"Dataset 2 - Total unique halos: {df['group_id'].nunique()}")
    print(df.columns)
    df = df[df['median_redshift'] < 0.2]
    
    # Group by halo and count members
    halo_counts = df.groupby('group_id').size()
    halos_with_5_plus = halo_counts[halo_counts >= 5].index
    print(f"Dataset 2 - Halos with 5+ members: {len(halos_with_5_plus)}")
    
    # Filter to only halos with 5+ members
    df_filtered = df[df['group_id'].isin(halos_with_5_plus)]
    
    # Sort by absolute magnitude for this dataset
    df_sorted = df_filtered.sort_values(['group_id', 'absolute_magnitude'])
    
    # Get the 5th brightest galaxy in each halo
    fifth_brightest = df_sorted.groupby('group_id').nth(4).reset_index()
    
    # Find mass columns
    mass_columns = [col for col in df.columns if 'massa' in col.lower()]
    
    if not mass_columns:
        print("Dataset 2 - Warning: No halo mass column found.")
        return None
    
    mass_column = mass_columns[0]
    print(f"Dataset 2 - Using mass column: {mass_column}")
    
    # Get halo masses and convert to log scale
    halo_masses = df_filtered.groupby('group_id')[mass_column].first()
    halo_masses = np.log10(halo_masses)
    fifth_brightest = fifth_brightest.merge(
        halo_masses.reset_index().rename(columns={mass_column: 'halo_mass'}), 
        on='group_id'
    )
    
    # Remove NaN values
    fifth_brightest = fifth_brightest.dropna(subset=['absolute_magnitude', 'halo_mass'])

    # remove NaN halo masses
    fifth_brightest = fifth_brightest.dropna(subset=['halo_mass'])

    # Remove galaxies with NaN absolute magnitudes
    fifth_brightest = fifth_brightest.dropna(subset=['absolute_magnitude'])

    # now remove those with inf of -inf magnitudes
    fifth_brightest = fifth_brightest[~fifth_brightest['absolute_magnitude'].isin([np.inf, -np.inf])]
    # now on halo mass
    fifth_brightest = fifth_brightest[~fifth_brightest['halo_mass'].isin([np.inf, -np.inf])]
    
    print(f"Dataset 2 - Final data points: {len(fifth_brightest)}")
    
    return fifth_brightest


def create_overlaid_contour_plot(data1, data2, data3, dataset1_name="SHARKS-r<19.65 Sim Values", dataset2_name="GAMA4-Nessie pipeline", dataset3_name="SHARKS-Nessie pipeline"):
    """
    Create overlaid contour plots for both datasets with MNRAS formatting.
    """
    # Set MNRAS style parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.0,
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Process dataset 1
    x1 = data1['halo_mass']
    y1 = data1['absolute_magnitude']
    
    # Convert halo mass to log scale if needed for dataset 1
    if x1.min() > 1e10:
        x1_log = np.log10(x1)
    else:
        x1_log = x1
    
    # Process dataset 2 (already in log scale)
    x2_log = data2['halo_mass']
    y2 = data2['absolute_magnitude']
    
    # Create grid for contours (use fixed limits)
    x_grid = np.linspace(12, 16, 50)
    y_grid = np.linspace(-24, -16, 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Process dataset 3
    x3_log = data3['halo_mass']
    y3 = data3['absolute_magnitude']


    
    # Create KDE for dataset 1
    try:
        xy1 = np.vstack([x1_log, y1])
        kde1 = gaussian_kde(xy1)
        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
        Z1 = kde1(positions).reshape(X_grid.shape)
        
        # Create contour plot for dataset 1 (top 3 levels only)
        contour1 = ax.contour(X_grid, Y_grid, Z1, levels=3, colors='blue', 
                             linewidths=1.2, linestyles='solid')
        
    except Exception as e:
        print(f"Could not create contours for dataset 1: {e}")
    
    # Create KDE for dataset 2
    try:
        xy2 = np.vstack([x2_log, y2])
        kde2 = gaussian_kde(xy2)
        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
        Z2 = kde2(positions).reshape(X_grid.shape)
        
        # Create contour plot for dataset 2 (top 3 levels only)
        contour2 = ax.contour(X_grid, Y_grid, Z2, levels=3, colors='red', 
                             linewidths=1.2, linestyles='dashed')

        
    except Exception as e:
        print(f"Could not create contours for dataset 2: {e}")

    # Create KDE for dataset 3
    try:
        xy3 = np.vstack([x3_log, y3])
        kde3 = gaussian_kde(xy3)
        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
        Z3 = kde3(positions).reshape(X_grid.shape)
        
        # Create contour plot for dataset 3 (top 3 levels only)
        contour3 = ax.contour(X_grid, Y_grid, Z3, levels=3, colors='green', 
                             linewidths=1.2, linestyles='dotted')
    
    except Exception as e:
        print(f"Could not create contours for dataset 3: {e}")
    # Scatter plot the actual data points for reference
    ax.scatter(x1_log, y1, c='lightblue', alpha=0.4, s=3, marker='o', 
               rasterized=True)
    ax.scatter(x2_log, y2, c='lightcoral', alpha=0.4, s=3, marker='s', 
               rasterized=True)
    ax.scatter(x3_log, y3, c='lightgreen', alpha=0.4, s=3, marker='^', 
               rasterized=True)
    

    
    # Labels and title
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo}/{\rm M}_{\odot})$')
    ax.set_ylabel(r'$M_{5,{\rm abs}}$ [mag]')
    ax.set_title('Halo mass vs abs mag of 5th brightest galaxy (230 [deg$^2$])')
    
    # Set axis limits
    ax.set_xlim(12, 16)
    ax.set_ylim(-24, -16)  # Corrected order for inverted axis
    
    # Invert y-axis since brighter objects have smaller magnitudes
    ax.invert_yaxis()
    
    # Add minor ticks and grid
    ax.minorticks_on()
    ax.grid(True, which='major', alpha=0.3, linewidth=0.5)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=1.2, linestyle='-', label=f'{dataset1_name}, N: {len(data1)}'),
        Line2D([0], [0], color='red', lw=1.2, linestyle='--', label=f'{dataset2_name}, N: {len(data2)}'),
        Line2D([0], [0], color='green', lw=1.2, linestyle=':', label=f'{dataset3_name}, N: {len(data3)}'),
    ]
    
    ax.legend(handles=legend_elements, frameon=True)
    
    # Add sample size info
    #ax.text(0.05, 0.95, f'{dataset1_name}: N = {len(data1)}\n{dataset2_name}: N = {len(data2)}', 
    #        transform=ax.transAxes, verticalalignment='top', 
    #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
    
    plt.tight_layout()
    #plt.savefig('overlaid_contour_comparison_gama_z0-2.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('overlaid_contour_comparison_gama4_z0-2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, ax

def main():
    """
    Main function to process both datasets and create the overlaid contour plot.
    """
    # File paths - update these to your actual file paths
    file1 = "/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet"
    #file2 = '/Users/sp624AA/Downloads/gama3/G3CGal_processed.parquet'
    file2 = '/Users/sp624AA/Downloads/gama3/gama4_nessie_groups.parquet'

    file3 = '/Users/sp624AA/Downloads/mocks/sharks_nessie_groups.parquet'
    
    print("Processing Dataset 1 (WAVES Wide)...")
    data1 = process_dataset_1(file1)
    
    print("\nProcessing Dataset 2 (GAMA3)...")
    data2 = process_dataset_2(file2)

    print("\nProcessing Dataset 3 (SHARKS processed like GAMA)...")
    data3 = process_dataset_2(file3)
    
    if data1 is not None and data2 is not None and data3 is not None:
        print("\nCreating overlaid contour plot...")
        fig, ax = create_overlaid_contour_plot(data1, data2, data3)
        
        return data1, data2, data3, fig, ax
    else:
        print("Could not process one or both datasets.")
        return None

# Usage
if __name__ == "__main__":
    result = main()