import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
from matplotlib.lines import Line2D

def load_and_process_data(filepath, group_col, stellar_mass_col, halo_mass_col, min_group_size=5):
    """
    Load parquet file and process to get the most massive galaxy per halo with percentage calculations.
    
    Parameters:
    -----------
    filepath : str
        Path to the parquet file
    group_col : str
        Column name for group ID
    stellar_mass_col : str
        Column name for stellar mass
    halo_mass_col : str
        Column name for halo mass
    min_group_size : int
        Minimum number of members required in a group (default: 5)
    
    Returns:
    --------
    DataFrame with the most massive galaxy per qualifying group and percentage calculations
    """
    # Load the data
    df = pd.read_parquet(filepath)
    print(df.columns)
    
    if 'v0.3.0/' in filepath:
        print('Making alterations to raw sharks data')
        df['log_stellar_mass'] = np.log10((df['mstars_disk'] + df['mstars_bulge'])/0.67)
        mask = (df['log_stellar_mass'] > 8) & (df["total_ab_dust_r_VST"] > -200) & (df["total_ap_dust_r_VST"] < 19.65) & (df['zobs'] < 0.2)
        df = df[mask]

        ra_min, ra_max = 157.00, 189.96
        dec_min, dec_max = -3.00, 4.

        df = df[
            (df['ra'] >= ra_min) & (df['ra'] <= ra_max) &
            (df['dec'] >= dec_min) & (df['dec'] <= dec_max)]

    else:
        df = df[(df['Z'] > 0.002) & (df['Z'] < 0.2)] # Redshift cut

    # Ensure required columns exist
    required_cols = [group_col, stellar_mass_col, halo_mass_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {filepath}: {missing_cols}")
    
    # Rename columns to standard names for processing
    df = df.rename(columns={
        group_col: 'group_id',
        stellar_mass_col: 'stellar_mass',
        halo_mass_col: 'halo_mass'
    })
    df['stellar_mass'] = 10**df['stellar_mass']
    
    # Filter for groups with minimum number of members
    group_counts = df.groupby('group_id').size()
    valid_groups = group_counts[group_counts >= min_group_size].index
    df_filtered = df[df['group_id'].isin(valid_groups)]
    
    # Calculate total stellar mass per group
    group_total_stellar = df_filtered.groupby('group_id')['stellar_mass'].sum()
    
    # Get the galaxy with maximum stellar mass per group
    idx_max = df_filtered.groupby('group_id')['stellar_mass'].idxmax()
    df_max_per_group = df_filtered.loc[idx_max].copy()
    
    # Add total group stellar mass and calculate percentages
    df_max_per_group['total_group_stellar_mass'] = df_max_per_group['group_id'].map(group_total_stellar)
    df_max_per_group['pct_stellar_in_largest'] = (df_max_per_group['stellar_mass'] / 
                                                   df_max_per_group['total_group_stellar_mass']) * 100
    df_max_per_group['pct_stellar_not_in_largest'] = 100 - df_max_per_group['pct_stellar_in_largest']
    
    return df_max_per_group


def create_contour_plot(datasets, labels, colors=['blue', 'red', 'green'],
                        levels=2, alpha=0.4, figsize=(6, 5), 
                        y_column='stellar_mass', y_label=None, title=None,
                        use_log_y=True):
    """
    Create overlaid contour plots for multiple datasets.
    
    Parameters:
    -----------
    datasets : list of DataFrames
        Processed datasets containing stellar_mass and halo_mass
    labels : list of str
        Labels for each dataset
    colors : list of str
        Colors for each dataset's contours
    levels : int
        Number of contour levels (default: 3)
    alpha : float
        Transparency of scatter points (default: 0.4)
    figsize : tuple
        Figure size (default: (6, 5))
    y_column : str
        Column to use for y-axis (default: 'stellar_mass')
    y_label : str
        Label for y-axis (if None, auto-generated)
    title : str
        Plot title (if None, auto-generated)
    use_log_y : bool
        Whether to use log scale for y-axis
    """
    # Set style parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.0,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # First pass: collect all data to determine global ranges
    all_y_values = []
    all_halo = []
    
    for df in datasets:
        y_values = df[y_column].values
        halo_mass = df['halo_mass'].values
        mask = np.isfinite(y_values) & np.isfinite(halo_mass) & (y_values > 0) & (halo_mass > 0)
        all_y_values.extend(y_values[mask])
        all_halo.extend(halo_mass[mask])
    
    # Now process each dataset for plotting
    for i, (df, label, color) in enumerate(zip(datasets, labels, colors)):
        y_values = df[y_column].values
        halo_mass = df['halo_mass'].values
        
        # Remove any NaN or infinite values
        mask = np.isfinite(y_values) & np.isfinite(halo_mass) & (y_values > 0) & (halo_mass > 0)
        y_values = y_values[mask]
        halo_mass = halo_mass[mask]
        
        if len(y_values) < 2:
            print(f"Warning: Dataset {label} has insufficient valid data points")
            continue
        
        # Create 2D kernel density estimate
        try:
            print(f"Processing {label}: {len(y_values)} points")
            print(f"  {y_column} range: {y_values.min():.2e} - {y_values.max():.2e}")
            print(f"  Halo mass range: {halo_mass.min():.2e} - {halo_mass.max():.2e}")
            
            # Work in log space for halo mass and optionally for y values
            if use_log_y:
                y_values_transformed = np.log10(y_values)
            else:
                y_values_transformed = y_values
            halo_mass_log = np.log10(halo_mass)
            
            # Create KDE
            xy = np.vstack([halo_mass_log, y_values_transformed])
            kde = gaussian_kde(xy)
            
            # Create fixed grid for all contours (ensures alignment)
            x_min = np.log10(min(all_halo))
            x_max = np.log10(max(all_halo))
            if use_log_y:
                y_min = np.log10(min(all_y_values))
                y_max = np.log10(max(all_y_values))
            else:
                y_min = min(all_y_values)
                y_max = max(all_y_values)
            
            # Create evaluation grid
            n_grid = 50
            x_grid = np.linspace(x_min, x_max, n_grid)
            y_grid = np.linspace(y_min, y_max, n_grid)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            
            # Evaluate KDE on grid
            positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
            Z = kde(positions).reshape(X_grid.shape)
            
            # Determine line style based on dataset index
            linestyles = ['solid', 'dashed', 'dotted']
            linestyle = linestyles[i % 3]
            
            # Plot scatter points with light colors and different markers
            scatter_colors = ['lightblue', 'lightcoral', 'lightgreen']
            markers = ['o', 's', '^']
            scatter_color = scatter_colors[i]
            marker = markers[i % 3]
            
            ax.scatter(halo_mass_log, y_values_transformed, c=scatter_color, alpha=0.4, 
                      s=3, marker=marker, rasterized=True)
            
            # Create contour plot
            contour = ax.contour(X_grid, Y_grid, Z, levels=levels, colors=color, 
                                linewidths=1.2, linestyles=linestyle)
            
        except Exception as e:
            print(f"Error creating contour for dataset {label}: {e}")
            continue
    
    # Set axis labels and title
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo}/{\rm M}_{\odot})$', fontsize=11)
    
    if y_label is None:
        if y_column == 'stellar_mass':
            y_label = r'$\log_{10}(M_{\rm stellar}/{\rm M}_{\odot})$'
        elif y_column == 'pct_stellar_in_largest':
            y_label = r'% of Group Stellar Mass in Largest Galaxy'
        elif y_column == 'pct_stellar_not_in_largest':
            y_label = r'$\log_{10}$(% of Group Stellar Mass NOT in Largest Galaxy)'
    ax.set_ylabel(y_label, fontsize=11)
    
    if title is None:
        if y_column == 'stellar_mass':
            title = 'Halo Mass vs Stellar Mass of Most Massive Galaxy per Halo\n(Groups with ≥5 members)'
        elif y_column == 'pct_stellar_in_largest':
            title = 'Halo Mass vs % of Stellar Mass in Largest Galaxy\n(Groups with ≥5 members)'
        elif y_column == 'pct_stellar_not_in_largest':
            title = 'Halo Mass vs % of Stellar Mass NOT in Largest Galaxy\n(Groups with ≥5 members)'
    ax.set_title(title, fontsize=12, pad=20)
    
    # Add minor ticks and grid
    ax.minorticks_on()
    ax.grid(True, which='major', alpha=0.3, linewidth=0.5)
    
    # Create legend with custom line styles
    linestyles = ['solid', 'dashed', 'dotted']
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=1.2, linestyle=linestyles[i], 
               label=f'{labels[i]}, N: {len(datasets[i])}')
        for i in range(len(datasets))
    ]
    ax.legend(handles=legend_elements, frameon=True, fontsize=9)
    
    plt.tight_layout()
    return fig, ax


def main():
    # Configuration - MODIFY THESE PARAMETERS FOR YOUR DATASETS
    # =========================================================
    
    # Dataset 1 configuration
    dataset1_path = "/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet"
    dataset1_group_col = "id_group_sky"
    dataset1_stellar_col = "log_stellar_mass"
    dataset1_halo_col = "mvir_hosthalo"
    dataset1_label = "SHARKS-r<19.65 Sim Values"
    
    # Dataset 2 configuration
    dataset2_path = '/Users/sp624AA/Downloads/gama3/gama4_nessie_groups.parquet'
    dataset2_group_col = "group_id"
    dataset2_stellar_col = "logmstar"
    dataset2_halo_col = "MassA"
    dataset2_label = "GAMA4-Nessie pipeline"
    
    # Dataset 3 configuration
    dataset3_path = '/Users/sp624AA/Downloads/mocks/sharks_nessie_groups.parquet'
    dataset3_group_col = "group_id"
    dataset3_stellar_col = "logmstar"
    dataset3_halo_col = "MassA"
    dataset3_label = "SHARKS-Nessie pipeline"
    
    # Plot configuration
    min_group_members = 5
    contour_levels = 2
    contour_alpha = 0.6
    figure_size = (10, 8)
    
    # =========================================================
    
    try:
        # Load and process each dataset
        print("Loading and processing datasets...")
        
        df1 = load_and_process_data(
            dataset1_path, 
            dataset1_group_col, 
            dataset1_stellar_col, 
            dataset1_halo_col,
            min_group_members
        )
        print(f"Dataset 1: {len(df1)} groups with ≥{min_group_members} members")
        
        df2 = load_and_process_data(
            dataset2_path,
            dataset2_group_col,
            dataset2_stellar_col,
            dataset2_halo_col,
            min_group_members
        )
        print(f"Dataset 2: {len(df2)} groups with ≥{min_group_members} members")
        
        df3 = load_and_process_data(
            dataset3_path,
            dataset3_group_col,
            dataset3_stellar_col,
            dataset3_halo_col,
            min_group_members
        )
        print(f"Dataset 3: {len(df3)} groups with ≥{min_group_members} members")
        
        # Create the original stellar mass contour plot
        print("\nCreating stellar mass contour plot...")
        fig1, ax1 = create_contour_plot(
            [df1, df2, df3],
            [dataset1_label, dataset2_label, dataset3_label],
            colors=['blue', 'red', 'green'],
            levels=contour_levels,
            alpha=contour_alpha,
            figsize=figure_size,
            y_column='stellar_mass',
            use_log_y=True
        )
        plt.savefig("halo_stellar_mass_contours.png", dpi=300, bbox_inches='tight')
        print("Plot saved as 'halo_stellar_mass_contours.png'")
        
        # Create the percentage in largest galaxy contour plot
        print("\nCreating percentage in largest galaxy contour plot...")
        fig2, ax2 = create_contour_plot(
            [df1, df2, df3],
            [dataset1_label, dataset2_label, dataset3_label],
            colors=['blue', 'red', 'green'],
            levels=contour_levels,
            alpha=contour_alpha,
            figsize=figure_size,
            y_column='pct_stellar_in_largest',
            use_log_y=False
        )
        plt.savefig("halo_pct_stellar_in_largest_contours.png", dpi=300, bbox_inches='tight')
        print("Plot saved as 'halo_pct_stellar_in_largest_contours.png'")
        
        # Create the percentage NOT in largest galaxy contour plot (log scale)
        print("\nCreating percentage NOT in largest galaxy contour plot (log scale)...")
        fig3, ax3 = create_contour_plot(
            [df1, df2, df3],
            [dataset1_label, dataset2_label, dataset3_label],
            colors=['blue', 'red', 'green'],
            levels=contour_levels,
            alpha=contour_alpha,
            figsize=figure_size,
            y_column='pct_stellar_not_in_largest',
            use_log_y=False
        )
        plt.savefig("halo_pct_stellar_not_in_largest_log_contours.png", dpi=300, bbox_inches='tight')
        print("Plot saved as 'halo_pct_stellar_not_in_largest_log_contours.png'")
        
        # Display all plots
        plt.show()
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("=" * 50)
        for df, label in zip([df1, df2, df3], [dataset1_label, dataset2_label, dataset3_label]):
            print(f"\n{label}:")
            print(f"  Stellar Mass - Min: {df['stellar_mass'].min():.2e}, "
                  f"Max: {df['stellar_mass'].max():.2e}, "
                  f"Median: {df['stellar_mass'].median():.2e}")
            print(f"  Halo Mass    - Min: {df['halo_mass'].min():.2e}, "
                  f"Max: {df['halo_mass'].max():.2e}, "
                  f"Median: {df['halo_mass'].median():.2e}")
            print(f"  % in Largest - Min: {df['pct_stellar_in_largest'].min():.1f}%, "
                  f"Max: {df['pct_stellar_in_largest'].max():.1f}%, "
                  f"Median: {df['pct_stellar_in_largest'].median():.1f}%")
            print(f"  % NOT in Largest - Min: {df['pct_stellar_not_in_largest'].min():.1f}%, "
                  f"Max: {df['pct_stellar_not_in_largest'].max():.1f}%, "
                  f"Median: {df['pct_stellar_not_in_largest'].median():.1f}%")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please update the file paths in the configuration section.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())