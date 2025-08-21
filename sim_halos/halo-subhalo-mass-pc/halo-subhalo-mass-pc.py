import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm

# Read the parquet file
# Replace 'your_catalog.parquet' with your actual filename
df = pd.read_parquet("/Users/sp624AA/Downloads/mocks/v0.3.0/wide/waves_wide_gals.parquet")

df['log_stellar_mass'] = np.log10((df['mstars_disk'] + df['mstars_bulge'])/0.67)
#print(df['log_stellar_mass'])
mask = (df['log_stellar_mass'] > 8) & (df["total_ab_dust_Z_VISTA"] > -200) & (df['id_group_sky'] != -1) & (df['mvir_hosthalo'] > 10**11)
#print(mask)
df = df[mask]
# Display basic info about the dataset
# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# For each group (id_group_sky), we need:
# 1. The host halo mass (mvir_hosthalo) - should be the same for all galaxies in the group
# 2. The largest subhalo mass (max of mvir_subhalo) in that group

print(f"\nTotal number of galaxies: {len(df)}")
print(f"Number of unique groups (host halos): {df['id_group_sky'].nunique()}")

# Group by id_group_sky and get the required information
group_stats = df.groupby('id_group_sky').agg({
    'mvir_hosthalo': 'first',  # Host halo mass (should be same for all galaxies in group)
    'mvir_subhalo': 'max'      # Largest subhalo mass in the group
}).reset_index()

group_stats.columns = ['id_group_sky', 'host_halo_mass', 'largest_subhalo_mass']

# Calculate percentage of halo mass in biggest subhalo
group_stats['subhalo_fraction'] = (np.log10(group_stats['largest_subhalo_mass']) / 
                                 np.log10(group_stats['host_halo_mass'])) * 100

# Remove any invalid data
print(f"\nBefore cleaning: {len(group_stats)} groups")
#group_stats = group_stats.dropna()
#group_stats = group_stats[group_stats['host_halo_mass'] > 0]
#group_stats = group_stats[group_stats['largest_subhalo_mass'] > 0]
#group_stats = group_stats[group_stats['subhalo_fraction'] <= 100]  # Sanity check
print('Number of largest subhalos with mass greater than host halo mass:')
print((group_stats['largest_subhalo_mass'] > group_stats['host_halo_mass']).sum())

print(f"After cleaning: {len(group_stats)} groups")
print(f"Host halo mass range: {group_stats['host_halo_mass'].min():.2e} - {group_stats['host_halo_mass'].max():.2e}")
print(f"Largest subhalo mass range: {group_stats['largest_subhalo_mass'].min():.2e} - {group_stats['largest_subhalo_mass'].max():.2e}")
print(f"Subhalo fraction range: {group_stats['subhalo_fraction'].min():.1f}% - {group_stats['subhalo_fraction'].max():.1f}%")

# Check for consistency in host halo masses within groups (diagnostic)
host_consistency = df.groupby('id_group_sky')['mvir_hosthalo'].agg(['min', 'max', 'std']).reset_index()
inconsistent_groups = host_consistency[host_consistency['std'] > 0]
if len(inconsistent_groups) > 0:
    print(f"\nWarning: {len(inconsistent_groups)} groups have inconsistent host halo masses within the group")
    print("This might indicate an issue with the data structure")

# Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Density plot of host halo mass vs percentage of mass in biggest subhalo
x = np.log10(group_stats['host_halo_mass'])
y = group_stats['subhalo_fraction']

# Create 2D histogram for density plot with log-scale colorbar
h = ax1.hist2d(x, y, bins=50, cmap='viridis', alpha=0.8, norm=LogNorm())
ax1.set_xlabel('Log₁₀(Host Halo Mass [M☉])', fontsize=12)
ax1.set_ylabel('Percentage of log10Mass in Largest Subhalo (%)', fontsize=12)
ax1.set_title('Density Plot: Host Halo Mass vs Subhalo Mass Fraction', fontsize=14, pad=20)
ax1.grid(True, alpha=0.3)

# Add colorbar for density plot
cbar1 = plt.colorbar(h[3], ax=ax1)
cbar1.set_label('Number of Halos (log scale)', fontsize=11)

# Plot 2: Host halo mass vs mass of largest subhalo
x2 = np.log10(group_stats['host_halo_mass'])
y2 = np.log10(group_stats['largest_subhalo_mass'])

# Create scatter plot with density coloring and log-scale colorbar
h2 = ax2.hist2d(x2, y2, bins=50, cmap='plasma', alpha=0.8, norm=LogNorm())
ax2.set_xlabel('Log₁₀(Host Halo Mass [M☉])', fontsize=12)
ax2.set_ylabel('Log₁₀(Largest Subhalo Mass [M☉])', fontsize=12)
ax2.set_title('Host Halo Mass vs Largest Subhalo Mass', fontsize=14, pad=20)
ax2.grid(True, alpha=0.3)


# Add colorbar for second plot
cbar2 = plt.colorbar(h2[3], ax=ax2)
cbar2.set_label('Number of Halos (log scale)', fontsize=11)

plt.tight_layout()
plt.savefig('halo_mass_vs_subhalo_mass.png', dpi=300)
#plt.show()

# Print some summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Number of host halos analyzed: {len(group_stats)}")
print(f"Mean subhalo fraction: {group_stats['subhalo_fraction'].mean():.1f}%")
print(f"Median subhalo fraction: {group_stats['subhalo_fraction'].median():.1f}%")
print(f"Standard deviation: {group_stats['subhalo_fraction'].std():.1f}%")

print(f"\nCorrelation between log(host mass) and subhalo fraction: {np.corrcoef(np.log10(group_stats['host_halo_mass']), group_stats['subhalo_fraction'])[0,1]:.3f}")
print(f"Correlation between log(host mass) and log(subhalo mass): {np.corrcoef(np.log10(group_stats['host_halo_mass']), np.log10(group_stats['largest_subhalo_mass']))[0,1]:.3f}")

# Show distribution of galaxies per group
galaxies_per_group = df['id_group_sky'].value_counts()
print(f"\nGalaxies per group statistics:")
print(f"Mean: {galaxies_per_group.mean():.1f}")
print(f"Median: {galaxies_per_group.median():.1f}")
print(f"Min: {galaxies_per_group.min()}")
print(f"Max: {galaxies_per_group.max()}")


