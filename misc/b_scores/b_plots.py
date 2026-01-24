import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Data from the threshold files (excluding B=1 and B=5)
data = {
    'B_threshold': [7, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15],
    'S_tot': [0.18855758417976745, 0.21367936866789938, 0.21352748592856094, 
              0.21763639726440054, 0.2162315641710467, 0.21849291041486116,
              0.22715462058577332, 0.22793867171003923, 0.21588215476044437,
              0.21104274472104922, 0.210590322276587, 0.20125799265063088],
    'E_tot': [0.2957818248364034, 0.34997354697301791, 0.34845121283428793,
              0.3576737420577501, 0.36023082827763303, 0.36602547116025469,
              0.38126405924174506, 0.38312272291867971, 0.36447381851587224,
              0.3579034219211395, 0.35795069439829885, 0.33977248788708658],
    'Q_tot': [0.6374887445638638, 0.6105586279764554, 0.6127901928988483,
              0.6084774241807801, 0.6002583543582648, 0.596933622466945,
              0.5957934273624863, 0.5949494980970385, 0.5923118309005316,
              0.5896639478556045, 0.5883221504307489, 0.592331633153044],
    'E_asg': [0.41395793499043976, 0.48091603053435117, 0.4865470852017937,
              0.4994246260069045, 0.5076741440377804, 0.5194647201946472,
              0.5360696517412935, 0.5424588086185045, 0.5335051546391752,
              0.5342105263157895, 0.5395973154362416, 0.5404255319148936],
    'E_fid': [0.7145214521452146, 0.7277227722772270, 0.7161716171617162,
              0.7161716171617162, 0.7095709570957096, 0.7046204620462046,
              0.7112211221122112, 0.7062706270627063, 0.6831683168316832,
              0.6699669966996700, 0.6633663366336634, 0.6287128712871287],
    'Q_asg': [0.7360097323600974, 0.7551819813938306, 0.7652901202300052,
              0.7711424995530127, 0.7688451208594449, 0.7726856095325390,
              0.7767957878901843, 0.7795739825028528, 0.7831277357739753,
              0.7879662064702246, 0.7931547619047619, 0.7995243757431629],
    'Q_fid': [0.8661417322834646, 0.8084920496243229, 0.8007292616226072,
              0.7890596414196853, 0.7807272727272727, 0.7725439882697948,
              0.7669884886743409, 0.7631725935579966, 0.7563412759415834,
              0.7483365949119374, 0.7417495029821074, 0.7408550022036139]
}

# Convert to DataFrame for easier handling
df = pd.DataFrame(data)

# Set up the plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Define a formatter for 3 significant figures
def three_sig_figs(x, pos):
    if x == 0:
        return '0'
    return f'{x:.3g}'

formatter = FuncFormatter(three_sig_figs)

# Color scheme for consistency
colors = {
    'S_tot': '#3b82f6',
    'E_tot': '#059669', 
    'Q_tot': '#7c3aed',
    'E_asg': '#ea580c',
    'E_fid': '#0891b2',
    'Q_asg': '#be185d',
    'Q_fid': '#0f766e'
}

def create_individual_plot(metric, title, color):
    """Create individual plot for a single metric"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'B Threshold vs {title}', fontsize=16, fontweight='bold')
    
    # Line plot
    ax1.plot(df['B_threshold'], df[metric], 'o-', color=color, linewidth=2, markersize=6)
    ax1.set_xlabel('B Threshold')
    ax1.set_ylabel(f'{title}')
    ax1.set_title('Line Plot')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(formatter)
    
    # Scatter plot
    ax2.scatter(df['B_threshold'], df[metric], color='#dc2626', s=60, alpha=0.7, edgecolors='#991b1b')
    ax2.set_xlabel('B Threshold')
    ax2.set_ylabel(f'{title}')
    ax2.set_title('Scatter Plot')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(formatter)
    
    # Add statistics
    max_idx = df[metric].idxmax()
    min_idx = df[metric].idxmin()
    
    stats_text = f"""Statistics:
Max: B={df.loc[max_idx, 'B_threshold']}, {title}={df.loc[max_idx, metric]:.6f}
Min: B={df.loc[min_idx, 'B_threshold']}, {title}={df.loc[min_idx, metric]:.6f}
Range: {df[metric].min():.6f} to {df[metric].max():.6f}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_overlay_plot(metric1, metric2, title1, title2, color1, color2):
    """Create overlay plot for two metrics"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot both metrics
    line1 = ax.plot(df['B_threshold'], df[metric1], 'o-', color=color1, 
                    linewidth=2, markersize=6, label=title1)
    line2 = ax.plot(df['B_threshold'], df[metric2], 's-', color=color2, 
                    linewidth=2, markersize=6, label=title2)
    
    ax.set_xlabel('B Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'B Threshold vs {title1} and {title2}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(formatter)
    
    # Add peak annotations
    max1_idx = df[metric1].idxmax()
    max2_idx = df[metric2].idxmax()
    
    ax.annotate(f'Peak {title1}\nB={df.loc[max1_idx, "B_threshold"]}\n{title1}={df.loc[max1_idx, metric1]:.3f}',
                xy=(df.loc[max1_idx, 'B_threshold'], df.loc[max1_idx, metric1]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color1, alpha=0.3),
                arrowprops=dict(arrowstyle='->', color=color1))
    
    ax.annotate(f'Peak {title2}\nB={df.loc[max2_idx, "B_threshold"]}\n{title2}={df.loc[max2_idx, metric2]:.3f}',
                xy=(df.loc[max2_idx, 'B_threshold'], df.loc[max2_idx, metric2]),
                xytext=(-10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color2, alpha=0.3),
                arrowprops=dict(arrowstyle='->', color=color2))
    
    plt.tight_layout()
    return fig

def create_all_metrics_overview():
    """Create overview plot with all metrics"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('B Threshold Analysis - All Metrics Overview', fontsize=16, fontweight='bold')
    
    metrics = ['S_tot', 'E_tot', 'Q_tot', 'E_asg', 'E_fid', 'Q_asg', 'Q_fid']
    titles = ['S_tot Score', 'E_tot Score', 'Q_tot Score', 'E_asg Score', 
              'E_fid Score', 'Q_asg Score', 'Q_fid Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        ax.plot(df['B_threshold'], df[metric], 'o-', color=colors[metric], 
                linewidth=2, markersize=4)
        ax.set_xlabel('B Threshold')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(formatter)
    
    # Hide the last subplot (since we have 7 metrics in 2x4 grid)
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    # Create individual plots
    print("Creating individual metric plots...")
    
    # S_tot
    fig_s = create_individual_plot('S_tot', 'S_tot Score', colors['S_tot'])
    fig_s.savefig('b_threshold_vs_s_tot.png', dpi=300, bbox_inches='tight')
    
    # E_tot  
    fig_e = create_individual_plot('E_tot', 'E_tot Score', colors['E_tot'])
    fig_e.savefig('b_threshold_vs_e_tot.png', dpi=300, bbox_inches='tight')
    
    # Q_tot
    fig_q = create_individual_plot('Q_tot', 'Q_tot Score', colors['Q_tot'])
    fig_q.savefig('b_threshold_vs_q_tot.png', dpi=300, bbox_inches='tight')
    
    # E_asg
    fig_ea = create_individual_plot('E_asg', 'E_asg Score', colors['E_asg'])
    fig_ea.savefig('b_threshold_vs_e_asg.png', dpi=300, bbox_inches='tight')
    
    # E_fid
    fig_ef = create_individual_plot('E_fid', 'E_fid Score', colors['E_fid'])
    fig_ef.savefig('b_threshold_vs_e_fid.png', dpi=300, bbox_inches='tight')
    
    # Q_asg
    fig_qa = create_individual_plot('Q_asg', 'Q_asg Score', colors['Q_asg'])
    fig_qa.savefig('b_threshold_vs_q_asg.png', dpi=300, bbox_inches='tight')
    
    # Q_fid
    fig_qf = create_individual_plot('Q_fid', 'Q_fid Score', colors['Q_fid'])
    fig_qf.savefig('b_threshold_vs_q_fid.png', dpi=300, bbox_inches='tight')
    
    # Create overlay plots as requested
    print("Creating overlay plots...")
    
    # E_fid and E_asg overlay
    fig_e_overlay = create_overlay_plot('E_fid', 'E_asg', 'E_fid Score', 'E_asg Score', 
                                       colors['E_fid'], colors['E_asg'])
    fig_e_overlay.savefig('b_threshold_vs_e_fid_e_asg_overlay.png', dpi=300, bbox_inches='tight')
    
    # Q_fid and Q_asg overlay
    fig_q_overlay = create_overlay_plot('Q_fid', 'Q_asg', 'Q_fid Score', 'Q_asg Score', 
                                       colors['Q_fid'], colors['Q_asg'])
    fig_q_overlay.savefig('b_threshold_vs_q_fid_q_asg_overlay.png', dpi=300, bbox_inches='tight')
    
    # Create overview plot
    print("Creating overview plot...")
    fig_overview = create_all_metrics_overview()
    fig_overview.savefig('b_threshold_analysis_overview.png', dpi=300, bbox_inches='tight')
    
    # Display all plots
    plt.show()
    
    print("All plots created and saved!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for metric in ['S_tot', 'E_tot', 'Q_tot', 'E_asg', 'E_fid', 'Q_asg', 'Q_fid']:
        max_idx = df[metric].idxmax()
        min_idx = df[metric].idxmin()
        print(f"\n{metric}:")
        print(f"  Peak: B={df.loc[max_idx, 'B_threshold']}, Value={df.loc[max_idx, metric]:.6f}")
        print(f"  Min:  B={df.loc[min_idx, 'B_threshold']}, Value={df.loc[min_idx, metric]:.6f}")
        print(f"  Range: {df[metric].min():.6f} to {df[metric].max():.6f}")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("• Performance metrics (S_tot, E_tot, E_asg) peak around B=12.5")
    print("• Q_asg increases monotonically with B threshold")
    print("• Fidelity metrics (E_fid, Q_fid) decrease with higher B values")
    print("• Q_tot shows declining trend, opposite to performance metrics")
    print("• Clear trade-offs between performance and fidelity/quality metrics")