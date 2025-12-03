#!/usr/bin/env python3
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style and font - match visualize_results.py
plt.style.use('dark_background')

# Configuration labels
CONFIG_LABELS = {
    '0_write32s_reset_core_dma_switch_lock': 'Additional Reset of Everything\n(Core+DMA+Switch+Lock)',
    '1_write32s_reset_core_dma_switch': 'Additional Core+DMA+Switch Reset',
    '2_write32s_reset_core_dma': 'Additional Core+DMA Reset',
    '3_write32s_reset_core': 'Additional Core Reset',
    '4_write32s_reset_nothing': 'No Additional Reset',
}

def read_csv_file(csv_file):
    """Read benchmark results from CSV with format: variant,iteration,time_us"""
    times = []
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                time_us = float(row[2])
                times.append(time_us)
    
    return np.array(times)

def compute_statistics(times):
    """Compute mean, median, std, min, max for timing data"""
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'q1': np.percentile(times, 25),
        'q3': np.percentile(times, 75),
        'times': times
    }

def plot_results(stats_dict, output_file):
    """Create bar chart with box-and-whisker overlay for ablation study"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by config number (0-4)
    configs = sorted(stats_dict.keys(), key=lambda x: int(x.split('_')[0]))
    labels = [CONFIG_LABELS[c] for c in configs]
    means = [stats_dict[c]['mean'] for c in configs]
    
    # Color scheme - use seaborn palette (match visualize_results.py)
    colors = sns.color_palette('flare', n_colors=len(configs))
    
    # Create bar chart for means
    x_pos = np.arange(len(configs))
    bars = ax.bar(x_pos, means, color=colors, alpha=1.0, label='Mean', width=0.6)
    
    # Overlay box-and-whisker plots
    bp_data = [stats_dict[c]['times'] for c in configs]
    bp = ax.boxplot(bp_data, positions=x_pos, widths=0.4,
                    flierprops=dict(marker='.', markersize=2))
    
    ax.set_ylabel('Reconfiguration Time per Iteration (μs)', fontsize=12)
    ax.set_xlabel('Reset Configuration', fontsize=12)
    ax.set_title('AIE Reconfiguration Performance Ablation Study\n(Write32s Method)', 
                 fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{mean:.1f} μs',
                ha='center', va='bottom', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=1.0, label='Mean (bar)'),
        Patch(facecolor='white', edgecolor='black', label='Box: 25th-75th percentile'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
        plt.Line2D([0], [0], color='black', linewidth=1.5, label='Min/Max (whiskers)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}", file=sys.stderr)

def print_statistics(stats_dict):
    """Print detailed statistics to stderr"""
    print("\n" + "="*90, file=sys.stderr)
    print("ABLATION STUDY STATISTICS", file=sys.stderr)
    print("="*90, file=sys.stderr)
    print(file=sys.stderr)
    
    # Sort by config number
    configs = sorted(stats_dict.keys(), key=lambda x: int(x.split('_')[0]))
    
    print(f"{'Configuration':<40} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}", file=sys.stderr)
    print("-"*90, file=sys.stderr)
    
    for config in configs:
        s = stats_dict[config]
        label = CONFIG_LABELS[config]
        print(f"{label:<40} {s['mean']:>8.2f}   {s['median']:>8.2f}   "
              f"{s['std']:>8.2f}   {s['min']:>8.2f}   {s['max']:>8.2f}", file=sys.stderr)
    
    print(file=sys.stderr)
    print("="*90, file=sys.stderr)
    print("RELATIVE PERFORMANCE", file=sys.stderr)
    print("="*90, file=sys.stderr)
    print(file=sys.stderr)
    
    baseline = configs[0]
    baseline_mean = stats_dict[baseline]['mean']
    print(f"{'Configuration':<40} {'Mean (μs)':<12} {'Relative to ' + CONFIG_LABELS[baseline][:20]:<30}", file=sys.stderr)
    print("-"*90, file=sys.stderr)
    
    for config in configs:
        mean = stats_dict[config]['mean']
        relative = mean / baseline_mean
        label = CONFIG_LABELS[config]
        print(f"{label:<40} {mean:>10.2f}   {relative:>8.2f}x", file=sys.stderr)
    
    print(file=sys.stderr)
    print("="*90, file=sys.stderr)
    print("COMPARISON ANALYSIS", file=sys.stderr)
    print("="*90, file=sys.stderr)
    print(file=sys.stderr)
    
    fastest_config = min(configs, key=lambda c: stats_dict[c]['mean'])
    fastest_mean = stats_dict[fastest_config]['mean']
    slowest_config = max(configs, key=lambda c: stats_dict[c]['mean'])
    slowest_mean = stats_dict[slowest_config]['mean']
    
    print(f"Fastest configuration: {CONFIG_LABELS[fastest_config]} ({fastest_mean:.2f} μs mean)", file=sys.stderr)
    print(f"Slowest configuration: {CONFIG_LABELS[slowest_config]} ({slowest_mean:.2f} μs mean)", file=sys.stderr)
    print(file=sys.stderr)
    
    speedup = slowest_mean / fastest_mean
    improvement_pct = ((slowest_mean - fastest_mean) / slowest_mean) * 100
    print(f"Maximum speedup: {speedup:.2f}x", file=sys.stderr)
    print(f"  → {improvement_pct:.1f}% performance improvement from slowest to fastest", file=sys.stderr)
    print(file=sys.stderr)
    
    # Analyze incremental improvements
    print("Incremental Analysis (from most to least reset):", file=sys.stderr)
    for i in range(len(configs) - 1):
        curr = configs[i]
        next_cfg = configs[i + 1]
        curr_mean = stats_dict[curr]['mean']
        next_mean = stats_dict[next_cfg]['mean']
        
        if next_mean < curr_mean:
            speedup = curr_mean / next_mean
            improvement = ((curr_mean - next_mean) / curr_mean) * 100
            print(f"  {CONFIG_LABELS[curr][:30]} → {CONFIG_LABELS[next_cfg][:30]}", file=sys.stderr)
            print(f"    {speedup:.3f}x faster ({improvement:.1f}% improvement)", file=sys.stderr)
        else:
            slowdown = next_mean / curr_mean
            degradation = ((next_mean - curr_mean) / curr_mean) * 100
            print(f"  {CONFIG_LABELS[curr][:30]} → {CONFIG_LABELS[next_cfg][:30]}", file=sys.stderr)
            print(f"    {slowdown:.3f}x slower ({degradation:.1f}% degradation)", file=sys.stderr)
    
    print(file=sys.stderr)
    print("="*90, file=sys.stderr)

def main():
    ablation_dir = Path(".")
    
    # Find all CSV files
    csv_files = sorted(ablation_dir.glob("*.csv"))
    
    if not csv_files:
        print("Error: No CSV files found in current directory", file=sys.stderr)
        sys.exit(1)
    
    stats_dict = {}
    
    for csv_file in csv_files:
        config_name = csv_file.stem  # e.g., "0_write32s_reset_core_dma_switch_lock"
        
        if config_name not in CONFIG_LABELS:
            print(f"Warning: Unknown config {config_name}, skipping", file=sys.stderr)
            continue
        
        times = read_csv_file(csv_file)
        
        if len(times) == 0:
            print(f"Warning: No data in {csv_file}, skipping", file=sys.stderr)
            continue
        
        stats_dict[config_name] = compute_statistics(times)
        print(f"Loaded {len(times)} samples from {csv_file.name}", file=sys.stderr)
    
    if len(stats_dict) == 0:
        print("Error: No valid data found", file=sys.stderr)
        sys.exit(1)
    
    output_plot = "ablation_results.png"
    plot_results(stats_dict, output_plot)
    
    print_statistics(stats_dict)

if __name__ == "__main__":
    main()
