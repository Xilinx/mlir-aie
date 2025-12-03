#!/usr/bin/env python3
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style and font
# sns.set_theme(style='whitegrid')
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times']
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Times'
plt.style.use('dark_background')

def read_benchmark_results(csv_file):
    """Read benchmark results from CSV with format: variant,iteration,time_us"""
    data = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)  # DictReader automatically handles headers
        for row in reader:
            variant = row['variant']
            time_us = float(row['time_us'])
            
            if variant not in data:
                data[variant] = []
            data[variant].append(time_us)
    
    return data

def compute_statistics(data):
    """Compute mean, median, std, min, max for each variant"""
    stats = {}
    for variant, times in data.items():
        times_array = np.array(times)
        stats[variant] = {
            'mean': np.mean(times_array),
            'median': np.median(times_array),
            'std': np.std(times_array),
            'min': np.min(times_array),
            'max': np.max(times_array),
            'q1': np.percentile(times_array, 25),
            'q3': np.percentile(times_array, 75),
            'times': times_array
        }
    return stats

def sort_variants(variants):
    """Sort variants in the desired order for presentation"""
    # Define the desired order
    order = [
        'separate_xclbins',
        'runlist',
        'fused_transactions_loadpdi',
        'fused_write32s_reset_always',
        'fused_write32s_reset_ifused',
        'fused_write32s_reset_ifchanged',
        'fused_write32s_reset_ifchangedfinegrained',
        'fused_write32s_reset_never'
    ]
    
    # Sort variants according to the order, putting any unknown variants at the end
    sorted_variants = []
    for v in order:
        if v in variants:
            sorted_variants.append(v)
    
    # Add any variants not in the order list
    for v in variants:
        if v not in sorted_variants:
            sorted_variants.append(v)
    
    return sorted_variants

def plot_results(stats, output_file, all_variants=None, variants_to_show=None):
    """Create bar chart with box-and-whisker overlay
    
    Args:
        stats: Dictionary of statistics for all variants
        output_file: Output filename
        all_variants: Ordered list of all variants (for consistent positioning)
        variants_to_show: List of variants to show (None = all)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if all_variants is None:
        all_variants = list(stats.keys())
    
    total_variants = len(all_variants)
    
    # Determine which variants to display
    if variants_to_show is None:
        variants_to_show = all_variants
    else:
        # Filter to only include variants that exist in stats
        variants_to_show = [v for v in variants_to_show if v in stats]
    
    num_shown = len(variants_to_show)
    means = [stats[v]['mean'] for v in variants_to_show]
    
    # Color scheme - use seaborn palette for ALL variants (consistent colors)
    all_colors = sns.color_palette('flare', n_colors=total_variants)
    color_map = {v: all_colors[i] for i, v in enumerate(all_variants)}
    colors = [color_map[v] for v in variants_to_show]
    
    # Create bar chart for means - use full width for x-axis
    x_pos = np.arange(total_variants)
    shown_positions = [all_variants.index(v) for v in variants_to_show]
    
    bars = ax.bar(shown_positions, means, color=colors, label='Mean', width=0.6)
    
    # Overlay box-and-whisker plots
    bp_data = [stats[v]['times'] for v in variants_to_show]
    bp = ax.boxplot(bp_data, positions=shown_positions, widths=0.4,
                    medianprops=dict(color='white'),
                    flierprops=dict(marker='.'))
    
    ax.set_ylabel('Time per Iteration (μs)', fontsize=12)
    ax.set_title('SwiGLU Performance Comparison\n(4 Reconfigurations, Embedding Dim=2048, Hidden Dim=8192)', 
                 fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_variants, rotation=45, ha='right')
    ax.set_xlim(-0.5, total_variants - 0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean value labels on bars
    for i, (pos, mean) in enumerate(zip(shown_positions, means)):
        ax.text(pos, mean + max(means) * 0.02,
                f'{mean:.0f} μs',
                ha='center', va='bottom', fontsize=10)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}", file=sys.stderr)

def print_statistics(stats, all_variants=None):
    """Print detailed statistics to stderr"""
    if all_variants is None:
        all_variants = list(stats.keys())
    
    print("\n" + "="*70, file=sys.stderr)
    print("PERFORMANCE STATISTICS", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(file=sys.stderr)
    
    print(f"{'Variant':<40} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}", file=sys.stderr)
    print("-"*100, file=sys.stderr)
    
    for variant in all_variants:
        if variant in stats:
            s = stats[variant]
            print(f"{variant:<40} {s['mean']:>8.2f}   {s['median']:>8.2f}   "
                  f"{s['std']:>8.2f}   {s['min']:>8.2f}   {s['max']:>8.2f}", file=sys.stderr)
    
    print(file=sys.stderr)
    print("="*70, file=sys.stderr)
    print("RELATIVE PERFORMANCE", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(file=sys.stderr)
    
    baseline_variant = all_variants[0] if all_variants[0] in stats else list(stats.keys())[0]
    baseline_mean = stats[baseline_variant]['mean']
    print(f"{'Variant':<40} {'Mean (μs)':<12} {'Relative to ' + baseline_variant:<20}", file=sys.stderr)
    print("-"*100, file=sys.stderr)
    
    for variant in all_variants:
        if variant in stats:
            mean = stats[variant]['mean']
            relative = mean / baseline_mean
            print(f"{variant:<40} {mean:>10.2f}   {relative:>8.2f}x", file=sys.stderr)
    
    print(file=sys.stderr)
    print("="*70, file=sys.stderr)
    print("COMPARISON ANALYSIS", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(file=sys.stderr)
    
    variants_in_stats = [v for v in all_variants if v in stats]
    fastest_variant = min(variants_in_stats, key=lambda v: stats[v]['mean'])
    fastest_mean = stats[fastest_variant]['mean']
    
    print(f"Fastest variant: {fastest_variant} ({fastest_mean:.2f} μs mean)", file=sys.stderr)
    print(file=sys.stderr)
    
    for variant in variants_in_stats:
        if variant != fastest_variant:
            mean = stats[variant]['mean']
            speedup = mean / fastest_mean
            improvement_pct = ((mean - fastest_mean) / mean) * 100
            print(f"{fastest_variant} is {speedup:.2f}x faster than {variant}", file=sys.stderr)
            print(f"  → {improvement_pct:.1f}% performance improvement", file=sys.stderr)
            print(file=sys.stderr)
    
    print("="*70, file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        csv_file = "benchmark_results.csv"
    else:
        csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found", file=sys.stderr)
        print("Please run the benchmark script first: ./run_benchmarks.sh", file=sys.stderr)
        sys.exit(1)
    
    data = read_benchmark_results(csv_file)
    
    if len(data) == 0:
        print("Error: No data found in CSV file", file=sys.stderr)
        sys.exit(1)
    
    stats = compute_statistics(data)
    all_variants = sort_variants(list(stats.keys()))
    
    # Generate full plot
    output_plot = "benchmark_results.png"
    plot_results(stats, output_plot, all_variants=all_variants)
    
    # Generate incremental plots for presentation
    print("\nGenerating incremental plots...", file=sys.stderr)
    for i in range(1, len(all_variants) + 1):
        variants_subset = all_variants[:i]
        incremental_output = f"benchmark_results_{i-1}.png"
        plot_results(stats, incremental_output, all_variants=all_variants, variants_to_show=variants_subset)
        print(f"  Created {incremental_output} with {i} variant(s)", file=sys.stderr)
    
    print_statistics(stats, all_variants)

if __name__ == "__main__":
    main()
