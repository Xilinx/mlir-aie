#!/usr/bin/env python3
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def plot_results(stats, output_file):
    """Create bar chart with box-and-whisker overlay"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    variants = list(stats.keys())
    means = [stats[v]['mean'] for v in variants]
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Create bar chart for means
    x_pos = np.arange(len(variants))
    bars = ax.bar(x_pos, means, color=colors, alpha=0.7, label='Mean', width=0.6)
    
    # Overlay box-and-whisker plots
    bp_data = [stats[v]['times'] for v in variants]
    bp = ax.boxplot(bp_data, positions=x_pos, widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.8),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, 
                                   markeredgecolor='black', alpha=0.5))
    
    ax.set_ylabel('Time per Iteration (μs)', fontsize=12)
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_title('SwiGLU Performance Comparison\n(Bar=Mean, Box=Distribution)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variants)
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f} μs',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.7, label='Mean (bar)'),
        Patch(facecolor='white', edgecolor='black', label='Box: 25th-75th percentile'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Median'),
        plt.Line2D([0], [0], color='black', linewidth=1.5, label='Min/Max (whiskers)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}", file=sys.stderr)

def print_statistics(stats):
    """Print detailed statistics to stderr"""
    print("\n" + "="*70, file=sys.stderr)
    print("PERFORMANCE STATISTICS", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(file=sys.stderr)
    
    print(f"{'Variant':<20} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}", file=sys.stderr)
    print("-"*70, file=sys.stderr)
    
    variants = list(stats.keys())
    for variant in variants:
        s = stats[variant]
        print(f"{variant:<20} {s['mean']:>8.2f}   {s['median']:>8.2f}   "
              f"{s['std']:>8.2f}   {s['min']:>8.2f}   {s['max']:>8.2f}", file=sys.stderr)
    
    print(file=sys.stderr)
    print("="*70, file=sys.stderr)
    print("RELATIVE PERFORMANCE", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(file=sys.stderr)
    
    baseline_mean = stats[variants[0]]['mean']
    print(f"{'Variant':<20} {'Mean (μs)':<12} {'Relative to ' + variants[0]:<20}", file=sys.stderr)
    print("-"*70, file=sys.stderr)
    
    for variant in variants:
        mean = stats[variant]['mean']
        relative = mean / baseline_mean
        print(f"{variant:<20} {mean:>10.2f}   {relative:>8.2f}x", file=sys.stderr)
    
    print(file=sys.stderr)
    print("="*70, file=sys.stderr)
    print("COMPARISON ANALYSIS", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(file=sys.stderr)
    
    fastest_variant = min(variants, key=lambda v: stats[v]['mean'])
    fastest_mean = stats[fastest_variant]['mean']
    
    print(f"Fastest variant: {fastest_variant} ({fastest_mean:.2f} μs mean)", file=sys.stderr)
    print(file=sys.stderr)
    
    for variant in variants:
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
    
    output_plot = "benchmark_results.png"
    plot_results(stats, output_plot)
    
    print_statistics(stats)

if __name__ == "__main__":
    main()
