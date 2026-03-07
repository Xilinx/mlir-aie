#!/usr/bin/env python3
"""
Plot Conv3D spatial parallelism scaling results
"""
import matplotlib.pyplot as plt
import numpy as np

# Results from test runs
data = {
    '16x16': {
        1: 818.3,
        2: 1083.9,
        4: 932.0,
    },
    '32x32': {
        1: 2615.3,
        2: 1667.2,
        4: 1362.6,
    },
    '64x64': {
        4: 2544.9,  # Only 4-core works due to memory
    }
}

# Create figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Absolute Performance (Lower is Better)
for volume, results in data.items():
    cores = list(results.keys())
    times = list(results.values())
    ax1.plot(cores, times, 'o-', label=volume, linewidth=2, markersize=8)

ax1.set_xlabel('Number of Cores', fontsize=12)
ax1.set_ylabel('NPU Time (µs)', fontsize=12)
ax1.set_title('Conv3D Performance vs Core Count', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks([1, 2, 4])

# Plot 2: Speedup (relative to 1-core baseline)
for volume, results in data.items():
    if 1 in results:  # Only plot if we have 1-core baseline
        baseline = results[1]
        cores = list(results.keys())
        speedups = [baseline / results[c] for c in cores]
        ax2.plot(cores, speedups, 'o-', label=volume, linewidth=2, markersize=8)

# Add ideal scaling line
ideal_cores = [1, 2, 4]
ideal_speedup = [1, 2, 4]
ax2.plot(ideal_cores, ideal_speedup, 'k--', label='Ideal Linear', linewidth=1.5, alpha=0.5)

ax2.set_xlabel('Number of Cores', fontsize=12)
ax2.set_ylabel('Speedup (vs 1-core)', fontsize=12)
ax2.set_title('Scaling Efficiency', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks([1, 2, 4])
ax2.set_ylim(bottom=0)

# Plot 3: Parallel Efficiency
for volume, results in data.items():
    if 1 in results:
        baseline = results[1]
        cores = list(results.keys())
        speedups = [baseline / results[c] for c in cores]
        efficiencies = [(speedups[i] / cores[i]) * 100 for i in range(len(cores))]
        ax3.plot(cores, efficiencies, 'o-', label=volume, linewidth=2, markersize=8)

ax3.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Ideal (100%)')
ax3.axhline(y=80, color='g', linestyle=':', linewidth=1, alpha=0.5, label='Good (80%)')
ax3.axhline(y=50, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Fair (50%)')

ax3.set_xlabel('Number of Cores', fontsize=12)
ax3.set_ylabel('Parallel Efficiency (%)', fontsize=12)
ax3.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks([1, 2, 4])
ax3.set_ylim([0, 120])

plt.tight_layout()
plt.savefig('build/spatial_scaling_plots.png', dpi=300, bbox_inches='tight')
print("Plot saved to: build/spatial_scaling_plots.png")

# Print summary statistics
print("\nSummary Statistics:")
print("=" * 60)
for volume, results in data.items():
    print(f"\n{volume} volume:")
    if 1 in results:
        baseline = results[1]
        for cores, time in results.items():
            speedup = baseline / time
            efficiency = (speedup / cores) * 100
            print(f"  {cores} core(s): {time:.1f}µs, {speedup:.2f}x speedup, {efficiency:.1f}% efficiency")
    else:
        for cores, time in results.items():
            print(f"  {cores} core(s): {time:.1f}µs (no baseline for comparison)")

print("\nKey Insights:")
print("-" * 60)
print("1. 16x16: Single-core is fastest (multi-core overhead dominates)")
print("2. 32x32: Best scaling at 2 cores (78% efficiency, 1.56x speedup)")
print("3. 64x64: Requires 4 cores minimum due to memory constraints")
print("\nOptimal Configuration: 32x32 volume with 2 cores")
