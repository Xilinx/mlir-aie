# utilities.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import time
import numpy as np


def do_bench(
    fn,
    barrier_fn=lambda: None,
    preamble_fn=lambda: None,
    n_warmup=25,
    n_repeat=100,
    quantiles=None,
    return_mode="mean",
    verbose=True,
):
    """
    Advanced benchmarking function for timing function execution.
    
    Args:
        fn: Function to benchmark (should be a callable)
        barrier_fn: Function to call between runs for synchronization (default: no-op)
        preamble_fn: Function to call before benchmarking starts (default: no-op)
        n_warmup: Number of warmup runs (default: 25)
        n_repeat: Number of timing runs (default: 100)
        quantiles: List of quantiles to compute (default: None)
        return_mode: What to return - "mean", "median", "min", "max", "all" (default: "mean")
        verbose: Whether to print detailed output (default: True)
    
    Returns:
        float or dict: Timing results in seconds based on return_mode
    """
    if quantiles is None:
        quantiles = [0.5, 0.95, 0.99]
    
    # Run preamble function
    preamble_fn()
    
    # Warmup runs
    if verbose:
        print(f"Warming up with {n_warmup} runs...")
    for _ in range(n_warmup):
        fn()
        barrier_fn()
    
    # Timing runs
    if verbose:
        print(f"Timing {n_repeat} runs...")
    times = []
    
    for i in range(n_repeat):
        start_time = time.time()
        fn()
        end_time = time.time()
        
        elapsed = end_time - start_time  # Keep in seconds for return values
        times.append(elapsed)
        
        barrier_fn()
        
        if verbose and i < 3:  # Show details for first few runs
            print(f"Run {i+1}: {elapsed*1000:.2f}ms")
    
    # Calculate statistics
    times_array = np.array(times)
    mean_time = np.mean(times_array)
    median_time = np.median(times_array)
    min_time = np.min(times_array)
    max_time = np.max(times_array)
    std_time = np.std(times_array)
    
    # Calculate quantiles
    quantile_results = {}
    for q in quantiles:
        quantile_results[f"q{int(q*100)}"] = np.quantile(times_array, q)
    
    # Prepare results
    results = {
        'times': times,
        'mean': mean_time,
        'median': median_time,
        'min': min_time,
        'max': max_time,
        'std': std_time,
        'n_warmup': n_warmup,
        'n_repeat': n_repeat,
        **quantile_results
    }
    
    # Print summary
    if verbose:
        print(f"\nBenchmark Results:")
        print(f"Mean time:   {mean_time*1000:.2f}ms")
        print(f"Median time: {median_time*1000:.2f}ms")
        print(f"Min time:    {min_time*1000:.2f}ms")
        print(f"Max time:    {max_time*1000:.2f}ms")
        print(f"Std dev:     {std_time*1000:.2f}ms")
        
        for q_name, q_value in quantile_results.items():
            print(f"{q_name}:        {q_value*1000:.2f}ms")
    
    # Return based on mode
    if return_mode == "mean":
        return mean_time
    elif return_mode == "median":
        return median_time
    elif return_mode == "min":
        return min_time
    elif return_mode == "max":
        return max_time
    elif return_mode == "all":
        return results
    else:
        raise ValueError(f"Invalid return_mode: {return_mode}")


