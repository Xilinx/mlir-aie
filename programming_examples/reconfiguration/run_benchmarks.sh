#!/bin/bash

OUTPUT_CSV="benchmark_results.csv"

# Write CSV header
echo "variant,iteration,time_us" > "$OUTPUT_CSV"

# Array of base variants and their build directories
declare -A VARIANTS
VARIANTS["separate_xclbins"]="build"
VARIANTS["runlist"]="build"
VARIANTS["fused_transactions_loadpdi"]="build"
VARIANTS["fused_write32s_reset_always"]="build_reset_always"
VARIANTS["fused_write32s_reset_ifused"]="build_reset_ifused"
VARIANTS["fused_write32s_reset_ifchanged"]="build_reset_ifchanged"
VARIANTS["fused_write32s_reset_ifchangedfinegrained"]="build_reset_ifchangedfinegrained"
VARIANTS["fused_write32s_reset_never"]="build_reset_never"

for variant in "${!VARIANTS[@]}"; do
    build_dir="${VARIANTS[$variant]}"
    
    echo "Running benchmark for $variant..." >&2
    
    # Determine the directory based on variant name
    if [[ $variant == fused_write32s_* ]]; then
        base_dir="swiglu/fused_transactions_write32s"
    else
        base_dir="swiglu/$variant"
    fi
    
    pushd "$base_dir" > /dev/null
    
    if [ ! -f "$build_dir/test" ]; then
        echo "Error: $build_dir/test not found in $variant. Please run 'make' first." >&2
        popd > /dev/null
        continue
    fi
    
    cd "$build_dir"
    # Redirect stdout to CSV, stderr stays on terminal
    ./test 1>> "../../../$OUTPUT_CSV" 
    
    if [ $? -eq 0 ]; then
        echo "  ✓ $variant completed successfully" >&2
    else
        echo "  ✗ $variant failed" >&2
    fi
    
    popd > /dev/null
done

echo "" >&2
echo "Benchmark results saved to: $OUTPUT_CSV" >&2
echo "Total data points: $(( $(wc -l < "$OUTPUT_CSV") - 1 ))" >&2
