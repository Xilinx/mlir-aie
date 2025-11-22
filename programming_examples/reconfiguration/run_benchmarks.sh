#!/bin/bash

OUTPUT_CSV="benchmark_results.csv"

# Write CSV header
echo "variant,iteration,time_us" > "$OUTPUT_CSV"

VARIANTS=("separate_xclbins" "runlist" "fused_transactions_loadpdi")

for variant in "${VARIANTS[@]}"; do
    echo "Running benchmark for $variant..." >&2
    
    pushd "swiglu/$variant" > /dev/null
    
    if [ ! -f "build/test" ]; then
        echo "Error: build/test not found in $variant. Please run 'make' first." >&2
        popd > /dev/null
        continue
    fi
    
    cd build
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
