#!/bin/bash

# Define the ranges for the parameters
K_values=(50 100 150)
L_values=(2 3 4)
alpha_values=(36 81 144)
filter_scales_values=("1 2 4 8" "1 2 4 8 16" "1 2 4 8 16 32")

# Create a directory to store the results
mkdir -p ablation_results

# Loop over each combination of parameters
for K in "${K_values[@]}"; do
    for L in "${L_values[@]}"; do
        for alpha in "${alpha_values[@]}"; do
            for filter_scales in "${filter_scales_values[@]}"; do
                echo "Running ablation with K=$K, L=$L, alpha=$alpha, filter_scales=$filter_scales"
                
                # Save the results
                result_dir="./ablation_results/K_${K}_L_${L}_alpha_${alpha}_filter_scales_${filter_scales// /_}"
                mkdir -p $result_dir
                
                # Check if the result_dir directory exists and is empty
                if [ ! -d "$result_dir" ] || [ -z "$(ls -A $result_dir)" ]; then
                    # Run the Python script with the current set of parameters
                    python ablation.py --K $K --L $L --alpha $alpha --filter-scales $filter_scales --out-dir $result_dir
                else
                    echo "Skipping: $result_dir already exists and is not empty."
                fi
            done
        done
    done
done

echo "Ablation study completed."