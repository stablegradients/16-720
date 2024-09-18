import os
import matplotlib.pyplot as plt
import numpy as np

def read_accuracy(result_dir):
    """
    Reads the accuracy from the accuracy.txt file in the given directory.

    Parameters:
    result_dir (str): Path to the directory containing the accuracy.txt file.

    Returns:
    float: The accuracy value.
    """
    accuracy_file = os.path.join(result_dir, 'accuracy.txt')
    if os.path.exists(accuracy_file):
        with open(accuracy_file, 'r') as f:
            accuracy = float(f.readline().strip())
        return accuracy
    else:
        return None

def collect_accuracies(base_dir):
    """
    Collects the accuracies from all ablation result directories.

    Parameters:
    base_dir (str): Path to the base directory containing ablation result directories.

    Returns:
    dict: A dictionary with parameter combinations as keys and accuracies as values.
    """
    accuracies = {}
    for result_dir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, result_dir)
        if os.path.isdir(full_path):
            accuracy = read_accuracy(full_path)
            if accuracy is not None:
                accuracies[result_dir] = accuracy
    return accuracies

def plot_accuracies(accuracies):
    """
    Plots the accuracies using matplotlib.

    Parameters:
    accuracies (dict): A dictionary with parameter combinations as keys and accuracies as values.
    """
    # Sort the accuracies by parameter combinations
    sorted_accuracies = dict(sorted(accuracies.items()))
    # Extract keys and values
    keys = list(sorted_accuracies.keys())
    values = list(sorted_accuracies.values())

    # Determine the number of lines to plot
    num_lines = len(keys) // 9
    chunk_size = 9

    plt.figure(figsize=(12, 8))

    for i in range(num_lines):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        keys_chunk = keys[start_idx:end_idx]
        values_chunk = values[start_idx:end_idx]
        # Extract K and L values from the keys assuming the format 'K-<K value>,L-<L value>'
        # Extract the legend label from the first key in the chunk
        legend_label = keys_chunk[0].split(',')[0]
        
        # Format the keys for plotting
        formatted_keys = [
            ("alpha" + key.split("alpha")[-1])
            .replace("alpha_", "alpha=")
            .replace("_filter_", ", \n filter-")
            .replace("scales_", "scales=(")
            .replace("_", ",") + ")"
            for key in keys_chunk
        ]
        
        # Plot the values with the formatted keys
        plt.plot(formatted_keys, values_chunk, marker='o', label=legend_label.split("_alpha")[0].replace("_", "=").replace("=L", ",L"))

    plt.xlabel('Hyper-parameter groups', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title('Ablation Study Accuracies', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)  # Slant the x ticks at an angle and increase font size
    plt.yticks(ticks=np.arange(0.1, 0.65, 0.033333), fontsize=11)  # Increase y ticks resolution and font size
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()

# Example usage
base_dir = 'ablation_results'  # Path to the directory containing ablation results
accuracies = collect_accuracies(base_dir)
plot_accuracies(accuracies)