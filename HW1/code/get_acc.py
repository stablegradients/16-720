import os

def get_highest_accuracy(directory):
    highest_accuracy = 0.0
    highest_file = ""

    for root, _, files in os.walk(directory):
        for file in files:
            if file == "accuracy.txt":
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        accuracy = float(f.read().strip())
                        if accuracy > highest_accuracy:
                            highest_accuracy = accuracy
                            highest_file = file_path
                    except ValueError:
                        print(f"Could not convert the content of {file_path} to float.")

    return highest_accuracy, highest_file

if __name__ == "__main__":
    directory = "ablation_results"
    highest_accuracy, highest_file = get_highest_accuracy(directory)
    print(f"Highest Accuracy: {highest_accuracy} found in {highest_file}")