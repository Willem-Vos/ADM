import os
import json

def load_model_results(directory="Results_Test"):
    """
    Load all JSON files in the specified directory and organize results by model.

    Args:
        directory (str): Directory containing JSON files with model results.

    Returns:
        dict: Dictionary where keys are model names (file names without extensions)
              and values are the parsed JSON data.
    """
    model_results = {}

    # Iterate through all JSON files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):  # Only process JSON files
            model_name = file_name.replace(".json", "")  # Extract model name
            file_path = os.path.join(directory, file_name)

            # Load JSON data
            with open(file_path, "r") as file:
                model_results[model_name] = json.load(file)

    return model_results

def display_model_results(results):
    """
    Display summary of model results for analysis.

    Args:
        results (dict): Dictionary of model results.
    """
    for model, data in results.items():
        print(f"\nModel: {model}")
        print(f"{'-' * 40}")
        for instance, metrics in data.items():
            print(f"Instance: {instance}")
            for metric, value in metrics.items():
                # print(f"  {metric}: {value}")
                print(f"  {metric}")
                for key, value in value.items():
                    print(f"\t{key}: {value}")
        print(f"{'-' * 40}")

if __name__ == "__main__":
    # Specify the directory containing the JSON files
    results_dir = "Results_Test"

    # Load model results
    model_results = load_model_results(results_dir)

    # Display results
    display_model_results(model_results)
