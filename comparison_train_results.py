import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_model_results(directory="Results_Train"):
    """
    Load all JSON files in the specified directory and organize results by model.

    Args:
        directory (str): Directory containing JSON files with model results.

    Returns:
        dict: Dictionary where keys are model names (file names without extensions)
              and values are the parsed JSON data.
    """
    results = {}

    # Iterate through all JSON files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):  # Only process JSON files
            model_name = file_name.replace(".json", "")  # Extract model name
            file_path = os.path.join(directory, file_name)

            # Load JSON data
            with open(file_path, "r") as file:
                results[model_name] = json.load(file)

    return results

def display_model_results(results):
    """
    Display summary of model results for analysis.

    Args:
        results (dict): Dictionary of model results.
    """
    for model, data in results.items():
        print(f"\nModel: {model}")
        print(f"{'-' * 40}")
        for metric, value in data.items():
            # print(f"  {metric}: {value}")
            print(f"  {metric}")
        print(f"{'-' * 40}")

def plot_evolutions(results, evolution_type="objective_evolutions", title_suffix="Objective"):
    """
    Plot the specified evolution type (objective_evolutions or value_evolutions)
    for multiple models on the same plot.

    Args:
        results (dict): Dictionary containing results for multiple models.
        evolution_type (str): The type of evolution to plot (objective_evolutions or value_evolutions).
        title_suffix (str): Suffix for the plot title to distinguish between objective and value evolutions.
    """

    # Prepare a list to hold the data
    data_list = []

    for model_name, model_data in results.items():
        if evolution_type not in model_data:
            print(f"'{evolution_type}' not found in {model_name}. Skipping.")
            continue

        evolution_data = model_data[evolution_type]
        mean_evolution = {}

        # Calculate mean evolution for all instances
        for instance_id, instance_data in evolution_data.items():
            for iteration, value in instance_data.items():
                iteration = int(iteration)
                if iteration not in mean_evolution:
                    mean_evolution[iteration] = []
                mean_evolution[iteration].append(value)

        # Compute mean for each iteration
        mean_evolution = {iteration: sum(values) / len(values) for iteration, values in mean_evolution.items()}

        # Append the data to the list
        for iteration, mean_value in mean_evolution.items():
            data_list.append({"Iteration": iteration, "Value": mean_value, "Model": model_name})

    # Convert the list to a Pandas DataFrame
    df = pd.DataFrame(data_list)

    # Plot using Seaborn
    plt.figure(figsize=(6, 8))
    sns.lineplot(data=df, x="Iteration", y="Value", hue="Model", linewidth=2)

    # Customize the plot
    plt.xlabel("Iteration")
    plt.ylabel(f"{title_suffix} Evolution (Mean)")
    plt.title(f"Mean {title_suffix} Evolution for Different Models")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_dir = "Results_Train"
    results = load_model_results(results_dir)
    display_model_results(results)
    plot_evolutions(results, evolution_type="objective_evolutions", title_suffix="Objective")
    plot_evolutions(results, evolution_type="value_evolutions", title_suffix="Initial State Value")
    print(results)
