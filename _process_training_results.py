import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import warnings
from matplotlib.cm import tab10
warnings.filterwarnings("ignore")


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

def plot_evolutions(results, evolution_type="objective_evolutions", title_suffix="Objective", clip_value=-300):
    """
    Plot the specified evolution type (objective_evolutions or value_evolutions)
    for multiple models on the same plot.

    Args:
        results (dict): Dictionary containing results for multiple models.
        evolution_type (str): The type of evolution to plot (objective_evolutions or value_evolutions).
        title_suffix (str): Suffix for the plot title to distinguish between objective and value evolutions.
    """
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
        mean_evolution = {iteration: max(clip_value, sum(values) / len(values)) for iteration, values in mean_evolution.items()}

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
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.show()

def plot_evolutions_filtered(results, evolution_type, title_suffix):
    """
    Plot the specified evolution type (objective_evolutions or value_evolutions)
    for multiple models on the same plot, removing outliers only for visualization.

    Args:
        results (dict): Dictionary containing results for multiple models.
        evolution_type (str): The type of evolution to plot (objective_evolutions or value_evolutions).
        title_suffix (str): Suffix for the plot title to distinguish between objective and value evolutions.
    """
    filtered_data_list = []
    smoothed_data_list = []
    for model_name, model_data in results.items():
        if evolution_type not in model_data:
            print(f"'{evolution_type}' not found in {model_name}. Skipping.")
            continue

        evolution_data = model_data[evolution_type]
        mean_evolution = {}
        all_values = []

        # Calculate mean evolution for all instances
        for instance_id, instance_data in evolution_data.items():
            for iteration, value in instance_data.items():
                iteration = int(iteration)
                if iteration not in mean_evolution:
                    mean_evolution[iteration] = []
                mean_evolution[iteration].append(value)
                all_values.append(value)

        # Compute mean for each iteration
        mean_evolution = {iteration: sum(values) / len(values) for iteration, values in mean_evolution.items()}

        values = list(mean_evolution.values())
        # Remove outliers for plotting using IQR
        q1 = pd.Series(values).quantile(0.25)
        q3 = pd.Series(values).quantile(0.75)
        iqr = q3 - q1

        if evolution_type == 'value_evolutions':
            lower_bound = q1 - 1.5 * iqr
            upper_bound = 0

        if evolution_type == 'objective_evolutions':
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 1 * iqr

        filtered_mean_evolution = {}
        for iteration, mean_value in mean_evolution.items():
            if lower_bound <= mean_value <= upper_bound:
                filtered_mean_evolution[iteration] = mean_value

        # Append filtered data to the list for plotting
        for iteration, mean_value in filtered_mean_evolution.items():
            filtered_data_list.append({"Iteration": iteration, "Value": mean_value, "Model": model_name})

    df_filtered = pd.DataFrame(filtered_data_list)
    # Smoothing the data for each model
    smoothed_data_list = []
    for model_name in df_filtered['Model'].unique():
        model_data = df_filtered[df_filtered['Model'] == model_name]
        model_data = model_data.sort_values(by="Iteration")  # Ensure data is sorted by iteration
        smoothed_values = savgol_filter(model_data['Value'], window_length=50, polyorder=3)  # Adjust window_length as needed
        for i, value in enumerate(smoothed_values):
            smoothed_data_list.append({
                "Iteration": model_data.iloc[i]['Iteration'],
                "Value": value,
                "Model": model_name
            })

    # Convert smoothed data to DataFrame
    smoothed_df = pd.DataFrame(smoothed_data_list)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df_filtered, x="Iteration", y="Value", hue="Model", linewidth=0.5, alpha=0.3)
    sns.lineplot(data=smoothed_df, x="Iteration", y="Value", hue="Model", linewidth=2)

    # Customize the plot
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(f"{title_suffix} Evolution (Mean)", fontsize=12)
    plt.title(f"Mean {title_suffix} Evolution for Different Models (No Outliers)", fontsize=14)
    plt.grid(alpha=0.8)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter out duplicates in the legend
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Model", loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_evolutions_normalized(results, evolution_type, title_suffix):
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="darkgrid")

    # Store noisy and smoothed data for plotting
    noisy_data = []
    smoothed_data = []

    for model_name, model_data in results.items():
        if evolution_type not in model_data:
            print(f"'{evolution_type}' not found in {model_name}. Skipping.")
            continue

        evolution_data = model_data[evolution_type]
        mean_evolution = {}

        # Process data
        for instance_id, instance_data in evolution_data.items():
            for iteration, value in instance_data.items():
                iteration = int(iteration)
                if iteration not in mean_evolution:
                    mean_evolution[iteration] = []
                mean_evolution[iteration].append(value)

        # Compute mean for each iteration
        mean_evolution = {k: np.mean(v) for k, v in mean_evolution.items()}
        iterations = list(mean_evolution.keys())
        values = list(mean_evolution.values())

        values = list(mean_evolution.values())
        # Remove outliers for plotting using IQR
        q1 = pd.Series(values).quantile(0.25)
        q3 = pd.Series(values).quantile(0.75)
        iqr = q3 - q1

        if evolution_type == 'value_evolutions':
            lower_bound = q1 - 1.5 * iqr
            upper_bound = 0

        if evolution_type == 'objective_evolutions':
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 1 * iqr

        filtered_mean_evolution = {}
        for iteration, mean_value in mean_evolution.items():
            if lower_bound <= mean_value <= upper_bound:
                filtered_mean_evolution[iteration] = mean_value

        values = list(filtered_mean_evolution.values())
        smoothed_values = savgol_filter(values, window_length=201, polyorder=1)

        # Normalize using min and max of smoothed data
        min_val = min(smoothed_values)
        max_val = max(smoothed_values)
        normalized_smoothed_values = [(val - min_val) / (max_val - min_val) for val in smoothed_values]

        # Normalize noisy data using the same min and max
        normalized_noisy_values = [(val - min_val) / (max_val - min_val) for val in values]

        # Append data for plotting
        noisy_data.extend(
            {"Iteration": it, "Value": val, "Model": model_name}
            for it, val in zip(iterations, normalized_noisy_values)
        )
        smoothed_data.extend(
            {"Iteration": it, "Value": val, "Model": model_name}
            for it, val in zip(iterations, normalized_smoothed_values)
        )

    # Convert to DataFrames
    noisy_df = pd.DataFrame(noisy_data)
    smoothed_df = pd.DataFrame(smoothed_data)

    # Plot noisy and smoothed lines with Seaborn
    sns.lineplot(
        data=noisy_df,
        x="Iteration",
        y="Value",
        hue="Model",
        linewidth=0.5,
        alpha=0.3,
        legend=None,  # Avoid double legends
    )
    sns.lineplot(
        data=smoothed_df,
        x="Iteration",
        y="Value",
        hue="Model",
        linewidth=2,
        alpha=0.8,
    )

    # Customize the plot
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(f"{title_suffix} (Normalized)", fontsize=12)
    plt.title(f"Convergence for Different Scenarios (Normalized)", fontsize=14)
    plt.grid(alpha=0.8)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.ylim(-0.25, 1.25)
    # Clean up legend to remove duplicates and maintain consistency
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), title="Model", loc="best", fontsize=10)

    plt.tight_layout()
    plt.show()


def analyze_features(csv_file, model_name):
    # Load the data
    df = pd.read_csv(csv_file)
    print(model_name)
    # Filter the data by the model name if applicable
    # if model_name:
    #     df = df[df['folder'] == model_name]

    # Summary statistics
    print("Summary Statistics:")
    print(df.describe())
    print(df.head())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    print(df.columns)
    df = df.drop(['prev_action', 'folder'], axis=1)

    # Plotting
    # 1. Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # 2. Distribution of key features
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    # 3. Box plots for key features to detect outliers
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.show()

    # 4. Feature analysis over time (if time column exists)
    if 't' in df.columns:
        for col in numerical_columns:
            if col != 't':
                plt.figure(figsize=(10, 6))
                sns.lineplot(x='t', y=col, data=df)
                plt.title(f"Trend of {col} over Time")
                plt.xlabel("Time (t)")
                plt.ylabel(col)
                # plt.show()

    # Correlation with target variable (if any)
    if 'value' in df.columns:
        correlations = df.corr()['value'].sort_values(ascending=False)
        print("\nCorrelation with Target Variable (value):")
        print(correlations)

    # Return the filtered DataFrame for further analysis if needed
    return df


if __name__ == "__main__":
    results_dir = "Results_Train"
    results = load_model_results(results_dir)
    for model_name in results:
        csv_file = '_state_features_' + model_name + '.csv'
    analyze_features(csv_file, model_name)


    # results = results.pop('single_RS1_6x24_2')
    display_model_results(results)
    # plot_evolutions_filtered(results, evolution_type="objective_evolutions", title_suffix="Objective")
    # plot_evolutions_filtered(results, evolution_type="value_evolutions", title_suffix="Initial State Value")
    # plot_evolutions_normalized(results, evolution_type="objective_evolutions", title_suffix="Objective")
    # plot_evolutions_normalized(results, evolution_type="value_evolutions", title_suffix="Initial State Value")
