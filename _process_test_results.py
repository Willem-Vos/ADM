import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        # for instance, metrics in data.items():
        #     print(f"Instance: {instance}")
            # for metric, value in metrics.items():
            #     # print(f"  {metric}: {value}")
            #     print(f"  {metric}")
                # for key, value in value.items():
                #     print(f"\t{key}: {value}")
        print(f"{'-' * 40}")


def process_results_for_comparison(results):
    # List to store flattened data for KPIs and robustness
    data = []
    # Dictionary to store accumulated rewards for each model and test
    rewards_data = []

    # Iterate through each model and its results
    for model_name, tests in results.items():
        for test_name, metrics in tests.items():
            # Extract KPIs and robustness data
            kpi_data = metrics.get('KPIS', {})
            robustness_data = metrics.get('robustness', {})

            # Combine all metrics into a single dictionary
            combined_data = {**kpi_data, **robustness_data}

            # Handle accumulated rewards separately
            accumulated_rewards = combined_data.pop('accumulated_rewards', None)
            if accumulated_rewards is not None:
                # Extract 'config' and 'policy' for the rewards data
                if model_name.count("_") > 1:
                    policy, config = model_name.split("_", 1)  # Split into two parts: policy and the rest as config
                else:
                    policy = model_name
                    config = "Unknown"

                rewards_data.append({
                    "config": config,
                    "policy": policy,
                    "test_name": test_name,
                    "accumulated_rewards": accumulated_rewards
                })

            # Extract 'config' and 'policy' for the main data
            if model_name.count("_") >= 3:
                parts = model_name.split("_")
                config = "_".join(parts[:3])  # First three parts form the config
                policy = parts[3]  # Fourth part is the policy
                combined_data['policy'] = policy
                combined_data['config'] = config
            else:
                combined_data['policy'] = "Unknown"
                combined_data['config'] = "Unknown"

            # Add test_name to the combined data
            combined_data['test_name'] = test_name

            # Append to the data list
            data.append(combined_data)

    # Convert the KPI and robustness data to a DataFrame
    df = pd.DataFrame(data)

    # Convert the rewards data to a DataFrame
    rewards_df = pd.DataFrame(rewards_data)

    # Rearrange columns for clarity in the main DataFrame
    df = df[['config', 'policy', 'test_name'] + [col for col in df.columns if col not in ['config', 'policy', 'test_name']]]
    print(f'{100 * '#'}')
    print(df.head())
    return df, rewards_df

def plot_metric_comparison(df, metric, title=None):
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    sns.kdeplot(data=df, x=metric, hue='model_name', fill=True, common_norm=False, alpha=0.5)
    plt.title(title if title else f"Comparison of {metric} Across Models")
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.legend(title="Model")
    plt.show()

# Function to plot a scatter plot for two combined features
def plot_combined_features(df, feature_x, feature_y, title=None):
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    sns.scatterplot(data=df, x=feature_x, y=feature_y, hue='model_name', alpha=0.7, s=100)
    plt.title(title if title else f"Comparison of {feature_x} vs {feature_y} Across Models")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(title="Model")
    plt.show()

# Function to plot a pairplot of selected features
def plot_pairwise_comparisons(df, features, title=None):
    sns.set_style("darkgrid")
    pairplot = sns.pairplot(df, vars=features, hue="policy", diag_kind="kde", corner=True, palette="tab10")
    pairplot.fig.suptitle(title if title else "Pairwise Feature Comparisons", y=1.02)
    plt.show()

def plot_kpi_comparison(df, kpi_column, configs, policies, title="KPI Comparison by Configurations and Policies"):
    # Filter the DataFrame for the specified configs and policies
    filtered_df = df[df['config'].isin(configs) & df['policy'].isin(policies)]

    # Create the plot
    plt.figure(figsize=(8, 6))  # Adjusted figure size for better visibility
    sns.set_theme(style="darkgrid")  # Cleaner grid style
    palette = sns.color_palette("Set2")  # More visually distinct color palette
    ax = sns.boxplot(
        data=filtered_df,
        x='config',
        y=kpi_column,
        hue='policy',
        palette=palette,
        dodge=True,
        linewidth=1.2,  # Thicker lines for box edges
    )

    # Add jittered points for better data visualization
    sns.stripplot(
        data=filtered_df,
        x='config',
        y=kpi_column,
        hue='policy',
        palette=palette,
        dodge=True,
        size=3,  # Size of the points
        alpha=0.6,  # Transparency for points
        linewidth=0.5,  # Thinner outline for the points
        edgecolor="gray",  # Edge color for better contrast
    )

    # Customize the plot
    ax.set_title(title, fontsize=16, weight='bold', color='darkblue')  # Improved title styling
    ax.set_xlabel("Configurations", fontsize=13, weight='bold')
    ax.set_ylabel(kpi_column.replace("_", " ").title(), fontsize=13, weight='bold')
    ax.tick_params(axis='x', labelsize=10, rotation=30)  # Improved x-axis label rotation
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(title="Policy", fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))  # Legend placement outside the plot
    sns.despine()  # Removes top and right spines for a cleaner look
    plt.tight_layout()  # Adjust layout for proper spacing
    plt.show()

if __name__ == "__main__":
    configs = ['single_RS1_6x24', 'multi_RS1_6x24']
    policies = ['PROACTIVE', 'REACTIVE', 'MYOPIC']

    results_dir = "Results_Test"
    model_results = load_model_results(results_dir)

    df, rewards_df = process_results_for_comparison(model_results)
    display_model_results(model_results)
    # display_model_results(target_model_results)

    # Display the DataFrames
    print("KPI and Robustness Data:")
    print(df)
    print("\nAccumulated Rewards Data:")
    print(rewards_df)

    target_df = df[df['config'].isin(configs)]
    # jointplot?
    # plot_metric_comparison(target_df, 'objective_value', title="Objective Value Comparison")# Plot combined features
    # plot_combined_features(target_df, 'affected_flights', 'n_actions', title="Affected Flights vs Number of Actions")
    plot_pairwise_comparisons(target_df, ['objective_value', 'n_actions', 'affected_flights'], title="Pairwise Feature Analysis")

    plot_kpi_comparison(df, kpi_column='objective_value', configs=configs, policies=policies, title="Number of Actions by Configurations and Policies")
