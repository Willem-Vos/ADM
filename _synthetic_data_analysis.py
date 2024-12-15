import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from generate_disruptions import *
# General
# Disruptions start times (std dev, and mean over al folders)
# Avg disrupted flights per schedule
# Utilization min, max, mean per instance (std dev, and mean over al folders)
# Disruptions start times (std dev, and mean over al folders)
# flight durations

# Before:
# Distribution of flights over the day

# After
# Distribution of flights over the day
# Utilization min, max, mean per instance (std dev, and mean over al folders)

if __name__ == "__main__":
    plot = False
    nr_folders = 1
    all_disruptions = load_disruptions("Disruptions_test")
    for instance in range(1, nr_folders + 1):
        folder = f'TEST{instance}'
        aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)
        disruptions = all_disruptions[instance]
        print(flight_data)




    if plot:
        # Use Seaborn style
        sns.set_style("darkgrid")  # Set the background and grid style
        sns.set_context("notebook", font_scale=1.2)  # Adjust font size for labels and titles

        # Simulated data for the plot
        timesteps = np.arange(0, 25)
        disruptions = [np.clip(np.random.uniform(0.3, 0.8, size=25), 0.05, 0.95) for _ in range(5)]
        upper_clip = np.full_like(timesteps, 0.95)
        lower_clip = np.full_like(timesteps, 0.05)

        # Plot the data
        plt.figure(figsize=(10, 6))

        # Add clipping lines
        plt.plot(timesteps, upper_clip, 'r--', label="Upper Clip (0.95)", alpha=0.8)
        plt.plot(timesteps, lower_clip, 'r--', label="Lower Clip (0.05)", alpha=0.8)

        # Plot disruptions
        for i, disruption in enumerate(disruptions):
            plt.plot(timesteps, disruption, label=f"Uncertain Disruption {i + 1} (starts at t={np.random.randint(18, 25)})")

        # Add labels, title, and legend
        plt.xlabel("xlabel")
        plt.ylabel("ylabel")
        plt.title("Title")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)  # Legend outside the plot
        plt.tight_layout()  # Adjust layout for readability
        plt.show()

        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Simulated data for the plot
        steps = np.linspace(0, 1e6, 100)  # Environment steps
        mean_rewards = {
            "Myopic": np.linspace(-5000, 15000, 100) + np.random.normal(0, 1000, 100),
            "Proactive": np.linspace(-5000, 15000, 100) + np.random.normal(0, 800, 100),
            "Reactive": np.linspace(-5000, 14000, 100) + np.random.normal(0, 1200, 100),
        }
        std_devs = {
            "Myopic": np.random.uniform(200, 1000, 100),
            "Proactive": np.random.uniform(200, 800, 100),
            "Reactive": np.random.uniform(200, 1200, 100),
        }

        # Set Seaborn style
        sns.set_style("darkgrid")

        # Plot each method with confidence intervals
        plt.figure(figsize=(10, 6))
        for label, mean in mean_rewards.items():
            std = std_devs[label]
            plt.plot(steps, mean, label=label)  # Line plot
            plt.fill_between(steps, mean - std, mean + std, alpha=0.3)  # Shaded region

        # Add labels, title, and legend
        plt.xlabel("Environment Steps (Frames)")
        plt.ylabel("Episode Reward")
        plt.title("Averaged Episode Rewards over 3 Seeds (DQN, 6ac-100-mixed-high)")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

