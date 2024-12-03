import numpy as np
import pandas as pd
from preprocessing import *
import os
import pickle
import random
from scipy.stats import norm

def set_seed(seed):
    """Set the seed for NumPy's random number generator."""
    np.random.seed(seed)

def sample_duration(fixed_duration):
    duration_min = 1  # minimum duration in hours
    duration_max = 5  # maximum duration in hours
    half_hour_increments = 0.5  # 30-minute increments

    # Create an array of possible durations: 1.0, 1.5, 2.0, ..., 5.0 hours
    possible_durations = np.arange(duration_min, duration_max + half_hour_increments, half_hour_increments)

    # Randomly choose a duration from the possible options
    duration = np.random.choice(possible_durations)
    if fixed_duration:
        return fixed_duration
    return duration

def sample_disruption_path(aircraft_ids, n_disrupted_ac, steps, periods, seed, distribution):
    fixed_duration = 6
    disruptions = {}
    one_disruption_for_instance = False

    if seed is not None:
        set_seed(seed)
    for t in steps:
        disruptions[t] = {}
        for aircraft_id in aircraft_ids:
            disruptions[t][aircraft_id] = []
    mu = len(steps[1:-2]) / 2 + 1  # Center of the time horizon
    s = len(steps[1:-2]) / 3  # Controls spread; adjust as needed
    probabilities = norm.pdf(range(len(steps[1:-2])), mu, s)
    probabilities /= probabilities.sum()  # Normalize to sum to 1
    cumulative_probabilities = np.cumsum(probabilities)

    for i in range(n_disrupted_ac):
        aircraft_id = aircraft_ids[i]

        disruption_timestep = random.choice(steps[1:-4])

        # Determine if the disruption should be realized based on the sampled probability
        p = round(random.uniform(0.5, 0.9), 2)               # get probability of disruption happening for this aircraft
        disruption_realized = False                                # Flag if disruption happened for this aircraft
        for t in steps[1:-2]:                                      # Time at which the disruption might become known
            if distribution == 'single_prob':

                if disruption_realized:
                    # If a disruption was realized, carry it forward to future timesteps
                    disruptions[t][aircraft_id] = disruptions[t - 1][aircraft_id]

                elif t == disruption_timestep:
                    start_time = periods[t]
                    unavailability_duration = sample_duration(fixed_duration)
                    unavailability_end_time = start_time + pd.Timedelta(hours=unavailability_duration)

                    # Determine if the disruption should be realized based on the sampled probability
                    realized = random.uniform(0, 1) < p

                    # Store the disruption details
                    disruptions[t][aircraft_id].append((start_time, unavailability_end_time, p, realized))
                    disruption_realized = True

            if distribution == 'normal':
                # If a disruption was realized in the past, carry it forward
                if disruption_realized:
                    # Carry forward the disruption for future timesteps
                    disruptions[t][aircraft_id] = disruptions[t - 1][aircraft_id]
                else:
                    # Check if a disruption should be realized at this timestep
                    if np.random.rand() < cumulative_probabilities[t - 1]:
                        start_time = periods[t]
                        unavailability_duration = sample_duration(fixed_duration)  # Fixed 4-hour duration
                        unavailability_end_time = start_time + pd.Timedelta(hours=unavailability_duration)

                        # Store the disruption for the aircraft
                        disruptions[t][aircraft_id].append((start_time, unavailability_end_time))
                        disruption_realized = True

            if distribution == 'uniform':
                if disruption_realized:
                    # If a disruption was realized, carry it forward to the next timestep
                    disruptions[t][aircraft_id] = disruptions[t - 1][aircraft_id]
                else:
                    # Disruption will happen exactly at the 'disruption_timestep'
                    if t == disruption_timestep:
                        start_time = periods[t]
                        unavailability_duration = sample_duration(fixed_duration)
                        unavailability_end_time = start_time + pd.Timedelta(hours=unavailability_duration)

                        # Store the disruption for the aircraft
                        disruptions[t][aircraft_id].append((start_time, unavailability_end_time))
                        disruption_realized = True

    # Fill last two timesteps with the disruptions:
    disruptions[steps[-4]] = disruptions[steps[-5]]
    disruptions[steps[-3]] = disruptions[steps[-4]]
    disruptions[steps[-2]] = disruptions[steps[-3]]
    disruptions[steps[-1]] = disruptions[steps[-2]]

    return disruptions

def save_disruptions(data, filename):
    """Save the disruptions to a binary file using pickle."""
    disruptions_folder = "disruptions"
    if not os.path.exists(disruptions_folder):
        os.makedirs(disruptions_folder)

    disruptions_file = os.path.join(disruptions_folder, f"{filename}.pkl")

    with open(disruptions_file, 'wb') as f:
        pickle.dump(data, f)  # Serialize and save the disruptions
    print(f"Disruptions saved at {disruptions_file}")

def load_disruptions(filename):
    """Load the disruptions from a binary file using pickle."""
    disruptions_file = os.path.join("disruptions", f"{filename}.pkl")

    with open(disruptions_file, 'rb') as f:
        data = pickle.load(f)  # Deserialize and load the disruptions
    # print(f"Disruptions loaded from {disruptions_file}")
    return data

def get_shuffled_disruption_paths(disruptions, N):
    disruption_indices = list(range(1, N+1))
    random.shuffle(disruption_indices)
    shuffled_paths = [disruptions[n] for n in disruption_indices[:N]]
    return shuffled_paths

if __name__ == '__main__':
    folder = 'TEST1'
    aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)

    aircraft_ids = [aircraft['ID'] for aircraft in aircraft_data]
    num_flights = len(flight_data)
    num_aircraft = len(aircraft_data)

    interval = 60  # minutes
    intervals = pd.date_range(start=recovery_start, end=recovery_end, freq=str(interval) + 'T')
    periods = {i: start for i, start in enumerate(intervals)}
    steps = [i for i in periods.keys()]

    train_disruptions = {}
    test_disruptions = {}
    n_disrupted_ac = 2
    nr_instances = 10001
    for i in range(1, nr_instances + 1):
        disruption_sample = sample_disruption_path(aircraft_ids, n_disrupted_ac, steps, periods, seed=None, distribution='single_prob')
        train_disruptions[i] = disruption_sample

    for i in range(1, nr_instances + 1):
        disruption_sample = sample_disruption_path(aircraft_ids, n_disrupted_ac, steps, periods, seed=None, distribution='single_prob')
        test_disruptions[i] = disruption_sample

    save_disruptions(train_disruptions, f"Disruptions_train")
    save_disruptions(train_disruptions, f"Disruptions_test")
    print(test_disruptions[1])

