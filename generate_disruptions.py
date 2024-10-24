import numpy as np
import pandas as pd
from preprocessing import *
import os
import pickle
import random

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

def sample_disruption_path(aircraft_ids, single_aircraft, steps, periods, p, seed=None):
    # Set the seed for reproducibility if provided
    if seed is not None:
        set_seed(seed)

    disruptions = {}

    # Initialize the disruptions dictionary for all steps and aircraft
    for t in steps:
        disruptions[t] = {}
        for aircraft_id in aircraft_ids:
            disruptions[t][aircraft_id] = []

    if single_aircraft:
        aircraft_id = random.choice(aircraft_ids)
        disruption_realized = False
        for t in steps[1:-2]:  # Time at which the disruption might become known

            # If a disruption was realized in the past, carry it forward
            if disruption_realized:
                disruptions[t][aircraft_id] = disruptions[t - 1][aircraft_id]
            else:
                start_time = periods[t+1]
                if np.random.rand() < p:  # Randomly decide if a disruption occurs
                    unavailability_duration = sample_duration(fixed_duration=4)
                    unavailability_end_time = start_time + pd.Timedelta(hours=unavailability_duration)

                    # Store the disruption for the aircraft
                    disruptions[t][aircraft_id].append((start_time, unavailability_end_time))
                    disruption_realized = True

    if not single_aircraft:
        disruption_realized = False
        for t in steps[1:-2]:  # Time at which the disruption might become known
            for aircraft_id in aircraft_ids:
                # If a disruption was realized in the past, carry it forward
                if disruption_realized:
                    disruptions[t][aircraft_id] = disruptions[t - 1][aircraft_id]
                else:
                    start_time = periods[t+1]
                    if np.random.rand() < p:  # Randomly decide if a disruption occurs
                        unavailability_duration = sample_duration(fixed_duration=4)
                        unavailability_end_time = start_time + pd.Timedelta(hours=unavailability_duration)

                        # Store the disruption for the aircraft
                        disruptions[t][aircraft_id].append((start_time, unavailability_end_time))
                        disruption_realized = True

    # Fill last two timesteps with the disruptions:
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

# save_disruptions(disruptions, f"Disruption_{N}")
def load_disruptions(filename):
    """Load the disruptions from a binary file using pickle."""
    disruptions_file = os.path.join("disruptions", f"{filename}.pkl")

    with open(disruptions_file, 'rb') as f:
        data = pickle.load(f)  # Deserialize and load the disruptions
    print(f"Disruptions loaded from {disruptions_file}")
    return data


if __name__ == '__main__':
    folder = 'TEST1'
    p = 0.15 # probability of unavailability occurring at each timestep.
    aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)

    aircraft_ids = [aircraft['ID'] for aircraft in aircraft_data]
    num_flights = len(flight_data)
    num_aircraft = len(aircraft_data)

    interval = 60  # minutes
    intervals = pd.date_range(start=recovery_start, end=recovery_end, freq=str(interval) + 'T')
    periods = {i: start for i, start in enumerate(intervals)}
    steps = [i for i in periods.keys()]

    disruptions = {}
    single_aircraft= False
    N = 10001
    for n in range(1, N + 1):
        disruption_sample = sample_disruption_path(aircraft_ids, single_aircraft, steps, periods, p, seed=None)
        disruptions[n] = disruption_sample

    save_disruptions(disruptions, f"Disruptions")