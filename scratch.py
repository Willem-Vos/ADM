from preprocessing import *
from environment import *
folder = 'A01_small'
# Helper function to parse time strings


if __name__ == "__main__":
    # Read initial data:
    aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)

    # Plot initial data:
    plot_schedule(aircraft_data, flight_data, disruptions, iteration=0)
    # plot_time_space(flight_data)

    # # Update schedule following disruptions :
    # aircraft_data, flight_data = update_data_post_disruptions(aircraft_data, flight_data, disruptions)
    # plot_schedule(aircraft_data, flight_data, disruptions)
