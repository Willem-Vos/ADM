import sys

sys.path.append('/opt/anaconda3/lib/python3.11/site-packages')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# folder = 'A01_example'
folder = 'A01_small'
colors = sns.color_palette("husl", 10)  # Change the number based on the number of airports

# Helper function to parse time strings
def parse_time_string(time_str):
    ''' # Helper function to parse time strings, input is string in format "DD/MM/YYYY HH:MM" '''
    if '+1' in time_str:
        # Remove the +1 and parse the time string
        time_str = time_str.replace('+1', '')
        timestamp = pd.to_datetime(time_str, format='%d/%m/%y %H:%M')
        # Add one day to the timestamp
        timestamp += pd.Timedelta(days=1)
    else:
        # Parse the time string directly
        timestamp = pd.to_datetime(time_str, format='%d/%m/%y %H:%M')

    return timestamp

def read_data(folder):
    """Reads data from files in specified folder and returns lists of disctionaries containing informations on aircraft, flights, rotations, and disruptions respectively"""
    # CONFIGURATION
    with open(f'Data/{folder}/config.csv', 'r') as file:
        lines = file.readlines()
        att = lines[0].split()
        recovery_start_str = str(att[0] + ' ' + att[1])
        recovery_end_str = str(att[2] + ' ' + att[3])
        print(recovery_start_str, recovery_end_str)
        recovery_start = parse_time_string(recovery_start_str)
        recovery_end = parse_time_string(recovery_end_str)


    ############################################## SCHEDULE DATA: ##############################################
    # AIRCRAFT DATA
    aircraft_data = []
    with open(f'Data/{folder}/aircraft.csv', 'r') as file:
        lines = file.readlines()
        att = []
        for line in lines[:-1]:  # dont include the last line of each file that contains an "#"
            att = list(line.split())  # split lines into attributes of aircraft.
            aircraft_info = {
                "ID": att[0],
                # "Model": att[1],
                # "Family": att[2],
                # "Configuration": att[3],
                # "Dist": att[4],
                # "Cost": att[5],
                # "TurnRound": att[6],
                # "Transit": att[7],
                # "Orig": att[8],
                # "Maint": att[9] if len(att) > 9 else None,
                "AssignedFlights": list()
            }
            aircraft_data.append(aircraft_info)

    # FLIGHT DATA
    flight_data = []
    # Flight Orig Dest SDT SAT PrevFlight
    with open(f'Data/{folder}/flights.csv', 'r') as file:
        lines = file.readlines()
        att = []
        for line in lines[:-1]:  # dont include the last line of each file that contains an "#"
            att = list(line.split())  # split lines into attributes of aircraft.
            flight_info = {
                "Flightnr": att[0],
                "Orig": att[1],
                "Dest": att[2],
                "SDT": att[3],
                "SAT": att[4],
                "ADT": att[3],
                "AAT": att[4]
                # "PrevFlight": att[5]
            }
            flight_data.append(flight_info)

    # ROTATIONS DATA
    rotations_data = []
    # Flight Orig Dest SDT SAT PrevFlight
    with open(f'Data/{folder}/rotations.csv', 'r') as file:
        lines = file.readlines()
        att = []
        for line in lines[:-1]:  # dont include the last line of each file that contains an "#"
            att = list(line.split())  # split lines into attributes of aircraft.
            rotations_info = {
                "Flightnr": att[0],
                "DepDate": att[1],
                "Aircraft": att[2],
            }
            rotations_data.append(rotations_info)
    ###################################################################################################################

    ############################################## DISRUPTIONS DATA: ##############################################
    disruptions = []
    # aircraft disruptions
    with open(f'Data/{folder}/alt_aircraft.csv', 'r') as file:
        # %Aircraft StartDate StartTime EndDate EndTime
        lines = file.readlines()
        att = []
        if lines == ['#']:
            print('No aircraft disruptions')
        else:
            for line in lines[:-1]:  # dont include the last line of each file that contains an "#"
                att = list(line.split())  # split lines into attributes of aircraft.
                aircraft_disruption = {
                    "Type": "AU",  # Aircraft unavailablity
                    "Aircraft": att[0],  # aircraft ID
                    "StartTime": att[1] + ' ' + att[2], # Timestamp for start of disruption
                    "EndTime": att[3] + ' ' + att[4]  # Timesstamp for end of disruption
                }
                disruptions.append(aircraft_disruption)

    # Flight Disruptions:
    with open(f'Data/{folder}/alt_flights.csv', 'r') as file:
        # %Flight DepDate Delay
        lines = file.readlines()
        att = []
        if lines == ['#']:
            print('No delays')
        else:
            for line in lines[:-1]:  # dont include the last line of each file that contains an "#"
                att = list(line.split())  # split lines into attributes of aircraft.
                flight_disruption = {
                    "Type": "Delay",
                    "Flightnr": att[0],  # flight number
                    "DepDate": att[1],  # Departure Date
                    "Delay": att[2]  # Delay in minutes,
                }
                disruptions.append(flight_disruption)

    # Airport Disruptions:
    with open(f'Data/{folder}/alt_airports.csv', 'r') as file:
        # %Airport StartDate StartTime EndDate EndTime Dep/h Arr/h
        lines = file.readlines()
        att = []
        if lines == ['#']:
            print('No airport disruptions')
        else:
            for line in lines[:-1]:  # dont include the last line of each file that contains an "#"
                att = list(line.split())  # split lines into attributes of aircraft.
                airport_disruption = {
                    "Type": "AC",  # Airport Closure
                    "Airport": att[0],  # Airport
                    "StartTime": att[1] + ' ' + att[2], # Timestamp for start of disruption
                    "EndTime": att[3] + ' ' + att[4]  # Timesstamp for end of disruption
                }
                disruptions.append(airport_disruption)

    # print(f'Disruptions: {disruptions}')
    ###################################################################################################################


    # Add dates to timestrings of flights
    for flight in flight_data:
        for rotation in rotations_data:
            if flight['Flightnr'] == rotation['Flightnr']:
                flight["SAT"] = rotation["DepDate"] + ' ' + flight["SAT"]
                flight["SDT"] = rotation["DepDate"] + ' ' + flight["SDT"]
                flight["AAT"] = rotation["DepDate"] + ' ' + flight["AAT"]
                flight["ADT"] = rotation["DepDate"] + ' ' + flight["ADT"]

    # Fill the "AssignedFlights" attribute of the aircraft
    for aircraft in aircraft_data:
        assigned_flights = [rotation['Flightnr'] for rotation in rotations_data if rotation['Aircraft'] == aircraft['ID']]
        aircraft['AssignedFlights'].extend(assigned_flights) # list of strings

    # Add Assigned Aicraft to Flights:
    flight_to_aircraft = {}            # Dictionary with flight numbers and aircraft ids as keys and values respectively
    for aircraft in aircraft_data:
        for assigned_flight in aircraft['AssignedFlights']:
            try:
                flight_to_aircraft[assigned_flight] = aircraft['ID']
            except:
                print(f'{assigned_flight} not assigned to aircraft')
                flight_to_aircraft[assigned_flight] = 'None'


    for flight in flight_data:
        flight_number = flight['Flightnr']
        flight['AssignedAircraft'] = flight_to_aircraft.get(flight_number, 'Not Assigned')


    # Convert SDT and SAT to Timestamp
    for flight in flight_data:
        flight['SDT'] = parse_time_string(flight['SDT'])
        flight['SAT'] = parse_time_string(flight['SAT'])
        flight['ADT'] = parse_time_string(flight['ADT'])
        flight['AAT'] = parse_time_string(flight['AAT'])

    # Convert "StartTime" and "EndTime" to Timestamps for airport and aircraft disruptions (not needed for flight delays)
    for disruption in disruptions:
        if disruption['Type'] != "Delay":
            disruption["StartTime"], disruption["EndTime"] = parse_time_string(disruption["StartTime"]), parse_time_string(
                disruption["EndTime"])

    return aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end

def plot_schedule(aircraft_data, flight_data, disruptions, iteration):
    # Plotting
    plt.figure(figsize=(12, 5))

    # plot flights
    for aircraft in aircraft_data:
        for flight_nr in aircraft['AssignedFlights']:
            # Find the corresponding flight in flight_data
            flight = next((flight for flight in flight_data if flight['Flightnr'] == flight_nr), None)
            if flight is not None:
                SDT = flight.get('SDT')
                SAT = flight.get('SAT')
                ADT = flight.get('ADT')
                AAT = flight.get('AAT')

                if SDT and SAT and ADT and AAT:
                    # Plot the flight
                    plt.plot([ADT, AAT], [aircraft['ID'], aircraft['ID']], marker='o', color='blue',
                             linewidth=3.5)
                    # Calculate the midpoint of the flight for labeling
                    midpoint_time = ADT + (AAT - ADT) / 2
                    # Add the flight number as a label in the middle of the flight
                    plt.text(midpoint_time, aircraft['ID'], flight_nr,
                             verticalalignment='bottom', horizontalalignment='center', fontsize=10, color='black')

    # Plot disruptions
    for disruption in disruptions:
        if disruption["Type"] == 'AU':
            start_time = disruption['StartTime']
            end_time = disruption['EndTime']
            plt.plot([start_time, end_time], [disruption['Aircraft'], disruption['Aircraft']], linestyle='--',
                     color='orange', linewidth=2)  # Dashed orange line for disruption
            plt.scatter([start_time, end_time], [disruption['Aircraft'], disruption['Aircraft']], color='orange',
                        marker='x', s=100)  # Markers for disruption start and end

        elif disruption["Type"] == 'Delay':
            flightnr = disruption['Flightnr']
            delay_minutes = int(disruption['Delay'])
            for aircraft in aircraft_data:
                if flightnr in aircraft['AssignedFlights']:
                    flight = get_flight_dict(flightnr, flight_data)
                    ADT = flight['SDT']
                    AAT = flight['SAT']
                    if SDT and SAT and ADT and AAT:
                        plt.plot([ADT, ADT + pd.Timedelta(minutes=delay_minutes)], [aircraft['ID'], aircraft['ID']], linestyle='--',
                                 color='red', linewidth=2)  # Dashed red line for Delay disruption
                        plt.scatter([ADT, ADT + pd.Timedelta(minutes=delay_minutes)], [aircraft['ID'], aircraft['ID']], color='red',
                                    marker='x', s=100)  # Markers for Delay disruption start and end

    plt.xlabel('Time')
    plt.ylabel('Aircraft')
    plt.title(f'Aircraft Flight Schedule, iteration {iteration}')
    plt.grid(True)

    # Format x-axis to show only time
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)  # Rotate x-axis labels to 45 degrees
    plt.show()

def plot_time_space(flight_data):
    # Extract unique OD pairs and map them to colors, treating pairs as bidirectional
    od_pairs = sorted(set(tuple(sorted((flight['Orig'], flight['Dest']))) for flight in flight_data))
    od_colors = {od: colors[i] for i, od in enumerate(od_pairs)}

    # Extract unique locations and map them to numerical values
    locations = sorted(set(flight['Orig'] for flight in flight_data) | set(flight['Dest'] for flight in flight_data))
    location_map = {loc: i for i, loc in enumerate(locations)}

    fig, ax = plt.subplots(figsize=(6, 4))

    # Extract data
    for flight in flight_data:
        dep_time = flight['ADT']
        arr_time = flight['AAT']
        dep_loc = location_map[flight['Orig']]
        arr_loc = location_map[flight['Dest']]
        color = od_colors[tuple(sorted((flight['Orig'], flight['Dest'])))]  # Get the color for the OD pair

        # Plot the flight segment with an arrow
        ax.plot([dep_time, arr_time], [dep_loc, arr_loc], marker='o', markersize=4, color=color)
        ax.annotate('', xy=(arr_time, arr_loc), xytext=(dep_time, dep_loc),
                    arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.5', color=color, lw=4))

        # Display the flight number at the start and end of the flight
        ax.text(dep_time, dep_loc, f"{flight['Flightnr']}", verticalalignment='bottom', horizontalalignment='right',
                color=color)
        ax.text(arr_time, arr_loc, f"{flight['Flightnr']}", verticalalignment='bottom', horizontalalignment='left',
                color=color)

    # Formatting the plot
    ax.set_yticks(list(location_map.values()))
    ax.set_yticklabels(list(location_map.keys()))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Airport')
    plt.title('Time-Space Plot of Flight Schedule')
    plt.grid(True)
    plt.show()

def affected_flights(disruptions, aircraft_data, flight_data):
    flights_affected = []
    for disruption in disruptions:
        if disruption['Type'] != 'AU':
            start_time, end_time = disruption["StartTime"], disruption["EndTime"]
            aircraft_id = disruption['Aircraft']

            for aircraft in aircraft_data:
                if aircraft['ID'] == aircraft_id:
                    for flight_nr in aircraft['AssignedFlights']:
                        flight = get_flight_dict(flight_nr, flight_data)
                        if (flight['SDT'] < end_time) and (flight['SAT'] > start_time):
                            flights_affected.append((flight_nr, flight))

        if disruption['Type'] == 'Delay':
            flight_nr = disruption['Flightnr']
            for flight in flight_data:
                if flight['Flightnr'] == flight_nr:
                    flights_affected.append((flight_nr, flight))

    return flights_affected

# def all_disrupted_flights(aircraft_data, flight_data, disruptions):
#     """Returns a list of lists containing flights affected by a disruption for all disruptions """
#     return [affected_flights(disruption, aircraft_data, flight_data) for disruption in disruptions if
#             affected_flights(disruption, aircraft_data, flight_data) != []]

def update_data_post_disruptions(aircraft_data, flight_data, disruptions):
    for disruption in disruptions:
        if disruption['Type'] == 'Delay':
            disrupted_flight_nr = disruption['Flightnr']
            delay = pd.Timedelta(minutes=int(disruption['Delay']))  # Convert delay to a Timedelta

            # flight data
            disrupted_flight = get_flight_dict(disrupted_flight_nr, flight_data)
            disrupted_flight['ADT'] = disrupted_flight['SDT'] + delay
            disrupted_flight['AAT'] = disrupted_flight['SAT'] + delay

    return aircraft_data, flight_data


def get_flight_dict(flight_nr, flight_data):
    """Get flight data for a specified flight nr"""
    return next(flight for flight in flight_data if flight['Flightnr'] == flight_nr)

def get_aircraft_dict(flight_nr, aircraft_data):
    """Get aircraft data for a specified flight"""
    for aircraft in aircraft_data:
            if flight_nr in aircraft['AssignedFlights']:
                return aircraft
    # return next(aircraft for aircraft in aircraft_data if flight_nr in aircraft["AssignedFlights"])

def ac_index(aircraft_data, aircraft_id):
    """Retrieve index of aircraft in aircraft_data list"""
    for idx, aircraft in enumerate(aircraft_data):
        if aircraft['ID'] == aircraft_id:
            return idx
    return -1  # Return -1 if the tail number is not found

def flight_index(flight_data, flight_nr):
    """Retrieve index of flight in flight_data list"""
    for idx, flight in enumerate(flight_data):
        if flight['Flightnr'] == flight_nr:
            return idx
    return -1  # Return -1 if the flight number is not found

if __name__ == "__main__":
    # Read intial data:
    aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)
    print(affected_flights(disruptions, aircraft_data, flight_data))
    # Plot initial data:
    plot_schedule(aircraft_data, flight_data, disruptions, iteration='before update')
    # plot_time_space(flight_data)

    # Update schedule following disruptions :
    aircraft_data, flight_data = update_data_post_disruptions(aircraft_data, flight_data, disruptions)
    plot_schedule(aircraft_data, flight_data, disruptions, iteration='after update')

    print('\nRECOVERY WINDOW:')
    print(recovery_start)
    print(recovery_end)

    print('\nAIRCRAFT:')
    for aircraft in aircraft_data:
        print(aircraft)

    print('\nFLIGHTS:')
    for flight in flight_data:
        print(flight)
        # print(f'{flight['Flightnr']}: {flight["Orig"]} - {flight["Dest"]} | {flight['SDT']}-{flight['SAT']}')
        # print(f'{flight['SDT']}')
        # print(f'{flight['SAT']}')

    print('\nROTATIONS:')
    for rotation in rotations_data:
        print(rotation)

    print('\nDISRUPTIONS:')
    for disruption in disruptions:
        print(disruption)

    # print('\n CHECK UPDATES IN DATA:')
    # print('\n AIRCRAFT DATA:')
    # for aircraft in aircraft_data:
    #     for flight in aircraft['AssignedFlights']:
    #         print(flight['Flightnr'], flight['ADT'], flight['AAT'])
    #
    # print('\n FLIGHT DATA:')
    # for flight in flight_data:
    #     print(flight['Flightnr'], flight['ADT'], flight['AAT'])

    # print(flight_data)
    # departure_time_flight_7 = next(flight['SDT'] for flight in flight_data if flight['Flightnr'] == '7')
    # print(departure_time_flight_7)


    # print(aircraft_data)
    # aircraft_ids = [aircraft['ID'] for aircraft in aircraft_data]
    # print(aircraft_ids)

    # print(flight_data)
    # airport_ids = list({flight['Orig'] for flight in flight_data} | {flight['Dest'] for flight in flight_data})
    # print(airport_ids)