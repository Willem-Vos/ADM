import random
import os
from datetime import datetime, timedelta

def round_to_nearest_quarter_hour(dt):
    minute = dt.minute
    # Round to the nearest 15-minute mark
    new_minute = (minute + 7) // 15 * 15  # +7 ensures we round to the nearest quarter

    # If new_minute is 60, increment the hour and reset minutes to 0
    if new_minute == 60:
        dt = dt + timedelta(hours=1)
        new_minute = 0

    return dt.replace(minute=new_minute, second=0, microsecond=0)

def format_time_with_next_day(dt, base_date):
    """Formats time and adds +1 if the time exceeds midnight."""
    if dt.day > base_date.day:
        return dt.strftime('%H:%M') + '+1'
    else:
        return dt.strftime('%H:%M')

def generate_flight_data(num_aircraft, num_flights, start, end, output_folder="TRAIN"):
    # Set base date and time range for departures
    date = '10/01/08'
    base_date = datetime.strptime('10/01/08', '%m/%d/%y')
    start_time = timedelta(hours=start)  # Flights start earliest at 06:00
    end_time = timedelta(hours=end)  # Flights latest departure at 20:00
    max_flight_duration = timedelta(hours=8)
    min_flight_duration = timedelta(hours=1)

    # Generate aircraft list
    aircraft = [f'#{i + 1}' for i in range(num_aircraft)]

    ##### INPUT: ##########
    nr_train_instances = 1000
    nr_test_instances = 100
    #######################
    for instance in range(1, nr_train_instances + nr_test_instances + 1):
        if instance > nr_train_instances:
            instance_folder = os.path.join(output_folder, f"TEST{instance-nr_train_instances}")
        else:
            instance_folder = os.path.join(output_folder, f"TRAIN{instance}")

        # Ensure folder exists
        if not os.path.exists(instance_folder):
            os.makedirs(instance_folder)

        # Initialize data for rotations and flights
        rotations_data = []
        flights_data = []

        # Create a list of times when aircraft are available
        aircraft_availability = {ac: [] for ac in aircraft}  # Track flights for each aircraft

        flight_nr = 1
        while flight_nr <= num_flights:
            # Randomly assign origin and destination
            origin = random.choice(['CDG', 'LHR', 'JFK', 'NCE', 'BKK'])
            destination = random.choice(['CDG', 'LHR', 'JFK', 'NCE', 'BKK'])

            # Randomly assign an aircraft
            assigned_aircraft = random.choice(aircraft)

            # Randomize the flight duration between 1 and 5 hours
            flight_duration = random.randint(60, 300)  # Duration in minutes (1 to 5 hours)
            flight_duration = timedelta(minutes=flight_duration)

            # Ensure no overlap by finding a valid departure time
            valid_flight = False
            max_attempts = 30  # Limit the number of attempts to avoid infinite loops
            attempts = 0

            while not valid_flight and attempts < max_attempts:
                # Randomize the departure time within the allowed window
                available_time = start_time + timedelta(minutes=random.randint(0, int((end_time - start_time).total_seconds() // 60)))
                departure_time = available_time
                arrival_time = departure_time + flight_duration

                # Round to nearest 15 minutes
                departure_time = round_to_nearest_quarter_hour(base_date + departure_time)
                arrival_time = round_to_nearest_quarter_hour(base_date + arrival_time)

                # Check for overlap with existing flights for the assigned aircraft
                overlap = False
                for flight in aircraft_availability[assigned_aircraft]:
                    if not (arrival_time <= flight['ADT'] or departure_time >= flight['AAT']):
                        overlap = True
                        break

                # If no overlap and within time range, flight is valid
                if not overlap and arrival_time <= (base_date + end_time + timedelta(hours=2)):  # Ensure arrival is within the day
                    valid_flight = True
                else:
                    attempts += 1

            if valid_flight:
                # Add the flight to the aircraft's schedule
                aircraft_availability[assigned_aircraft].append({'ADT': departure_time, 'AAT': arrival_time})

                # Append to rotations and flights
                rotations_data.append([flight_nr, base_date.strftime('%m/%d/%y'), assigned_aircraft])
                flights_data.append([flight_nr, origin, destination,
                                     departure_time.strftime("%H:%M"), format_time_with_next_day(arrival_time, base_date)])

                flight_nr += 1

        # Write rotations.csv
        rotations_path = os.path.join(instance_folder, 'rotations.csv')
        with open(rotations_path, 'w') as f:
            for row in rotations_data:
                f.write(' '.join(str(x) for x in row) + '\n')
            f.write('#\n')

        # Write flights.csv
        flights_path = os.path.join(instance_folder, 'flights.csv')
        with open(flights_path, 'w') as f:
            for row in flights_data:
                f.write(' '.join(str(x) for x in row) + '\n')
            f.write('#\n')

        # Write aircraft.csv
        aircraft_path = os.path.join(instance_folder, 'aircraft.csv')
        with open(aircraft_path, 'w') as f:
            for ac in aircraft:
                f.write(f'{ac}\n')
            f.write('#\n')


        # Convert timedelta to hours and minutes
        start_hours, start_remainder = divmod(start_time.seconds, 3600)
        end_hours, end_remainder = divmod(end_time.seconds, 3600)
        start_minutes = start_remainder // 60
        end_minutes = end_remainder // 60

        # Format as HH:MM
        formatted_start_time = f'{start_hours:02}:{start_minutes:02}'
        formatted_end_time = f'{end_hours:02}:{end_minutes:02}'

        # Write config.csv
        config_path = os.path.join(instance_folder, 'config.csv')
        with open(config_path, 'w') as f:
            for row in rotations_data:
                f.write(date+' '+formatted_start_time+' '+date+' '+formatted_end_time+'\n')
            f.write('#\n')

        # Generate alt_aircraft.csv
        generate_alt_aircraft(instance_folder, base_date, aircraft, start, end)

def generate_alt_aircraft(instance_folder, base_date, aircraft_list, start, end):
    max_ua_period = end-start
    min_ua_period = 3
    # Set the window of time
    max_availability_period = timedelta(hours=max_ua_period)

    # Randomly select an aircraft
    selected_aircraft = random.choice(aircraft_list)

    # Randomly choose a start time within the allowed window (between start and end hours)
    start_time = timedelta(hours=random.randint(start, end - max_ua_period))

    # Randomly generate the end time (ensuring it's within 6 hours and before the end of the day window)
    end_time = start_time + timedelta(hours=random.randint(min_ua_period, max_ua_period))

    # Format the time to only display hours and minutes (e.g., HH:MM)
    formatted_start_time = (base_date + start_time).strftime('%H:%M')
    formatted_end_time = (base_date + end_time).strftime('%H:%M')

    # Write alt_aircraft.csv
    alt_aircraft_path = os.path.join(instance_folder, 'alt_aircraft.csv')
    with open(alt_aircraft_path, 'w') as f:
        f.write(f'{selected_aircraft} {base_date.strftime("%m/%d/%y")} {formatted_start_time} {base_date.strftime("%m/%d/%y")} {formatted_end_time}\n')
        f.write('#\n')

if __name__ == '__main__':
    num_aircraft = 4
    num_flights = 12
    start = 8
    end = 18
    generate_flight_data(num_aircraft, num_flights, start, end, output_folder="/Users/willemvos/Thesis/ADM/Data")

