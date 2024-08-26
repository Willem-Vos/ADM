# update aircraft and flight data for future steps as well
steps = [0,1, 2, 3, 4]
next_step = 3
temp_aircraft_data = ['a', 'b', 'c', 'd', 'e']
temp_flight_data = ['x', 'y', 'z', 'u', 'v']


# for t in range(len(steps)):
#     if next_step < len(steps) - 1:
#         temp_aircraft_data[t] = temp_aircraft_data[next_step] if next_step >= t else temp_aircraft_data[t]
#         temp_flight_data[t] = temp_flight_data[next_step] if next_step >= t else temp_flight_data[t]
#
for t in range(next_step + 1, len(steps)):
    # Ensure that the next step data is correctly propagated forward
    temp_aircraft_data[t] = temp_aircraft_data[t - 1]
    temp_flight_data[t] = temp_flight_data[t - 1]

print(temp_aircraft_data)
print(temp_flight_data)


def plot_schedule(self, step):
    """ Plots the current schedule from the data"""
    # Plotting
    plt.figure(figsize=(10, 5))

    # Plot flights based on the stored order
    for aircraft_id in self.aircraft_order:
        aircraft = next(ac for ac in self.aircraft_data if ac['ID'] == aircraft_id)
        for flight_nr in aircraft['AssignedFlights']:
            # Find the corresponding flight in flight_data
            flight = get_flight_dict(flight_nr, self.flight_data)
            if flight is not None:
                ADT = flight.get('ADT')
                AAT = flight.get('AAT')
                ADT = flight.get('ADT')
                AAT = flight.get('AAT')

                if ADT and AAT and ADT and AAT:
                    # Plot the flight
                    plt.plot([ADT, AAT], [aircraft['ID'], aircraft['ID']], marker='o', color='blue',
                             linewidth=3.5)
                    # Calculate the midpoint of the flight for labeling
                    midpoint_time = ADT + (AAT - ADT) / 2
                    # Add the flight number as a label in the middle of the flight
                    plt.text(midpoint_time, aircraft['ID'], flight_nr,
                             verticalalignment='bottom', horizontalalignment='center', fontsize=10, color='black')

        # Plot disruptions
        for disruption in self.disruptions:
            if disruption["Type"] == 'AU' and aircraft_id == disruption["Aircraft"]:
                start_time = disruption['StartTime']
                end_time = disruption['EndTime']
                plt.plot([start_time, end_time], [disruption['Aircraft'], disruption['Aircraft']], linestyle='--',
                         color='orange', linewidth=2)  # Dashed orange line for disruption
                plt.scatter([start_time, end_time], [disruption['Aircraft'], disruption['Aircraft']], color='orange',
                            marker='x', s=100)  # Markers for disruption start and end

            elif (disruption["Type"] == 'Delay' and
                  get_flight_dict(disruption["Flightnr"], self.flight_data)["AssignedAircraft"] == aircraft_id):
                flightnr = disruption['Flightnr']
                delay_minutes = int(disruption['Delay'])
                if flightnr in get_ac_dict(self.aircraft_data, aircraft_id)['AssignedFlights']:
                    flight = get_flight_dict(flightnr, self.flight_data)
                    ADT = flight['ADT']
                    AAT = flight['AAT']
                    if ADT and AAT and ADT and AAT:
                        plt.plot([ADT, ADT + pd.Timedelta(minutes=delay_minutes)], [aircraft['ID'], aircraft['ID']],
                                 linestyle='--',
                                 color='red', linewidth=2)  # Dashed red line for Delay disruption
                        plt.scatter([ADT, ADT + pd.Timedelta(minutes=delay_minutes)],
                                    [aircraft['ID'], aircraft['ID']], color='red',
                                    marker='x', s=100)  # Markers for Delay disruption start and end

    # Retrieve the current time associated with the step
    current_time = self.periods[step] if step < len(self.periods) else self.recovery_end

    # Plot the current time as a vertical line
    plt.axvline(x=current_time, color='black', linestyle='-', linewidth=2, label='Current Time')

    # Plot recovery_start and recovery_end as vertical lines
    plt.axvline(x=self.recovery_start, color='purple', linestyle='--', linewidth=2, label='Recovery Start')
    plt.axvline(x=self.recovery_end, color='purple', linestyle='--', linewidth=2, label='Recovery End')

    plt.xlabel('Time')
    plt.ylabel('Aircraft')
    plt.title(f'Aircraft Flight Schedule: Step: {step}')
    plt.grid(True)

    # Format x-axis to show only time
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)  # Rotate x-axis labels to 45 degrees

    plt.legend()
    plt.show()