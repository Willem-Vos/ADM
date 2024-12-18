from itertools import product
from old.environment import *
from generate_disruptions import *
import os
import json
import random
import numpy as np
import time
from datetime import datetime, timedelta
import pickle
from scipy.signal import savgol_filter
import gurobipy as gp
from gurobipy import GRB
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# from joblib import Parallel, delayed
# TODO:
# >>>>>> Use aggregate states to find best actions during training?
# DONE - Aggregation value updates stepize dependent on times state is visited
# DONE - Create state keys that are only described by flights, not disruptions or conflicts as this is exogeneous info and encode them to numbers
# DONE - Add elapsed time to state vector.



class VFA_ADP:
    def __init__(self, aircraft_data, flight_data, disruptions, recovery_start, recovery_end, agg_lvl, folder):
        self.folder = folder
        self.aircraft_data = aircraft_data
        self.flight_data = flight_data
        self.disruptions = disruptions
        self.aircraft_ids = [aircraft['ID'] for aircraft in self.aircraft_data]
        self.recovery_start = recovery_start
        self.recovery_end = recovery_end

        self.interval = 60 # minutes
        self.intervals = pd.date_range(start=recovery_start, end=recovery_end, freq= str(self.interval)+'T')
        self.periods = {i: start for i, start in enumerate(self.intervals)}
        self.steps = [i for i in self.periods.keys()]
        self.period_length = pd.Timedelta(minutes=self.interval)
        self.total_recovery_time = (self.recovery_end - self.recovery_start).total_seconds() / 60  # Total recovery period in minutes
        self.cancellation_cost = 240

        self.T = self.steps[-1]
        self.N = 100                # Number of iterations per instance
        self.y = 0.8                # Discount factor
        # self.a = 1 / self.N       # learning rate or stepsize, fixed
        self.α = 0.1                # Learning rate or stepsize, fixed
        self.harmonic_a = 2.5       # Parameter for calculating harmonic stepsize
        self.p = 0.15               # Probability of aircraft being unavailable
        self.ε = 0.0                # Exploration probability
        self.ε_init = self.ε
        self.ε_end = 0
        self.harmonic_stepsize = True
        self.decaying_ε = False
        self.deterministic = False  # False for probabilistically sampled disruptions, True for deterministic disruptions from data
        self.plot = True

        # States
        self.states = dict()
        self.agg_states = dict()
        self.aggregation_level = agg_lvl
        self.initial_state = self.initialize_state()
        self.initial_state_key = self.create_hashable_state_key(self.initial_state)


# INITIALIZATION FUNCTIONS:
    def initialize_state(self):
        t = 0
        current_time = self.periods[t]
        self.states = {}
        state_dict = dict()
        agg_dict = dict()

        state_dict['t'] = 0
        for aircraft in self.aircraft_data:
            aircraft_state = {}
            aircraft_id = aircraft['ID']
            ass_flights = self.ass_flights(aircraft_id, t)
            unavailibilty = []
            conflicts = []
            remaining_flights = len(ass_flights)

            aircraft_state = {'conflicts': conflicts, 'UA': unavailibilty, 'flights':ass_flights, 'n_remaining_flights': remaining_flights}
            state_dict[aircraft_id] = aircraft_state

        initial_value = -self.cancellation_cost * self.num_conflicts(state_dict)
        # initial_value = -self.cancellation_cost * (len(self.flight_data) / len(self.aircraft_ids))

        # set value of the state to initial value and iteration to zero:
        state_dict['value'] = initial_value
        state_dict['iteration'] = 0

        agg_dict['count'] = 0
        agg_dict['value'] = initial_value
        # agg_dict['iteration'] = [0]

        state_key = self.create_hashable_state_key(state_dict)
        aggregate_state_key = self.G(state_dict, self.aggregation_level)

        self.states[state_key] = state_dict
        self.agg_states[aggregate_state_key] = agg_dict
        self.agg_states[aggregate_state_key]['count'] = 1

        return state_dict

    def ass_flights(self, aircraft_id, t):
        """Returns a list of copies flight dictionaries (with nr, adt, aat) assigned to an aircraft at a step, read from the data for intializing the states"""
        ass_flights = []
        flight_nrs = get_ac_dict(self.aircraft_data, aircraft_id)['AssignedFlights']
        for flight_nr in flight_nrs:
            flight = copy.deepcopy(get_flight_dict(flight_nr, self.flight_data))
            del flight["AssignedAircraft"]
            ass_flights.append(flight)
        return ass_flights

######### HELPER FUNCTIONS: #########
    def get_flight(self, flight_nr, current_state):
        for aircraft_id in self.aircraft_ids:
            aircraft_state = current_state[aircraft_id]
            for flight in aircraft_state['flights']:
                if flight['Flightnr'] == flight_nr:
                    return flight
        return 'None, Flight not found'

    def G(self, s, lvl):
        """
        Create a hashable aggregate key from a state dictionary that generalizes to unseen instances for testing purposes.

        The first level key includes:
        - n_remaining_flights per aircraft
        - rounded flight times to hours.

        The second level key includes:
        - n_remaining_flights per aircraft
        - n_remaining conflicts per aircraft

        Args:
            s (dict): A dictionary representing the state of the system.

        Returns:
            tuple: A hashable representation of the aggregate state.
        """
        aggregate_state_key = []

        # First level of aggregation
        if lvl == 1:
            # 1. Add the number of remaining flights for each aircraft
            for aircraft_id in self.aircraft_ids:
                aircraft_state = s[aircraft_id]
                # Create a list to store the aircraft attributes
                aircraft_key = []
                aircraft_key.append(aircraft_state['n_remaining_flights'])
                unavailability_times = []

                if aircraft_state['UA']:
                    unavailability_start = (aircraft_state['UA'][0][0] -  self.recovery_start).total_seconds() / 60 / self.total_recovery_time # Convert to minutes
                    unavailability_end = (aircraft_state['UA'][0][1] -  self.recovery_start).total_seconds() / 60 / self.total_recovery_time # Convert to minutes

                    unavailability_times.append(unavailability_start)
                    unavailability_times.append(unavailability_end)
                aircraft_key.append(tuple(unavailability_times))

                # 2. Round the flight times (departure and arrival) to the nearest hour
                rounded_flight_times = []

                # Sort the flights by departure time to ensure they are in chronological order
                sorted_flights = sorted(aircraft_state['flights'], key=lambda flight: flight['ADT'])

                for flight in sorted_flights:
                    # Round departure and arrival times to nearest hour
                    rounded_departure = flight['ADT'].replace(minute=0, second=0, microsecond=0) \
                                        + timedelta(hours=1 if flight['ADT'].minute >= 30 else 0)
                    rounded_arrival = flight['AAT'].replace(minute=0, second=0, microsecond=0) \
                                      + timedelta(hours=1 if flight['AAT'].minute >= 30 else 0)

                    # Relative arrtime and deptime to the recovery start time
                    dep_time = (rounded_departure - self.recovery_start).total_seconds() / 60 / self.total_recovery_time  # Convert to minutes
                    arr_time = (rounded_arrival - self.recovery_start).total_seconds() / 60 / self.total_recovery_time  # Convert to minutes

                    # Append the rounded times to the list
                    rounded_flight_times.append(dep_time)
                    rounded_flight_times.append(arr_time)

                # Add the number of remaining flights and the flight times (as a list, converted to tuple) for this aircraft
                aircraft_key.append(tuple(rounded_flight_times))
                aggregate_state_key.append(tuple(aircraft_key))

        # Second level of aggregation
        if lvl == 2:
            # 1. Add the number of remaining flights for each aircraft
            for aircraft_id in self.aircraft_ids:
                aircraft_state = s[aircraft_id]
                # Create a list to store the aircraft attributes
                aircraft_key = []
                n_conflicts = self.num_ac_conflicts(s, aircraft_id)

                aircraft_key.append(aircraft_state['n_remaining_flights'])
                aircraft_key.append(n_conflicts)

                aggregate_state_key.append(tuple(aircraft_key))

        # Convert the list to a tuple (to make it hashable)
        return tuple(aggregate_state_key)

    def create_hashable_state_key(self, state_dict):
        """
        Create a hashable key from a state dictionary that generalizes to unseen instances for testing purposes.

        The key includes:
        - One entry for the current time step 't'
        - One entry for the remaining time left in the day
        - A sequence of departure and arrival times (in chronological order) for each aircraft.

        Args:
            state_dict (dict): A dictionary representing the state of the system.

        Returns:
            tuple: A hashable representation of the state.
        """
        state_key = []

        # 1. Add the current step 't' to the state key
        state_key.append(state_dict['t'])

        # 2. Calculate and add the remaining time left in the day
        current_time = self.periods[state_dict['t']]
        remaining_time = (self.recovery_end - current_time).total_seconds() / 60 / self.total_recovery_time # Time left in minutes
        state_key.append(remaining_time)

        # 3. For each aircraft, add the sequence of departure and arrival times
        for aircraft_id in self.aircraft_ids:
            aircraft_state = state_dict[aircraft_id]
            # Create a list to store the aircraft attributes
            aircraft_key = []
            aircraft_key.append(aircraft_state['n_remaining_flights'])
            unavailability_times = []

            if aircraft_state['UA']:
                unavailability_start = (aircraft_state['UA'][0][0] -  self.recovery_start).total_seconds() / 60 / self.total_recovery_time # Convert to minutes
                unavailability_end = (aircraft_state['UA'][0][1] -  self.recovery_start).total_seconds() / 60 / self.total_recovery_time # Convert to minutes

                unavailability_times.append(unavailability_start)
                unavailability_times.append(unavailability_end)
            aircraft_key.append(tuple(unavailability_times))

            flight_times = []
            # Sort the flights by departure time to ensure they are in chronological order
            sorted_flights = sorted(aircraft_state['flights'], key=lambda flight: flight['ADT'])

            for flight in sorted_flights:
                # Append both the departure and arrival times relative to the recovery start time
                dep_time = (flight['ADT'] - self.recovery_start).total_seconds() / 60 / self.total_recovery_time # Convert to minutes
                arr_time = (flight['AAT'] - self.recovery_start).total_seconds() / 60 / self.total_recovery_time # Convert to minutes

                # Add both departure and arrival times to the sequence
                flight_times.append(dep_time)
                flight_times.append(arr_time)

            # Append the flight sequence to the state key for this aircraft
            aircraft_key.append(tuple(flight_times))
            state_key.append(tuple(aircraft_key))

        # Convert the list of state elements to a tuple (making it hashable)
        return tuple(state_key)

    def num_ac_conflicts(self, state, id):
        """Counts the total number of future conflicts for one aircraft id state"""
        n = 0
        current_time = self.periods[state['t']]
        for conflict in state[id]['conflicts']:
            if conflict['ADT'] >= current_time:
                n += 1
        return n

    def num_conflicts(self, state):
        """Counts the total number of future conflicts for the whole state"""
        n = 0
        current_time = self.periods[state['t']]
        for id in self.aircraft_ids:
            for conflict in state[id]['conflicts']:
                if conflict['ADT'] >= current_time:
                    n += 1
        return n

    def num_remaing_flights(self, next_state, aircraft_id, next_step):
        n = 0
        next_time = self.periods[next_step]
        for flight in next_state[aircraft_id]['flights']:
            if flight['ADT'] >= next_time:
                n += 1
        return n

########################### LOGIC ###########################
    def unavailable_for_flight(self, flight, aircraft_id, t):
        """ Checks if the aircraft is unavailable due to disruption at Departure time of flight"""
        for disruption in self.disruptions[t][aircraft_id]:
            return (disruption[0] <= flight['ADT'] <= disruption[1] or
                    flight['ADT'] <= disruption[0] <= flight['AAT'])

    def conflict_at_step(self, state, aircraft_id, t):
        aircraft_state = state[aircraft_id]
        conflicts = []
        for flight in aircraft_state['flights']:
            for ua in aircraft_state['UA']:
                if self.disrupted(ua, flight):
                    conflicts.append(flight)
        return conflicts

    def overlaps(self, flight_to_swap, flight):
        """Checks if two flight times overlap"""
        return (flight_to_swap['ADT'] <= flight['ADT'] <= flight_to_swap['AAT'] or
                flight_to_swap['ADT'] <= flight['AAT'] <= flight_to_swap['AAT'] or
                (flight_to_swap['ADT'] <= flight['ADT'] and flight_to_swap['AAT'] >= flight['AAT']) or
                (flight_to_swap['ADT'] >= flight['ADT'] and flight_to_swap['AAT'] <= flight['AAT'])
                )

    def disrupted(self, unavailability, flight):
        """Checks if two flight times overlap"""
        start = unavailability[0]
        end = unavailability[1]

        return (start <= flight['ADT'] <= end or
                start <= flight['AAT'] <= end or
                (start <= flight['ADT'] and end >= flight['AAT']) or
                (start >= flight['ADT'] and end <= flight['AAT'])
                )

    def X_ta(self, current_state):
        t = current_state['t']
        current_time = self.periods[current_state['t']]

        # Initialize actions with a default 'none' action
        actions = np.array([('none', 'none', 'none')], dtype=object)

        # Iterate over all aircraft
        for aircraft_id in self.aircraft_ids:
            aircraft_state = current_state[aircraft_id]

            # Iterate over all flights of the current aircraft
            for flight in aircraft_state['flights']:
                flight_nr = flight['Flightnr']
                flight_to_swap = self.get_flight(flight_nr, current_state)  # flight dict

                # Only consider flights that have not yet departed
                if flight_to_swap['ADT'] < current_time:
                    continue

                # Consider swapping this flight to every other aircraft
                for other_aircraft_id in self.aircraft_ids:
                    if self.calculate_utilization(current_state, other_aircraft_id) > 0.75:
                        continue
                    # Create the swap action and append it to the actions array
                    new_action = np.array([('swap', flight_nr, other_aircraft_id)], dtype=object)
                    actions = np.vstack([actions, new_action])

        return actions

    ####################### REWARD LOGIC ########################
    def check_overlapping_assignments(self, next_state, aircraft_id, changed_flight):
        overlapping_flights = []

        # Check if the swapped flight overlaps with other flights assigned to the aircraft
        for flight in next_state[aircraft_id]['flights']:
            if flight != changed_flight and self.overlaps(changed_flight, flight):
                overlapping_flights.append(flight)

        # Sort the list of overlapping flights by ADT (Actual Departure Time)
        overlapping_flights = sorted(overlapping_flights, key=lambda flight: flight['ADT'])

        return overlapping_flights  # Return sorted overlapping flights

    def calculate_utilization(self, current_state, aircraft_id):
        t = current_state['t']
        current_time = self.periods[current_state['t']]
        aircraft_state = current_state[aircraft_id]

        total_time_in_flight = pd.Timedelta(0)  # Initialize total flight time to zero
        latest_arrival = self.recovery_start

        # Iterate over all the flights assigned to this aircraft
        for flight in aircraft_state['flights']:
            departure_time = flight['ADT']  # Actual Departure Time
            arrival_time = flight['AAT']  # Actual Arrival Time
            latest_arrival = arrival_time if arrival_time > latest_arrival else latest_arrival
            flight_duration = pd.Timedelta(0)


            if departure_time > current_time:
                # Calculate flight duration
                flight_duration = arrival_time - departure_time

            elif arrival_time > current_time > departure_time:
                flight_duration = arrival_time - current_time

            total_time_in_flight += flight_duration

        # Calculate utilization ratio
        total_available_time = latest_arrival - current_time if latest_arrival > current_time else pd.Timedelta(0)
        utilization = total_time_in_flight / total_available_time if total_available_time > pd.Timedelta(0) else 0

        return utilization

    def delay_swapped_flight(self, next_state, aircraft_id, changed_flight, overlapping_flights, apply):
        """
        Delay the swapped flight and potentially other flights to prevent overlap.

        Args:
            next_state (dict): The state of the system after swapping.
            aircraft_id (str): The ID of the aircraft.
            changed_flight (dict): The swapped flight.
            overlapping_flights (list): A list of flights that overlap with the swapped flight.

        Returns:
            dict or None: The updated next state if conflicts are resolved, otherwise None.
        """
        temp_next_state = copy.deepcopy(next_state)
        step = next_state['t']
        if not overlapping_flights:
            return None

        # Process each overlapping flight
        while overlapping_flights:
            if apply:
                print(f'\n##################################')
                print(f'Changed Flight = {changed_flight['Flightnr']} is assigned to {aircraft_id}')
                print(f'{[flight["Flightnr"] for flight in overlapping_flights]}')


            for overlapping_flight in overlapping_flights:
                # Case 1: Swapped/changed flight departs later, delay swapped/changed flight
                if overlapping_flight['ADT'] < changed_flight['ADT']:
                    new_start_time = overlapping_flight['AAT'] + pd.Timedelta(minutes=10)
                    flight_duration = changed_flight['AAT'] - changed_flight['ADT']
                    new_end_time = new_start_time + flight_duration

                    if apply:
                        print(f'Delayed flight {changed_flight["Flightnr"]} with {new_start_time - changed_flight["ADT"]}')


                    # Apply the delay to the swapped/changed flight
                    changed_flight['ADT'] = new_start_time
                    changed_flight['AAT'] = new_end_time
                    delayed_flight = changed_flight

                    # IMPORTANT: Update the flight in the temp_next_state
                    for i, flight in enumerate(temp_next_state[aircraft_id]['flights']):
                        if flight['Flightnr'] == delayed_flight['Flightnr']:
                            temp_next_state[aircraft_id]['flights'][i] = delayed_flight

                # Case 2: Overlapping flight departs later, delay the overlapping flight
                else:
                    new_start_time = changed_flight['AAT'] + pd.Timedelta(minutes=10)
                    flight_duration = overlapping_flight['AAT'] - overlapping_flight['ADT']
                    new_end_time = new_start_time + flight_duration
                    if apply:
                        print(f'Delayed flight {overlapping_flight["Flightnr"]} with {new_start_time - overlapping_flight["ADT"]}')

                    # Apply the delay to the overlapping flight
                    overlapping_flight['ADT'] = new_start_time
                    overlapping_flight['AAT'] = new_end_time
                    delayed_flight = overlapping_flight

                    # IMPORTANT: Update the flight in the temp_next_state
                    for i, flight in enumerate(temp_next_state[aircraft_id]['flights']):
                        if flight['Flightnr'] == delayed_flight['Flightnr']:
                            temp_next_state[aircraft_id]['flights'][i] = delayed_flight

            # Re-check for new conflicts after each delay
            if apply:
                print(f'Check overlapping assignment for newly changed flight: {delayed_flight}')
            overlapping_flights = self.check_overlapping_assignments(temp_next_state, aircraft_id, delayed_flight)
            if apply:
                print(f'Overlapping flights:')
                print(f'{[flight for flight in overlapping_flights]}')
            changed_flight = delayed_flight

            # If new conflicts emerge, continue the loop, otherwise break
            if not overlapping_flights or overlapping_flights == []:
                break

        # next_state = temp_next_state
        # If no further conflicts, return the updated state and total delay
        return temp_next_state

    def delay_disrupted_flight(self, next_state, aircraft_id, disrupted_flight, unavailability_periods, apply):
        """
        Delay the disrupted flight due to aircraft unavailability and propagate delays for subsequent flights.

        Args:
            next_state (dict): The state of the system after the disruption.
            aircraft_id (str): The ID of the aircraft.
            disrupted_flight (dict): The disrupted flight due to aircraft unavailability.
            unavailability_periods (list): A list of (start_time, end_time) tuples for the aircraft's unavailability periods.
            apply (bool): If True, print debug information.

        Returns:
            dict: The updated next state with the delayed flight and any subsequent delays.
        """
        temp_next_state = copy.deepcopy(next_state)
        step = next_state['t']

        if not unavailability_periods:
            return None

        # Sort unavailability periods to ensure processing them in order
        unavailability_periods.sort(key=lambda x: x[1])  # Sort by end_time

        # Process each unavailability period
        for unavailability in unavailability_periods:
            unavailability_start, unavailability_end = unavailability
            # Check if the flight's ADT is within the unavailability period
            if self.disrupted(unavailability, disrupted_flight):
                # Delay the disrupted flight to after the unavailability ends
                new_start_time = unavailability_end + pd.Timedelta(minutes=10)  # Add a buffer of 10 minutes
                flight_duration = disrupted_flight['AAT'] - disrupted_flight['ADT']
                new_end_time = new_start_time + flight_duration

                if apply:
                    print(f'Delayed flight {disrupted_flight["Flightnr"]} to start after unavailability ends at {new_start_time}')
                    print(f'UA: {unavailability_periods}')
                    print(f'Disrupted flight: {disrupted_flight}')

                # Apply the delay to the disrupted flight
                disrupted_flight['ADT'] = new_start_time
                disrupted_flight['AAT'] = new_end_time

                # IMPORTANT: Update the flight in the temp_next_state
                for i, flight in enumerate(temp_next_state[aircraft_id]['flights']):
                    if flight['Flightnr'] == disrupted_flight['Flightnr']:
                        temp_next_state[aircraft_id]['flights'][i] = disrupted_flight


            # Check for subsequent flights and delay them as needed
            overlapping_flights = self.check_overlapping_assignments(temp_next_state, aircraft_id, disrupted_flight)
            if overlapping_flights != [] and self.delay_swapped_flight(temp_next_state, aircraft_id, disrupted_flight, overlapping_flights, apply=False) is not None:
                temp_next_state = self.delay_swapped_flight(temp_next_state, aircraft_id, disrupted_flight, overlapping_flights, apply=False)

        # Update the next state with all delayed flights
        # next_state = temp_next_state
        return temp_next_state

    def check_delays(self, current_state, next_state):
        total_delay = 0
        pre_flights = [flight for aircraft_id in self.aircraft_ids for flight in current_state[aircraft_id]['flights']]
        post_flights = [flight for aircraft_id in self.aircraft_ids for flight in next_state[aircraft_id]['flights']]

        pre_flights = sorted(pre_flights, key=lambda flight: flight['Flightnr'])
        post_flights = sorted(post_flights, key=lambda flight: flight['Flightnr'])

        for pre_flight, post_flight in zip(pre_flights, post_flights):
            # Calculate the delay in minutes (if any) by comparing ADT
            delay_minutes = (post_flight['ADT'] - pre_flight['ADT']).total_seconds() / 60.0
            if delay_minutes > 0:
                total_delay += delay_minutes

        return total_delay

    def check_canx(self, current_state, next_state):
        current_t = current_state['t']
        if current_t == self.T:
            return 0

        next_t = next_state['t']
        current_time = self.periods[current_t]
        next_time = self.periods[next_t]

        canx = 0
        for aircraft in self.aircraft_ids:
            pre_conflicts = current_state[aircraft]['conflicts']
            post_conflicts = next_state[aircraft]['conflicts']

            # Check for conflicts that started in the current period and remain unresolved in the next period
            for conflict in pre_conflicts:
                # Conflict started during the current period and still exists in the next state
                if current_time <= conflict['ADT'] < next_time and conflict in post_conflicts:
                    canx += 1

        return canx

    def compute_reward(self, pre_decision_state, post_decision_state, action):
        """
        Computes the reward for transitioning from the pre-decision state to the post-decision state.

        Args:
            pre_decision_state (dict): The state of the system before the action was taken.
            post_decision_state (dict): The state of the system after the action was taken.
            action (tuple): The action taken, which could be a swap or other types of actions.

        Returns:
            int: The computed reward.
        """
        reward = 0
        action_type, swapped_flight, old_aircraft_id = action

        # 1. Check how many flights did not get recoverd:
        canx = self.check_canx(pre_decision_state, post_decision_state)
        reward -= canx * self.cancellation_cost

        # 2. Check how many conflicts got resolved:
        # pre_conflicts = sum([len(pre_decision_state[aircraft_id]['conflicts']) for aircraft_id in self.aircraft_ids])
        # post_conflicts = sum([len(post_decision_state[aircraft_id]['conflicts']) for aircraft_id in self.aircraft_ids])
        # nr_resolved_conflicts = pre_conflicts - post_conflicts

        # reward += nr_resolved_conflicts * self.cancellation_cost

        # Penalties for performing a swap actions
        if action_type == 'swap':
            reward -= 10

            # Check if delays were necessary following the swaps:
            delay = self.check_delays(pre_decision_state, post_decision_state)
            reward -= delay
        return reward

    def parallel_process_action(self, S_t_dict, x, t, n):
        S_tx_dict = self.simulate_action_to_state(S_t_dict, x, t, n)
        r_t = self.compute_reward(S_t_dict, S_tx_dict, x)
        v_downstream = S_tx_dict['value']
        return tuple(x), r_t, v_downstream, S_tx_dict

################### SOLVE: ##################
    def solve_with_gurobi(self, S_t_dict, X_ta, t, n):
        # Initialize the Gurobi model
        model = gp.Model("ADP_Optimization")

        # Add decision variables for each action in X_ta
        action_vars = model.addVars(len(X_ta), vtype=GRB.BINARY, name="actions")

        # Objective function: maximize the expected value
        V_x = {}
        R_t = {}
        for i, x in enumerate(X_ta):
            S_tx_dict = self.simulate_action_to_state(S_t_dict, x, t, n)
            r_t = self.compute_reward(S_t_dict, S_tx_dict, x)
            v_downstream = S_tx_dict['value']
            V_x[i] = r_t + self.y * v_downstream
            R_t[tuple(x)] = r_t

        # Set the objective to maximize the sum of expected rewards for chosen actions
        model.setObjective(gp.quicksum(V_x[i] * action_vars[i] for i in range(len(X_ta))), GRB.MAXIMIZE)

        # Add any necessary constraints (e.g., constraints on actions)
        # For example: if only one action can be chosen at a time
        model.addConstr(gp.quicksum(action_vars[i] for i in range(len(X_ta))) == 1, "action_constraint")

        # Optimize the model
        model.optimize()

        # Extract the best action: find the action with a solution value of 1
        best_action_idx = None
        for i in range(len(X_ta)):
            if action_vars[i].x > 0.02:  # Gurobi returns solution values as floats (0.0 or 1.0)
                best_action_idx = i
                break

        if best_action_idx is not None:
            x_hat = X_ta[best_action_idx]
            v_hat = V_x[best_action_idx]
            return x_hat, v_hat, R_t
        else:
            raise ValueError("No valid action found")


    def train_with_vfa(self):
        objective_function_values = {}
        value_function_values = {}
        self.policy = {}
        agg_states_count = 1

        disruptions = load_disruptions("Disruptions")      # get pre sampled disruptions from storage
        count = 0
        # iterations/episodes
        for n in range(1, int(self.N) + 1):
            # print(f'{n= }')
            next_state = self.states[self.initial_state_key]
            self.disruptions = disruptions[n]                   # take n'th realization from storage
            f = (self.N - n) / self.N
            ε = (self.ε_init - self.ε_end)*f + self.ε_end       # Decaying ε
            initial_expected_value = next_state['value']

            count += count_disruptions(self.disruptions)
            accumulated_rewards = []
            for t in self.steps[:-1]:
                V_x = {}
                R_t = {}
                V_downstream = {}
                # if n == self.N:
                #     self.plot_schedule(next_state, n, folder)

                S_t_dict = next_state  # Pre-decsision state
                S_t = self.create_hashable_state_key(S_t_dict)
                S_tx_prev_dict = self.states[S_tx] if t > 0 else self.states[S_t]  # Carry the post-decision state forward from previous timestep (It gets updated at this step timestep)
                S_tx_prev = S_tx if t > 0 else S_t

                S_tx_g_prev = self.G(S_tx_prev_dict, self.aggregation_level)
                S_tx_g_prev_dict = self.agg_states[S_tx_g_prev] if S_tx_g_prev in self.agg_states else {'count': 1, 'value': S_tx_prev_dict['value']}
                self.agg_states[S_tx_g_prev] = S_tx_g_prev_dict

# CALCULATE STATES AND VALUES:
                X_ta = self.X_ta(S_t_dict) # Action set np array
                print(f'{len(X_ta)} actions')
                # x_hat, v_hat, R_t = self.solve_with_gurobi(S_t_dict, X_ta, t, n)
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.parallel_process_action, S_t_dict, x, t, n) for x in X_ta]
                    for future in as_completed(futures):
                        x, r_t, v_downstream, S_tx_dict = future.result()
                        V_x[x] = r_t + self.y * v_downstream
                        V_downstream[x] = v_downstream
                        R_t[x] = r_t
#
#                 # for x in X_ta:
#                 #     # get next states for all possible actions
#                 #     S_tx_dict = self.simulate_action_to_state(S_t_dict, x, t, n)
#                 #     r_t = self.compute_reward(S_t_dict, S_tx_dict, x)
#                 #     v_downstream = S_tx_dict['value'][-1]
#                 #     x = tuple(x)
#                 #     V_x[x] = r_t + self.y * v_downstream
#                 #     V_downstream[x] = v_downstream
#                 #     R_t[x] = r_t
#
                # Choose the best action
                x_hat = max(V_x, key=V_x.get)
                v_hat = V_x[x_hat]
                self.policy[S_t] = x_hat

                # print(f' {t = }')
                # print(f'\tBEST ACTION           >>>> {x_hat = }:')
                # # print(f'\tBEST IMMEDIATE REWARD >>>> {rewards[index] = }:')
                # print(f'\tDOWNSTREAM REWARD     >>>> {downstream_vals[index] = }:')
                # print(f'\tBEST ACTION VALUE     >>>> {v_hat = }')

# UPDATE VALUES:
    # Update the value of the pre-decision state
                v_n_prev = self.states[S_tx_prev]['value'] # Value of previous timestep post-decision state at previous iteration
    # Value Updates:
                if self.harmonic_stepsize:
                    # N = len(self.states[S_tx_prev]['value'])
                    N = self.states[S_tx_prev]['iteration']
                    α =  self.harmonic_a / (self.harmonic_a + N)
                    v_n_new = (1 - α) * v_n_prev + α * v_hat  # Value of previous timestep post-decision state at this timestep
                    # print(f'{v_n_prev = }')
                    # print(f'v_n_new = (1 - {α})*{v_n_prev} + {α}*{v_hat} = {v_n_new}\n')
                else:
                    v_n_new = (1 - self.α) * v_n_prev + self.α * v_hat # Value of previous timestep post-decision state at this timestep
                    # print(f'{v_n_prev = }')
                    # print(f'v_n_new = (1 - {self.α})*{v_n_prev} + {self.α}*{v_hat} = {v_n_new}\n ')

                # print(f'S_tx_t-1 values before update:')
                # print(f'{self.states[S_tx_prev]["value"][-5:]}')
                # self.print_state(self.states[S_tx_prev])

    # Update value to the post-decision state directly in self.states
                self.states[S_tx_prev]['value'] = v_n_new
                self.states[S_tx_prev]['iteration'] += 1

                # print(f'S_tx_t-1 values after update:')
                # print(f'{self.states[S_tx_prev]["value"][-5:]}')
                # self.print_state(self.states[S_tx_prev])

                # Update values of aggregated state
                a = self.agg_states[S_tx_g_prev]['count'] / agg_states_count
                v_g = self.agg_states[S_tx_g_prev]['value']
                v_0 = S_tx_prev_dict['value']
                v_smoothed = (1 - a) * v_g + a * v_0

                self.agg_states[S_tx_g_prev]['value'] = v_smoothed
                self.agg_states[S_tx_g_prev]['count'] += 1
                # self.agg_states[S_tx_g_prev]['iteration'].extend([n])
                agg_states_count += 1
# TRANSITION:
    # Select greedy move (if ε > 0)
                x_ε = random.choice(X_ta)
                x_hat = tuple((random.choices(population=[x_hat, x_ε], weights=[1 - ε, ε])[0] if self.decaying_ε
                        else random.choices(population=[x_hat, x_ε], weights=[1 - self.ε, self.ε])[0]))

                best_immediate_reward = R_t[x_hat]
                accumulated_rewards.append(best_immediate_reward)

    # Apply the action and get the post-decision state
                S_tx_dict = self.apply_action_to_state(S_t_dict, x_hat, t, n)
                S_tx = self.create_hashable_state_key(S_tx_dict)
                # print(f'S_tx value:')
                # print(f'{self.states[S_tx]["value"][-5:]}\n')
                # self.print_state(self.states[S_tx])

    # Aggregation of states
                S_tx_g = self.G(S_tx_dict, self.aggregation_level)

    # Update values of aggregated state
                if S_tx_g not in self.agg_states:
                    self.agg_states[S_tx_g] = {
                        'count': 1,
                        'value': S_tx_dict['value']
                        # 'iteration': [n],
                    }
                else:
                    S_tx_g_dict = self.agg_states[S_tx_g]

    # Add exogeneous information to post-decision state to get the next pre-decision state
                S_t_next_dict, S_t_next = self.add_exogeneous_info(S_tx, n)
                self.states[S_t_next] = S_t_next_dict
                # print(f'S(S_tx, W_t) values:')
                # print(f'{self.states[S_t_next]["value"][-5:]}\n')
                # self.print_state(self.states[S_t_next])
                next_state = self.states[S_t_next]

            # Store the objective value for this iteration
            objective_value = sum(accumulated_rewards)
            objective_function_values[n] = objective_value
            value_function_values[n] = initial_expected_value
            # print(f'Z({n}) = {objective_value}')
            # print(f'Rewards: {accumulated_rewards}')
            # print(f'E[V0_{n}]= {initial_expected_value}')

        # Plot the results
        print(f'No disruptions occurred in {100*count/ self.N}% of the iterations')
        self.avg_obj = np.mean(list(objective_function_values.values()))
        if self.plot:
            self.plot_values(value_function_values, metric="E[V0_n]")
            # self.plot_values(objective_function_values, metric='Objective value')
        print(f'Average Objective Value: {self.avg_obj}')

    def simulate_action_to_state(self, S_t_dict, x, t, n):
        '''Does the same as apply action to state, run when checking actions. Run apply_action_to_state for actually appliying actions'''
        # Create a copy of the current state to modify
        S_tx_dict = copy.deepcopy(S_t_dict)
        current_step = S_tx_dict['t']
        next_step = current_step + 1

        action_type, swapped_flight_nr, new_aircraft_id = x

        if action_type == 'swap':
            # 1. Swap the assignments of the aircraft for the flight
            old_aircraft_id = next((aircraft_id for aircraft_id, aircraft_state in S_tx_dict.items()
                                    if aircraft_id != 't' and
                                    aircraft_id != 'time_left' and
                                    any(flight['Flightnr'] == swapped_flight_nr for flight in aircraft_state['flights'])), None)
            old_aircraft_state = S_tx_dict[old_aircraft_id]

            # Find the flight in the current aircraft's stateflight_nr
            flight_to_swap = next(flight for flight in old_aircraft_state['flights'] if flight['Flightnr'] == swapped_flight_nr)

            # Remove the flight from the old aircraft and assign it to the new aircraft
            S_tx_dict[new_aircraft_id]['flights'].append(flight_to_swap)
            S_tx_dict[old_aircraft_id]['flights'].remove(flight_to_swap)

            # 2 Check for overlaps and delay flights if possible:
            swapped_flight = flight_to_swap
            overlapping_flights = self.check_overlapping_assignments(S_tx_dict, new_aircraft_id, swapped_flight)
            if overlapping_flights != [] and self.delay_swapped_flight(S_tx_dict, new_aircraft_id, swapped_flight, overlapping_flights, apply=False) is not None:
                S_tx_dict = self.delay_swapped_flight(S_tx_dict, new_aircraft_id, swapped_flight, overlapping_flights, apply=False)

            # 3 Check for unavailability and delay flights if possible
            unavailabilities = S_tx_dict[new_aircraft_id]['UA']
            if self.delay_disrupted_flight(S_tx_dict, new_aircraft_id, swapped_flight, unavailabilities, apply=False) is not None:
                S_tx_dict = self.delay_disrupted_flight(S_tx_dict, new_aircraft_id, swapped_flight, unavailabilities, apply=False)

        if action_type == 'none':
            # nothing happens to the assignments when doing nothing.
            pass

        for i, aircraft_id in enumerate(self.aircraft_ids):
            aircraft_state = S_tx_dict[aircraft_id]
            aircraft_state['conflicts'] = self.conflict_at_step(S_tx_dict, aircraft_id, next_step) if t != self.T else 0
            aircraft_state['n_remaining_flights'] = self.num_remaing_flights(S_tx_dict, aircraft_id, next_step)

        # update time for next state:
        S_tx_dict['t'] = next_step

        S_tx = self.create_hashable_state_key(S_tx_dict)
        if S_tx in self.states:
            return self.states[S_tx]
        else:
            # if it is a newly expored state: calculated the intial value as function of t
            S_tx_dict['value'] = -self.cancellation_cost * self.num_conflicts(S_tx_dict)
            # S_tx_dict['value'] = [-self.cancellation_cost * (len(self.flight_data) / len(self.aircraft_ids))]
            S_tx_dict['iteration'] = 0
            return S_tx_dict

    def apply_action_to_state(self, S_t_dict, x, t, n):
        '''Does the same as apply action to state, run when checking actions. Run apply_action_to_state for actually appliying actions'''
        # Create a copy of the current state to modify
        S_tx_dict = copy.deepcopy(S_t_dict)
        current_step = S_tx_dict['t']
        next_step = current_step + 1

        action_type, swapped_flight_nr, new_aircraft_id = x

        if action_type == 'swap':
            # 1. Swap the assignments of the aircraft for the flight
            old_aircraft_id = next((aircraft_id for aircraft_id, aircraft_state in S_tx_dict.items()
                                    if aircraft_id != 't' and
                                    aircraft_id != 'time_left' and
                                    any(flight['Flightnr'] == swapped_flight_nr for flight in aircraft_state['flights'])), None)
            old_aircraft_state = S_tx_dict[old_aircraft_id]

            # Find the flight in the current aircraft's stateflight_nr
            flight_to_swap = next(flight for flight in old_aircraft_state['flights'] if flight['Flightnr'] == swapped_flight_nr)

            # Remove the flight from the old aircraft and assign it to the new aircraft
            S_tx_dict[new_aircraft_id]['flights'].append(flight_to_swap)
            S_tx_dict[old_aircraft_id]['flights'].remove(flight_to_swap)

            # 2 Check for overlaps and delay flights if possible:
            swapped_flight = flight_to_swap
            overlapping_flights = self.check_overlapping_assignments(S_tx_dict, new_aircraft_id, swapped_flight)
            if overlapping_flights != [] and self.delay_swapped_flight(S_tx_dict, new_aircraft_id, swapped_flight, overlapping_flights, apply=False) is not None:
                S_tx_dict = self.delay_swapped_flight(S_tx_dict, new_aircraft_id, swapped_flight, overlapping_flights, apply=False)

            # 3 Check for unavailability and delay flights if possible
            unavailabilities = S_tx_dict[new_aircraft_id]['UA']
            if self.delay_disrupted_flight(S_tx_dict, new_aircraft_id, swapped_flight, unavailabilities, apply=False) is not None:
                S_tx_dict = self.delay_disrupted_flight(S_tx_dict, new_aircraft_id, swapped_flight, unavailabilities, apply=False)


        if action_type == 'none':
            # nothing happens to the assignments when doing nothing.
            pass

        for i, aircraft_id in enumerate(self.aircraft_ids):
            aircraft_state = S_tx_dict[aircraft_id]
            aircraft_state['conflicts'] = self.conflict_at_step(S_tx_dict, aircraft_id, next_step) if t != self.T else 0
            aircraft_state['n_remaining_flights'] = self.num_remaing_flights(S_tx_dict, aircraft_id, next_step)

        # update time for next state:
        S_tx_dict['t'] = next_step
        S_tx = self.create_hashable_state_key(S_tx_dict)

        if S_tx in self.states:
            return self.states[S_tx]

        else:
            S_tx_dict['value'] = -self.cancellation_cost * self.num_conflicts(S_tx_dict)
            # S_tx_dict['value'] = [-self.cancellation_cost * (len(self.flight_data) / len(self.aircraft_ids))]
            S_tx_dict['iteration'] = 0
            self.states[S_tx] = S_tx_dict
            return self.states[S_tx]

    def add_exogeneous_info(self, S_tx, n):
        S_tx_dict = self.states[S_tx]
        t = S_tx_dict['t']
        W_t_next = self.disruptions[t]
        S_t_next_dict = copy.deepcopy(S_tx_dict)

        for aircraft_id in self.aircraft_ids:
            aircraft_state = S_t_next_dict[aircraft_id]
            # Add Exogeneous information - new realizations of aircraft unavailabilities:
            aircraft_state['UA'] = W_t_next[aircraft_id]

            # Update state attribute 'conflicts' for new information
            aircraft_state['conflicts'] = self.conflict_at_step(S_t_next_dict, aircraft_id, t) if t != self.T else aircraft_state['conflicts']
            aircraft_state['n_remaining_flights'] = self.num_remaing_flights(S_t_next_dict, aircraft_id, t)

        S_t_next = self.create_hashable_state_key(S_t_next_dict)
        # if not S_t_next in self.states:
        #     # if it is a newly expored state: calculated the intial value as function of t
        #     S_t_next_dict['value'] = [-self.cancellation_cost * self.num_conflicts(S_t_next_dict)]
        #     S_t_next_dict['value'] = [S_tx_dict['value'][-1]]
        #     S_t_next_dict['iteration'] = [n]

        return S_t_next_dict, S_t_next

    #################### VISUALIZE ###################
    def plot_values(self, value_dict, metric):
        # Extract iterations and corresponding objective values
        iterations = list(value_dict.keys())
        objective_values = list(value_dict.values())
        smoothing_method = 'Savgol'
        window_size = int(self.N / 5) if self.N > 15 else 4
        # Apply the smoothing method (Moving Average or Savitzky-Golay)
        if smoothing_method == 'MA':
            smoothed_values = np.convolve(objective_values, np.ones(window_size) / window_size, mode='valid')
            smoothed_iterations = iterations[:len(smoothed_values)]  # Adjusted iterations
        elif smoothing_method == 'Savgol':
            smoothed_values = savgol_filter(objective_values, window_length=window_size, polyorder=3)
            smoothed_iterations = iterations  # No need to trim iterations for Savitzky-Golay

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(iterations, objective_values, label=f'Original {metric}', alpha=0.8)
        plt.plot(smoothed_iterations, smoothed_values, label=f'Smoothed {metric}', linewidth=2)

        # Adding labels and title
        plt.xlabel('Iteration')
        plt.ylabel(f'{metric}')
        plt.title(f'{metric} - (ε={self.ε}, γ={self.y}, α={self.α}, p={self.p}), Z={self.avg_obj} {self.folder}')
        plt.grid(True)
        plt.legend()

        # Show the plot
        plt.show()

    def plot_schedule(self, state, iteration, instance):
        # Plotting
        plt.figure(figsize=(10, 5))

        # Get the states for the specified step
        step = state['t']

        # Flags to ensure 'Unavailability' is added to the legend only once
        au_label_added = False

        # Plot flights based on the stored order
        for aircraft_id in self.aircraft_ids[::-1]:
            aircraft_state = state.get(aircraft_id)

            if aircraft_state:
                if aircraft_state['flights']:
                    for flight in aircraft_state['flights']:
                        ADT = flight.get('ADT')
                        AAT = flight.get('AAT')
                        flight_nr = flight.get('Flightnr')

                        if ADT and AAT:
                            # Plot the flight
                            plt.plot([ADT, AAT], [aircraft_id, aircraft_id], marker='|', color='blue',
                                     linewidth=4, markersize=10)
                            # Calculate the midpoint of the flight for labeling
                            midpoint_time = ADT + (AAT - ADT) / 2
                            # Add the flight number as a label in the middle of the flight
                            plt.text(midpoint_time, aircraft_id, flight_nr,
                                     verticalalignment='bottom', horizontalalignment='center', fontsize=10,
                                     color='black')
                else:
                    # Plot a placeholder for aircraft with no flights assigned
                    plt.plot([self.recovery_start, self.recovery_end], [aircraft_id, aircraft_id], marker='|',
                             color='gray', linewidth=2, linestyle=':')
                    plt.text(self.recovery_start, aircraft_id, 'No Flights',
                             verticalalignment='bottom', horizontalalignment='left', fontsize=8, color='gray')

                # Plot Aircraft Unavailability (AU) from the state
                if 'UA' in aircraft_state and aircraft_state['UA']:
                    for unavailability in aircraft_state['UA']:
                        start_time, end_time = unavailability
                        label = 'Unavailability' if not au_label_added else ""
                        plt.plot([start_time, end_time], [aircraft_id, aircraft_id],
                                 linestyle='--', color='orange', linewidth=2, label=label)
                        plt.scatter([start_time, end_time], [aircraft_id, aircraft_id],
                                    color='orange', marker='x', s=100)  # Markers for AU disruption start and end
                        au_label_added = True  # Only add the label once for 'AU'

        # Retrieve the current time associated with the step
        current_time = self.periods[step] if step < len(self.periods) else self.recovery_end

        # Plot the current time as a vertical line (only once)
        plt.axvline(x=current_time, color='black', linestyle='-', linewidth=1, label='Current Time')

        # Plot recovery_start and recovery_end as vertical lines (only once)
        plt.axvline(x=self.recovery_start, color='purple', linestyle='--', linewidth=1, label='Recovery Start')
        plt.axvline(x=self.recovery_end, color='purple', linestyle='--', linewidth=1, label='Recovery End')

        plt.xlabel('Time')
        plt.ylabel('Aircraft')
        plt.title(f'Flight Schedule: t= {step}, n={iteration}, {instance}')
        plt.grid(True)

        # Format x-axis to show only time
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)  # Rotate x-axis labels to 45 degrees

        plt.legend()
        plt.show()

    def print_state(self, state):
        print(f't: {state['t']}')
        for aircraft_id in self.aircraft_ids:
            print(f'\t-{aircraft_id}')
            for key, value in state[aircraft_id].items():
                print(f'\t\t-{key}: {value}')
        print(f'\t-Value = {state['value']}')
        print(f'\t-Iterations {state['iteration']}')
        state_key = self.create_hashable_state_key(state)
        print(state_key)
        print()

    def deterministic_disruptions(self, n):
        if n == 1:
            disruptions = dict()
            for t in self.steps:
                disruptions[t] = dict()
                for ac in self.aircraft_ids:
                    disruptions[t][ac] = []

                for disruption in self.disruptions:
                    ac = disruption['Aircraft']
                    Start, End = disruption['StartTime'], disruption['EndTime']
                    disruptions[t][ac].append((Start, End))

            self.disruptions = disruptions
            return self.disruptions
        else:
            return self.disruptions

def save_instance(data, filename):
    """Save the policy to a binary file using pickle."""
    policy_folder = "policies"
    if not os.path.exists(policy_folder):
        os.makedirs(policy_folder)

    policy_file = os.path.join(policy_folder, f"{filename}.pkl")

    with open(policy_file, 'wb') as f:
        pickle.dump(data, f)  # Serialize and save the policy
    print(f"Policy saved at {policy_file}")

def load_data(filename):
    """Load a previously saved policy from a pickle file."""
    policy_file = os.path.join("../policies", f"{filename}.pkl")

    if not os.path.exists(policy_file):
        raise FileNotFoundError(f"No saved policy found")

    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)  # Deserialize and load the policy

    return policy

def count_disruptions(disruptions):
    for t in disruptions:
        empty = False
        # Check if all disruption lists for all aircraft at time t are empty
        if all(not disruptions[t][aircraft] for aircraft in disruptions[t]):
            empty = True
    return 1 if empty else 0

def train_instance(instance_id):
    folder = f"TRAIN{instance_id}"
    agg_lvl = 2
    print(f"\nTraining for instance {instance_id} in folder {folder}")
    aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)
    m = VFA_ADP(aircraft_data, flight_data, disruptions, recovery_start, recovery_end, agg_lvl, folder)
    initial_state = m.initialize_state()
    m.train_with_vfa()

    return m.policy, m.states, m.agg_states, m.N

if __name__ == '__main__':
    nr_instances = 1
    agg_lvl = 2
    cumulative_policy = {}
    cumulative_states = {}
    cumulative_agg_states = {}

    start_time = time.time()
    # # Run training in parallel using ProcessPoolExecutor
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(train_instance, instance_id) for instance_id in range(1, nr_instances + 1)]
    #
    #     for future in concurrent.futures.as_completed(futures):
    #         policy, states, agg_states, N_iterations = future.result()
    #         # Merge policies from each instance into the cumulative policy
    #         cumulative_policy.update(policy)
    #         cumulative_states.update(states)
    #         cumulative_agg_states.update(agg_states)

    for _ in range(1, nr_instances + 1):
        folder = f"TRAIN{_}"
        print(f"\nTraining for instance {_} in folder {folder}")
        aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)
        m = VFA_ADP(aircraft_data, flight_data, disruptions, recovery_start, recovery_end, agg_lvl, folder)
        initial_state = m.initialize_state()
        m.train_with_vfa()

        N_iterations = m.N
        # Merge policies from this instance into the cumulative policy
        cumulative_policy.update(m.policy)
        cumulative_states.update(m.states)
        cumulative_agg_states.update(m.agg_states)

    end_time = time.time()
    T = end_time - start_time
    save_instance(cumulative_policy, "policy")
    save_instance(cumulative_states, 'states')
    save_instance(cumulative_agg_states, 'agg_states')

    print("Training done, states and policies saved.")
    print(f"TRAINED {nr_instances} INSTANCES IN {round((T), 2)} SECONDS")
    print(f'Solved with {round((N_iterations * nr_instances) /T , 2)} iterations per second')

    # policy = load_data('policy')
    # states = load_data('states')
    # agg_states = load_data('agg_states')

    # print(f'Loaded Policy:')
    # print(policy)
    # print('\n\n\n\n\n\n\n\n')
    #
    # print('\nAggregate State Keys:>>>>>>')
    # for key, value in cumulative_agg_states.items():
    #     print('Keys>> ', key)
    #     print('Values>> ', value)
    #
    # print('\nDisaggregate State Keys:>>>>>>')
    # for key, value in cumulative_states.items():
    #     print(key)

