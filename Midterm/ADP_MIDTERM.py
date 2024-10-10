from itertools import product
from old.environment import *
from disruptions import *
import os
import json
import random
import numpy as np
import time
from datetime import datetime, timedelta
import pickle
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
        num_flights = len(self.flight_data)
        num_aircraft = len(self.aircraft_data)

        self.interval = 60 # minutes
        self.intervals = pd.date_range(start=recovery_start, end=recovery_end, freq= str(self.interval)+'T')
        self.periods = {i: start for i, start in enumerate(self.intervals)}
        self.steps = [i for i in self.periods.keys()]
        self.period_length = pd.Timedelta(minutes=self.interval)
        self.total_recovery_time = (self.recovery_end - self.recovery_start).total_seconds() / 60  # Total recovery period in minutes

        self.T = self.steps[-1]
        self.N = 20 # number of iterations
        self.y = 0.8 # discount factor
        # self.a = 10 / self.N # learning rate or stepsize, fixed
        self.a = 0.3 # learning rate or stepsize, fixed
        self.epsilon = 0.5 # random state transition for exploration probabilty

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
        # initial_value = self.initial_value * (self.T - t) / self.T
        self.states = {}
        state_dict = dict()
        agg_dict = dict()


        state_dict['t'] = 0
        state_dict['time_elapsed'] = 0  # minutes
        for aircraft in self.aircraft_data:
            aircraft_state = {}
            aircraft_id = aircraft['ID']
            ass_flights = self.ass_flights(aircraft_id, t)
            unavailibilty = self.initialize_unavailable(aircraft_id, t)
            conflicts = self.initialize_conflict_at_step(ass_flights, unavailibilty, aircraft_id)
            remaining_flights = len(ass_flights)

            aircraft_state = {'conflicts': conflicts, 'UA': unavailibilty, 'flights':ass_flights, 'n_remaining_flights': remaining_flights}
            state_dict[aircraft_id] = aircraft_state

        initial_value = -1000 * self.num_conflicts(state_dict)

        # set value of the state to initial value and iteration to zero:
        state_dict['value'] = [initial_value]
        state_dict['iteration'] = [0]

        agg_dict['count'] = 0
        agg_dict['value'] = [initial_value]
        agg_dict['iteration'] = [0]

        state_key = self.create_hashable_state_key(state_dict)
        if self.aggregation_level == 2:
            aggregate_state_key = self.G_2(state_dict)
        if self.aggregation_level == 1:
            aggregate_state_key = self.G_1(state_dict)

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

    def initialize_conflict_at_step(self, ass_flights, unavailibilty, aircraft_id):
        conflicts = []
        for flight in ass_flights:
            for au in unavailibilty:
                if self.disrupted(au, flight):
                    conflicts.append(flight)
        return conflicts

    def intialize_has_flight_in_next_period(self, aircraft_id, t):
        """Check if the aircraft departs a flight in the next period. Return True/False and corresponding flight"""
        current_time = self.periods[t]
        periods = self.periods
        next_time = self.periods[t + 1] if t + 1 < self.T else None

        # If there's no next time (i.e., at the last step), return False
        if not next_time:
            return False, None

        aircraft = get_ac_dict(self.aircraft_data, aircraft_id)
        for flight_nr in aircraft['AssignedFlights']:
            flight = get_flight_dict(flight_nr, self.flight_data)
            if current_time <= flight['ADT'] < next_time:
                return True, flight

        return False, None

    def initialize_unavailable(self, aircraft_id, t):
        unavailabilities = []
        for disruption in self.disruptions:
            unavailability = {}
            if disruption['Aircraft'] == aircraft_id:
                unavailabilities.append( (disruption['StartTime'], disruption['EndTime']) )
        return unavailabilities

    def initialize_initial_values(self):
        pass

######### HELPER FUNCTIONS: #########
    def get_flight(self, flight_nr, current_state):
        for aircraft_id in self.aircraft_ids:
            aircraft_state = current_state[aircraft_id]
            for flight in aircraft_state['flights']:
                if flight['Flightnr'] == flight_nr:
                    return flight
        return 'None, Flight not found'

    #OLD FUNCTION - first level of aggregation
    def G_1(self, s):
        """
        Create a hashable aggregate key from a state dictionary that generalizes to unseen instances for testing purposes.

        The key includes:
        - n_remaining_flights per aircraft
        - rounded flight times to hours.

        Args:
            s (dict): A dictionary representing the state of the system.

        Returns:
            tuple: A hashable representation of the aggregate state.
        """
        aggregate_state_key = []

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

        # Convert the list to a tuple (to make it hashable)
        return tuple(aggregate_state_key)

    #NEW FUNCTION - second level of aggregation
    def G_2(self, s):
        """
        Create a hashable aggregate key from a state dictionary that generalizes to unseen instances for testing purposes.

        The key includes:
        - n_remaining_flights per aircraft
        - n_remaining_concflicts per aircraft

        Args:
            s (dict): A dictionary representing the state of the system.

        Returns:
            tuple: A hashable representation of the aggregate state.
        """
        aggregate_state_key = []

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
    def unavailable_for_flight(self, flight, aircraft_id):
        """ Checks if the aircraft is unavailable due to disruption at Departure time of flight"""
        for disruption in self.disruptions:
            if disruption["Type"] == "AU" and disruption['Aircraft'] == aircraft_id:
                return (disruption['StartTime'] <= flight['ADT'] <= disruption['EndTime'] or
                        flight['ADT'] <= disruption['StartTime'] <= flight['AAT'])

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

    def get_aircraft_actions(self, current_state, aircraft_id):
        current_time = self.periods[current_state['t']]
        aircraft_state = current_state[aircraft_id]
        actions = [('none', 'none', 'none')]
        # Iterate over all flights of the current aircraft
        for flight in aircraft_state['flights']:
            flight_nr = flight['Flightnr']
            flight_to_swap = self.get_flight(flight_nr, current_state)  # flight dict

            if flight_to_swap['ADT'] < current_time:
                continue

            # Consider swapping this flight to every other aircraft
            for other_aircraft_id in self.aircraft_ids:
                if other_aircraft_id == aircraft_id:
                    continue

                # 2. Check if the new aircraft can perform the flight
                if self.unavailable_for_flight(flight_to_swap, other_aircraft_id):  # flight_dict, str
                    continue  # Skip this swap if the new aircraft cannot perform the flight

                # If all checks pass, add this as a possible action
                actions += [('swap', flight_nr, other_aircraft_id)]

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

        next_state = temp_next_state
        # If no further conflicts, return the updated state and total delay
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

        reward -= canx * 1000

        # Penalties for performing a swap actions
        if action_type == 'swap':
            reward -= 10

            # Check if delays were necessary following the swaps:
            delay = self.check_delays(pre_decision_state, post_decision_state)
            reward -= delay
        return reward


################### SOLVE: ###################
    def train_with_vfa(self):
        """
        This function applies value function approximation to the realization.
        After each iteration n, the value is stored in the states.
        Note that the states therefore implicitly represent post-decision states.

        :return: n/a
        """
        # print('\n\n\n\nSolve with VFA:')
        # print("------------------------------------")
        # Track objective function value for each iteration
        objective_function_values = {}
        value_function_values = {}
        self.policy = {}
        agg_states_count = 1

        # iterations/episodes
        for n in range(1, int(self.N)+1):
            # print(f'##############################################################################################################################')
            # print(f'################################################### Iteration: {n} ###########################################################')
            # print(f'##############################################################################################################################')
            # initial state
            next_state = self.states[self.initial_state_key]
            omega_n = None
            initial_expected_value = next_state['value'][-1]
            accumulated_rewards = []

            for t in self.steps[:-1]:
                # print(f'\n##################### t: {t} ##################################################')
                S_t_dict = copy.deepcopy(next_state)
                S_t = self.create_hashable_state_key(S_t_dict)
                # print(f'Pre decision State:')
                # self.print_state(S_t_dict)

                action_values = {}
                immediate_rewards = {}
                for aircraft_id in self.aircraft_ids:
                    rewards = []
                    values = []
                    ac_actions = self.get_aircraft_actions(S_t_dict, aircraft_id)

                    # calculate immediate rewards for each a set at time t and store in action_values
                    for x in ac_actions:
                        # get next states for all possible actions
                        S_tx_dict = self.simulate_action_to_state(S_t_dict, x, t, n)
                        S_tx = self.create_hashable_state_key(S_tx_dict)

                        if self.aggregation_level == 2:
                            S_tx_g = self.G_2(S_tx_dict)
                        elif self.aggregation_level == 1:
                            S_tx_g = self.G_1(S_tx_dict)

                        R_t = self.compute_reward(S_t_dict, S_tx_dict, x)
                        V_sn = S_tx_dict['value'][-1]

                        action_values[x] = R_t + self.y * V_sn
                        immediate_rewards[x] = R_t

                x_star = max(action_values, key=action_values.get)
                self.policy[S_t] = x_star
                # print(f'\nBEST ACTION: ---------> {a_star}')
                v_star = action_values[x_star]
                best_immediate_reward =  immediate_rewards[x_star]
                accumulated_rewards.append(best_immediate_reward)
                # print(f'Best reward, total value: {best_immediate_reward}, {v_star}')


                S_tx_dict = self.apply_action_to_state(S_t_dict, x_star, t, n)
                S_tx = self.create_hashable_state_key(S_tx_dict)

                if self.aggregation_level == 2:
                    S_tx_g = self.G_2(S_tx_dict)
                elif self.aggregation_level == 1:
                    S_tx_g = self.G_1(S_tx_dict)


                # Update values of aggregated state:
                if not S_tx_g in self.agg_states:
                    self.agg_states[S_tx_g] = {'count': 1,
                                                'value': [S_tx_dict['value'][-1]],
                                                'iteration': [n],
                                               }
                else:
                    x = self.agg_states[S_tx_g]['count'] / agg_states_count
                    v_g = self.agg_states[S_tx_g]['value'][-1]
                    v_0 = S_tx_dict['value'][-1]
                    V_sn = (1 - x) * v_g + x * v_0

                    self.agg_states[S_tx_g]['value'].append(V_sn)
                    self.agg_states[S_tx_g]['count'] += 1
                    self.agg_states[S_tx_g]['iteration'].append(n)

                agg_states_count += 1

                # print(f'Post decision State:')
                # self.print_state(S_ta_dict)

                # update new value of the current state
                v_n_prev = self.states[S_t]['value'][-1]
                v_n_curr = (1 - self.a) * v_n_prev + self.a * v_star

                # Add newly calculated value and current iteration to state
                S_t_dict = self.states[S_t]
                self.states[S_t]['iteration'].append(n)
                self.states[S_t]['value'].append(v_n_curr)

                # print(f'Pre decision State with updated value:')
                # self.print_state(S_t_dict)

                # Add the post decisions state to states and update state for the next iteration:
                self.states[S_tx] = S_tx_dict
                next_state = S_tx_dict

                # ##### ADD EXOGENEOUS INFO HERE #######
                # S_t_next = S_tx

                #
                if n == self.N and next_state['t'] == self.T:
                    self.plot_schedule(next_state, n, folder)

            objective_value = sum(accumulated_rewards)
            if n == self.N:
                print(f'Objective value for iteration {n = }: {objective_value}')
                print(f'rewards: {accumulated_rewards}')
                print(f'Expected value of initial state for {n = }: {initial_expected_value}')

            # Store the objective value for this iteration
            objective_function_values[n] = objective_value
            value_function_values[n] = initial_expected_value
        self.plot_values(value_function_values)

    def simulate_action_to_state(self, current_state, action, t, n):
        '''Does the same as apply action to state, run when checking actions. Run apply_action_to_state for actually appliying actions'''
        # Create a copy of the current state to modify
        next_state = copy.deepcopy(current_state)
        current_state = copy.deepcopy(current_state)
        current_step = next_state['t']
        next_step = current_step + 1

        action_type, swapped_flight_nr, new_aircraft_id = action

        if action_type == 'swap':
            # 1. Swap the assignments of the aircraft for the flight
            old_aircraft_id = next((aircraft_id for aircraft_id, aircraft_state in current_state.items()
                                    if aircraft_id != 't' and
                                    aircraft_id != 'time_left' and
                                    aircraft_id != 'time_elapsed' and
                                    any(flight['Flightnr'] == swapped_flight_nr for flight in aircraft_state['flights'])), None)
            old_aircraft_state = current_state[old_aircraft_id]

            # Find the flight in the current aircraft's stateflight_nr
            flight_to_swap = next(flight for flight in old_aircraft_state['flights'] if flight['Flightnr'] == swapped_flight_nr)

            # Remove the flight from the old aircraft and assign it to the new aircraft
            next_state[new_aircraft_id]['flights'].append(flight_to_swap)
            next_state[old_aircraft_id]['flights'].remove(flight_to_swap)

            # 2 Check for overlaps and delay flights if possible:
            swapped_flight = flight_to_swap
            overlapping_flights = self.check_overlapping_assignments(next_state, new_aircraft_id, swapped_flight)
            if overlapping_flights != [] and self.delay_swapped_flight(next_state, new_aircraft_id, swapped_flight, overlapping_flights, apply=False) is not None:
                next_state = copy.deepcopy(self.delay_swapped_flight(next_state, new_aircraft_id, swapped_flight, overlapping_flights, apply=False))

        if action_type == 'none':
            # nothing happens to the assignments when doing nothing.
            pass

        for i, aircraft_id in enumerate(self.aircraft_ids):
            aircraft_state = next_state[aircraft_id]
            aircraft_state['conflicts'] = self.conflict_at_step(next_state, aircraft_id, next_step) if t != self.T else 0
            aircraft_state['n_remaining_flights'] = self.num_remaing_flights(next_state, aircraft_id, next_step)

        # update time for next state:
        next_state['t'] = next_step

        next_state['time_elapsed'] = next_state['time_elapsed'] + (self.interval / self.total_recovery_time)
        next_state_key = self.create_hashable_state_key(next_state)
        if not next_state_key in self.states:
            # if it is a newly expored state: calculated the intial value as function of t
            # downstream rewards gets closer to zero when less time on horizon
            # next_state['value'] = [self.initial_value * (self.T - next_step)  / self.T]
            next_state['value'] = [-1000 * self.num_conflicts(next_state)]
            # next_state['value'] = [0]
            next_state['iteration'] = [n]
        else:
            # if state is already explored, value list is same as already explored value list
            # if state is already explored, iteration list is same as already exlpored iteration list
            next_state['value'] = copy.deepcopy(self.states[next_state_key]['value'])
            next_state['iteration'] = copy.deepcopy(self.states[next_state_key]['iteration'])

        # Return the updated state as the next state
        return next_state

    def apply_action_to_state(self, current_state, action, t, n):
        """
        Apply the given action set to the current state and return the resulting next state.

        Args:
            current_state (dict): The current state of the system.
            action_set (tuple): The action set to apply.

        Returns:
            dict: The next state resulting from applying the action set.
        """
        '''Does the same as apply action to state, run when checking actions. Run apply_action_to_state for actually appliying actions'''
        # Create a copy of the current state to modify
        next_state = copy.deepcopy(current_state)
        current_state = copy.deepcopy(current_state)
        current_step = next_state['t']
        next_step = current_step + 1

        action_type, swapped_flight_nr, new_aircraft_id = action

        if action_type == 'swap':
            # 1. Swap the assignments of the aircraft for the flight
            old_aircraft_id = next((aircraft_id for aircraft_id, aircraft_state in current_state.items()
                                    if aircraft_id != 't' and
                                    aircraft_id != 'time_left' and
                                    aircraft_id != 'time_elapsed' and
                                    any(flight['Flightnr'] == swapped_flight_nr for flight in aircraft_state['flights'])), None)
            old_aircraft_state = current_state[old_aircraft_id]

            # Find the flight in the current aircraft's stateflight_nr
            flight_to_swap = next(flight for flight in old_aircraft_state['flights'] if flight['Flightnr'] == swapped_flight_nr)

            # Remove the flight from the old aircraft and assign it to the new aircraft
            next_state[new_aircraft_id]['flights'].append(flight_to_swap)
            next_state[old_aircraft_id]['flights'].remove(flight_to_swap)

            # 2 Check for overlaps and delay flights if possible:
            swapped_flight = flight_to_swap
            overlapping_flights = self.check_overlapping_assignments(next_state, new_aircraft_id, swapped_flight)
            if overlapping_flights != [] and self.delay_swapped_flight(next_state, new_aircraft_id, swapped_flight, overlapping_flights, apply=False) is not None:
                next_state = copy.deepcopy(self.delay_swapped_flight(next_state, new_aircraft_id, swapped_flight, overlapping_flights, apply=False))

        if action_type == 'none':
            # nothing happens to the assignments when doing nothing.
            pass

        for i, aircraft_id in enumerate(self.aircraft_ids):
            aircraft_state = next_state[aircraft_id]
            aircraft_state['conflicts'] = self.conflict_at_step(next_state, aircraft_id, next_step) if t != self.T else 0
            aircraft_state['n_remaining_flights'] = self.num_remaing_flights(next_state, aircraft_id, next_step)

        # update time for next state:
        next_state['t'] = next_step
        next_state['time_elapsed'] = next_state['time_elapsed'] + (self.interval / self.total_recovery_time)
        next_state_key = self.create_hashable_state_key(next_state)
        if not next_state_key in self.states:
            next_state['value'] = [-1000 * self.num_conflicts(next_state)]
            next_state['iteration'] = [n]
        else:
            next_state['value'] = copy.deepcopy(self.states[next_state_key]['value'])
            next_state['iteration'] = copy.deepcopy(self.states[next_state_key]['iteration'])

        # Return the updated state as the next state
        return next_state

    #################### VISUALIZE ###################
    def plot_values(self, value_dict):
        # Extract iterations and corresponding objective values
        iterations = list(value_dict.keys())
        objective_values = list(value_dict.values())

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, objective_values, linestyle='-', color='r', marker='x', label='Objective Value')

        # Adding labels and title
        plt.xlabel('Iteration')
        plt.ylabel('Expected Value of initial state')
        plt.title(f'Expected Value of initial state - evolution over iterations - {self.folder}')
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
        print(f'Relative time elapsed: {state['time_elapsed']}')
        for aircraft_id in self.aircraft_ids:
            print(f'\t-{aircraft_id}')
            for key, value in state[aircraft_id].items():
                print(f'\t\t-{key}: {value}')
        print(f'\t-Values = {state['value']}')
        print(f'\t-Iterations {state['iteration']}')
        state_key = self.create_hashable_state_key(state)
        print(state_key)


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

if __name__ == '__main__':
    nr_instances = 1
    agg_lvl = 2
    cumulative_policy = {}
    cumulative_states = {}
    cumulative_agg_states = {}

    start_time = time.time()
    for _ in range(1, nr_instances + 1):
        folder = f"TRAIN{_}"
        print(f"\nTraining for instance {_} in folder {folder}")

        aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)
        m = VFA_ADP(aircraft_data, flight_data, disruptions, recovery_start, recovery_end, agg_lvl, folder)

        initial_state = m.initialize_state()
        # m.plot_schedule(initial_state, iteration=0, instance=folder)
        m.train_with_vfa()

        # # Merge policies from this instance into the cumulative policy
        # cumulative_policy.update(m.policy)
        # cumulative_states.update(m.states)
        # cumulative_agg_states.update(m.agg_states)
    end_time = time.time()

    print('Cumulative policy:')
    print(cumulative_policy)
    # save_instance(cumulative_policy, "policy")
    # save_instance(cumulative_states, 'states')
    # save_instance(cumulative_agg_states, 'agg_states')
    #
    # policy = load_data('policy')
    # states = load_data('states')
    # agg_states = load_data('agg_states')

    # print(f'Loaded Policy:')
    # print(policy)
    # print('\n\n\n\n\n\n\n\n')

    print('\nAggregate State Keys:>>>>>>')
    for key, value in cumulative_agg_states.items():
        print('Keys>> ', key)
        print('Values>> ', value)

    print('\nDisaggregate State Keys:>>>>>>')
    for key, value in cumulative_states.items():
        print(key)

    # print('\nPolicy:>>>>>>>>')
    # for key, value in policy.items():
    #     print('Keys>> ', key)
    #     print('Values>> ', value)

    print("Training done, states and policies saved.")
    print(f"TRAINED {nr_instances} INSTANCES IN {round((end_time - start_time), 2)} SECONDS")