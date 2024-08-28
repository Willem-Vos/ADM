import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import copy
from itertools import product
from preprocessing import *
from environment import *
# TODO:
# Reward structure en initial state values revisen
#



class VFA_ADP:
    def __init__(self, aircraft_data, flight_data, disruptions, recovery_start, recovery_end):
        self.aircraft_data = aircraft_data
        self.flight_data = flight_data
        self.disruptions = disruptions
        self.aircraft_ids = [aircraft['ID'] for aircraft in self.aircraft_data]
        self.recovery_start = recovery_start
        self.recovery_end = recovery_end
        num_flights = len(self.flight_data)
        num_aircraft = len(self.aircraft_data)
        # Store the initial order of aircraft IDs
        self.aircraft_order = [aircraft['ID'] for aircraft in aircraft_data]

        # Set time interval for consequent steps
        interval = 60 # minutes
        self.intervals = pd.date_range(start=recovery_start, end=recovery_end, freq= str(interval)+'T')
        self.periods = {i: start for i, start in enumerate(self.intervals)}
        self.steps = [i for i in self.periods.keys()]
        self.T = self.steps[-1]
        self.period_length = pd.Timedelta(minutes=interval)


        self.N = 1 # number of iterations
        self.y = 1 # discount factor
        self.a = 0.02 # learning rate, or stepsize; fixed

        # States
        self.states = dict()

        # intial value is lowest possible value: All flights conflicted and all aircraft perform swap
        self.initial_value = -1000 * num_flights + (-10 * num_aircraft)
        self.initial_state(initial_value= self.initial_value)

        # copy of states to trace back policy and visualize solution:
        self.states_copy = copy.deepcopy(self.states)



# INITIALIZATION FUNCTIONS:
    def initial_state(self, initial_value):
        step = 0
        self.states = {}
        state_dict = dict()
        state_dict['t'] = 0
        for aircraft in self.aircraft_data:
            aircraft_state = {}
            aircraft_id = aircraft['ID']
            ass_flights = self.ass_flights(aircraft_id, step)
            conflict = self.initialize_conflict_at_step(aircraft_id, step)

            aircraft_state = {'id': aircraft_id, 'conflict': conflict, 'flights':ass_flights}
            state_dict[aircraft_id] = aircraft_state

        # set value of the state to initial value and iteration to zero:
        state_dict['value'] = [initial_value]
        state_dict['iteration'] = [0]

        state_key = self.create_hashable_state_key(state_dict)
        self.states[state_key] = state_dict

        return state_dict

    def ass_flights(self, aircraft_id, step):
        """Returns a list of copies flight dictionaries (with nr, adt, aat) assigned to an aircraft at a step, read from the data for intializing the states"""
        ass_flights = []
        flight_nrs = get_ac_dict(self.aircraft_data, aircraft_id)['AssignedFlights']
        for flight_nr in flight_nrs:
            flight = copy.deepcopy(get_flight_dict(flight_nr, self.flight_data))
            del flight["AssignedAircraft"]
            ass_flights.append(flight)
        return ass_flights

    def initialize_conflict_at_step(self, aircraft_id, step):
        if step > self.T:
            return 0
        conflict = False
        has_flight, flight = self.intialize_has_flight_in_next_period(aircraft_id, step)
        if flight:
            unavailable = self.unavailable_for_flight(flight, aircraft_id)

        if has_flight and unavailable:
            conflict = True

        if aircraft_id == 'A320#2' and step == 3:
            print(conflict)

        return 1 if conflict else 0

    def intialize_has_flight_in_next_period(self, aircraft_id, step):
        """Check if the aircraft departs a flight in the next period. Return True/False and corresponding flight"""
        current_time = self.periods[step]
        periods = self.periods
        next_time = self.periods[step + 1] if step + 1 < self.T else None

        # If there's no next time (i.e., at the last step), return False
        if not next_time:
            return False, None

        aircraft = get_ac_dict(self.aircraft_data, aircraft_id)
        for flight_nr in aircraft['AssignedFlights']:
            flight = get_flight_dict(flight_nr, self.flight_data)
            if current_time <= flight['ADT'] < next_time:
                return True, flight

        return False, None


######### HELPER FUNCTIONS: #########
    def get_flight(self, flight_nr, current_state):
        for aircraft_id in self.aircraft_ids:
            aircraft_state = current_state[aircraft_id]
            for flight in aircraft_state['flights']:
                if flight['Flightnr'] == flight_nr:
                    return flight
        return 'None, Flight not found'

    def create_hashable_state_key(self, state_dict):
        """
        Create a hashable key from a state dictionary.

        Args:
            state_dict (dict): A dictionary representing the state of the system.

        Returns:
            tuple: A hashable representation of the state.
        """
        state_key = []

        # Add the time step to the key
        state_key.append(state_dict['t'])

        # Add each aircraft's state to the key
        for aircraft_id in self.aircraft_ids:
            aircraft_state = state_dict[aircraft_id]
            # Convert the flights list to a tuple of tuples (each flight dict converted to a tuple)
            if aircraft_state['flights']:
                flights_tuple = tuple(
                    tuple(flight.items()) for flight in aircraft_state['flights']
                )
            else:
                flights_tuple = tuple()

            # Create a tuple for the current aircraft's state
            aircraft_state_key = (
                aircraft_state['id'],
                aircraft_state['conflict'],
                flights_tuple
            )

            # Add the aircraft state key to the list of keys
            state_key.append((aircraft_id, aircraft_state_key))

        # Convert the list of aircraft state keys to a tuple (making it hashable)
        return tuple(state_key)

    def create_state_dict_from_hashable_key(self, state_key):
        """
        Create a state dictionary from a hashable state key.

        Args:
            state_key (tuple): A hashable representation of the state.

        Returns:
            dict: A dictionary representing the state of the system.
        """
        # Initialize the state dictionary
        state_dict = {'t': state_key[0]}

        # Iterate over the aircraft states in the key
        for item in state_key[1:]:
            # Ensure the item is structured as expected
            if isinstance(item, tuple) and len(item) == 2:
                aircraft_id, aircraft_state_key = item

                # Ensure aircraft_state_key has the expected structure
                if len(aircraft_state_key) == 3:
                    aircraft_id_from_key, conflict, flights_tuple = aircraft_state_key

                    # Rebuild the flights list from the tuple
                    flights_list = [dict(flight) for flight in flights_tuple]

                    # Add the aircraft state back to the dictionary
                    state_dict[aircraft_id] = {
                        'id': aircraft_id_from_key,
                        'conflict': conflict,
                        'flights': flights_list
                    }
                else:
                    print(f"Unexpected structure for aircraft_state_key: {aircraft_state_key}")
                    raise ValueError(f"Unexpected structure for aircraft_state_key: {aircraft_state_key}")
            else:
                print(f"Unexpected item in state key: {item}")
                raise ValueError(f"Unexpected item in state key: {item}")

        state_dict['value'] = [0]
        state_dict['iteration'] = [0]
        return state_dict

########################### LOGIC ###########################
    def unavailable_for_flight(self, flight, aircraft_id):
        """ Checks if the aircraft is unavailable due to disruption at Departure time of flight"""
        for disruption in self.disruptions:
            if disruption["Type"] == "AU" and disruption['Aircraft'] == aircraft_id:
                return (disruption['StartTime'] <= flight['ADT'] <= disruption['EndTime'] or
                        flight['ADT'] <= disruption['StartTime'] <= flight['AAT'])

    def has_flight_in_next_period(self, aircraft_state, step):
        """Check if the aircraft departs a flight in the next period. Return True/False and corresponding flight"""
        current_time = self.periods[step]
        next_time = self.periods[step + 1] if step + 1 < len(self.steps) else None

        # If there's no next time (i.e., at the last step), return False
        if not next_time:
            return False, None

        for flight in aircraft_state['flights']:
            if current_time <= flight['ADT'] < next_time:
                return True, flight

        return False, None

    def conflict_at_step(self, aircraft_state, step):
        conflict = False
        aircraft_id = aircraft_state['id']
        has_flight, flight = self.has_flight_in_next_period(aircraft_state, step)
        if flight:
            unavailable = self.unavailable_for_flight(flight, aircraft_id)

        if has_flight and unavailable:
            conflict = True

        return 1 if conflict else 0

    def overlaps(self, flight_to_swap, flight):
        """Checks if two flight times overlap"""
        return (flight_to_swap['ADT'] <= flight['ADT'] <= flight_to_swap['AAT'] or
                flight_to_swap['ADT'] <= flight['AAT'] <= flight_to_swap['AAT'] or
                (flight_to_swap['ADT'] <= flight['ADT'] and flight_to_swap['AAT'] >= flight['AAT']) or
                (flight_to_swap['ADT'] >= flight['ADT'] and flight_to_swap['AAT'] <= flight['AAT'])
                )

    def get_individual_actions(self, aircraft_state, step):
        """ Returns a list of all possible actions for one aircraft, no restrictions yet"""
        possible_actions = [('none', 'none', 'none')]
        aircraft_id = aircraft_state['id']

        # Iterate over all flights of the current aircraft
        for flight in aircraft_state['flights']:
            flight_nr = flight['Flightnr']

            # Consider swapping this flight to every other aircraft
            for other_aircraft_id in self.aircraft_ids:
                if other_aircraft_id != aircraft_id:  # Exclude swapping to the same aircraft
                    possible_actions.append(('swap', flight_nr, other_aircraft_id))
        return possible_actions

    def get_valid_action_sets(self, current_state):
        step = current_state['t']

        # No action at last step:
        if step == self.T:
            return [tuple((('none', 'none', 'none') for _ in range(len(self.aircraft_ids))))]

        # Generate all possible actions for individual aircraft
        all_individual_actions = {}
        for aircraft_id in self.aircraft_ids:
            aircraft_state = current_state[aircraft_id]
            all_individual_actions[aircraft_id] = self.get_individual_actions(aircraft_state, step)

        # Generate all possible combinations of actions
        all_action_sets = list(product(*all_individual_actions.values()))

        # now check all combinations for validity
        valid_action_sets = []
        for action_set in all_action_sets:
            # Validate each action set
            if self.is_valid_action_set(action_set, current_state):
                valid_action_sets.append(action_set)

        return valid_action_sets

    def is_valid_action_set(self, action_set, current_state):
        step = current_state['t']
        swapped_flights = {}

        for action in action_set:
            action_type, flight_nr, new_aircraft_id = action

            if action_type == 'swap':
                flight_to_swap = self.get_flight(flight_nr, current_state) # flight dict
                new_aircraft_flights = current_state[new_aircraft_id]['flights'] # list of flight dicts

                # 1. Check for overlap with existing flights on the new aircraft
                for assigned_flight in new_aircraft_flights:
                    if self.overlaps(flight_to_swap, assigned_flight):
                        return False

                # 2. Check if the new aircraft can perform the flight
                if self.unavailable_for_flight(flight_to_swap, new_aircraft_id): # flight_dict, str
                    return False

                # 3. Track swapped flights to check for overlaps in future iterations
                if new_aircraft_id not in swapped_flights:
                    swapped_flights[new_aircraft_id] = []
                swapped_flights[new_aircraft_id].append(flight_to_swap)

        # 3. Check for overlaps in swapped flights
        for flights in swapped_flights.values():
            for i, flight1 in enumerate(flights):
                for flight2 in flights[i + 1:]:
                    if self.overlaps(flight1, flight2):
                        return False

        return True

    def compute_total_reward(self, current_state, action_set):
        total_reward = 0
        for aircraft_id, action in zip(self.aircraft_ids, action_set):
            aircraft_state = current_state[aircraft_id]
            individual_reward = self.compute_individual_reward(aircraft_state, action)
            total_reward += individual_reward
        return total_reward

    def compute_individual_reward(self, aircraft_state, action):
        reward = 0

        aircraft_id, conflict, flights = aircraft_state.values()
        action_type, flight_nr, new_aircraft_id = action

        if action_type == 'swap':
            reward += -10               # cost of swapping
            if conflict == 1:
                reward += 0             # conflict averted

            elif conflict == 0:
                reward += 0             # Avoid unnecessary swaps

        elif action_type == 'none':
            if conflict == 1:      # No action is taken when in conflict at next step
                reward -= 1000

            elif conflict == 0:    #
                reward += 0

        # if aircraft_state['id'] == 'B777#1' and action == ('swap', '4', 'B767#1') and conflict == 1:
        #     print('REWARD', reward)

        return reward


################### SOLVE: ###################
    def solve_with_vfa(self):
        """
        This function applies value function approximation to the realization.
        After each iteration n, the value is stored in the states.
        Note that the states therefore implicitly represent post-decision states.

        :return: n/a
        """

        print('\nSolve with VFA:')
        print("------------------------------------")
        # iterations or episodes
        for n in range(1, int(self.N) + 1):

            # initial state
            next_state = self.initial_state(self.initial_value)

            aqcuired_rewards = []
            for t in self.steps:
                print(f'\n##################### t: {t} #########################')
                # select current state (use a copy of the current state)
                current_state = next_state
                self.plot_schedule(current_state)
                print(f'\nCurrent state: {current_state}\n')

                if t == self.T:
                    objective_value = sum(aqcuired_rewards)
                    print(f'aqcuired rewards: {aqcuired_rewards}')
                    print(f'Objective value: {objective_value}')
                    break

                current_state_key = self.create_hashable_state_key(current_state)

                # Get possible action sets.
                valid_action_sets = self.get_valid_action_sets(current_state)

                # reward_per_move stores the reward (immediate and downstream) for all possible moves
                reward_per_action_set = {}
                immediate_rewards = {}
                # calculate immediate rewards for each action set at time t and store in reward_per_action_set
                for action_set in  valid_action_sets:
                    immediate_reward = self.compute_total_reward(current_state, action_set)
                    # print(f'{immediate_reward} <> {action_set}')
                    # Get the next state following the action set applied:
                    post_decision_state = self.apply_action_set_to_state(current_state, action_set)
                    post_decision_state_key = self.create_hashable_state_key(post_decision_state)

                    if post_decision_state_key in self.states:
                        downstream_reward = self.states[post_decision_state_key]['value'][-1]
                    else:
                        # downstream rewards gets closer to zero when less time on horizon
                        downstream_reward = self.initial_value * (self.T - t) / self.T

                    reward_per_action_set[action_set] = immediate_reward + self.y * downstream_reward  # T EN ACTION ALS KEYS? DOET DE TIJD ERTOE?
                    immediate_rewards[action_set] = immediate_reward


                # Get best action set and following state
                best_action_set = max(reward_per_action_set, key=reward_per_action_set.get)
                post_decision_state = self.apply_action_set_to_state(current_state, best_action_set)
                post_decision_state_key = self.create_hashable_state_key(post_decision_state)
                print(f'best action set: {best_action_set}')
                print(f'Best immediate and downstream reward: {reward_per_action_set[best_action_set]}')


                # get the immediate reward corresponding to the best action set and keep track of acquired rewards
                best_immediate_reward = immediate_rewards[best_action_set]
                aqcuired_rewards.append(best_immediate_reward)
                print(f'Best immediate reward: {best_immediate_reward}')

                # calculate new approximate value of the current state
                current_values = self.states[current_state_key]['value']
                current_value = self.states[current_state_key]['value'][-1]
                if len(current_values) > 1:
                    print("First Value iteration for state:")
                    print(f'new_value = (1 - {self.a}) * {current_value} + {self.a} * {reward_per_action_set[best_action_set]}')
                    new_value = (1 - self.a) * current_value + self.a * reward_per_action_set[best_action_set]

                else:
                    new_value = reward_per_action_set[best_action_set]


                # Add newly calculated value and current iteration to state
                print(f'Updated approximate value = {new_value}')
                self.states[current_state_key]['iteration'].append(n)
                self.states[current_state_key]['value'].append(new_value)

                print(f'Pre decision State: {current_state}')
                print(f'Post decision State: {post_decision_state}')


                # Add the post decisions state to states and update state for the next iteration:
                self.states[post_decision_state_key] = post_decision_state
                next_state = post_decision_state

                # # transition
                # if t < self.last_period:
                #     # select greedy move (if epsilon > 0)
                #     greedy_move = random.choice(self.nodes)
                #     move = random.choices(population=[best_move, greedy_move],
                #                           weights=[1 - self.epsilon, self.epsilon])[0]

    def apply_action_set_to_state(self, current_state, action_set):
        """
        Apply the given action set to the current state and return the resulting next state.

        Args:
            current_state (dict): The current state of the system.
            action_set (tuple): The action set to apply.

        Returns:
            dict: The next state resulting from applying the action set.
        """
        # Create a copy of the current state to modify
        next_state = copy.deepcopy(current_state)
        current_step = next_state['t']

        # MODIFY NEXT STATE:
        # Apply each action in the action set to the corresponding aircraft
        for action, aircraft_id in zip(action_set, self.aircraft_ids):
            action_type, flight_nr, new_aircraft_id = action
            aircraft_state = next_state[aircraft_id]

            if action_type == 'swap':
                old_aircraft_id = aircraft_id

                # Find the flight in the current aircraft's state
                flight_to_swap = next(flight for flight in aircraft_state['flights'] if flight['Flightnr'] == flight_nr)

                # Remove the flight from the old aircraft and assign it to the new aircraft
                next_state[new_aircraft_id]['flights'].append(flight_to_swap)
                next_state[old_aircraft_id]['flights'].remove(flight_to_swap)

            if action_type == 'none':
                # nothing happens to the assignments when doing nothing.
                pass

        # Check new assignments for conflicts in next step and current step and update
        for aircraft_id in self.aircraft_ids:
            aircraft_state = next_state[aircraft_id]
            aircraft_state['conflict'] = self.conflict_at_step(aircraft_state, current_step+1)

        # Lastly, set the time of the new state to the next step:
        next_state['t'] = current_step + 1

        # Return the updated state as the next state
        return next_state


    #################### VISUALIZE ###################
    def plot_schedule(self, state, action=None):
        # Plotting
        plt.figure(figsize=(10, 5))

        # Get the states for the specified step
        step = state['t']
        # Plot flights based on the stored order
        for aircraft_id in self.aircraft_ids:
            aircraft_state = state.get(aircraft_id)

            if aircraft_state:
                for flight in aircraft_state['flights']:
                    ADT = flight.get('ADT')
                    AAT = flight.get('AAT')
                    flight_nr = flight.get('Flightnr')

                    if ADT and AAT:
                        # Plot the flight
                        plt.plot([ADT, AAT], [aircraft_id, aircraft_id], marker='o', color='blue',
                                 linewidth=3.5)
                        # Calculate the midpoint of the flight for labeling
                        midpoint_time = ADT + (AAT - ADT) / 2
                        # Add the flight number as a label in the middle of the flight
                        plt.text(midpoint_time, aircraft_id, flight_nr,
                                 verticalalignment='bottom', horizontalalignment='center', fontsize=10, color='black')

                # Plot disruptions
                for disruption in self.disruptions:
                    if disruption["Type"] == 'AU' and aircraft_id == disruption["Aircraft"]:
                        start_time = disruption['StartTime']
                        end_time = disruption['EndTime']
                        plt.plot([start_time, end_time], [disruption['Aircraft'], disruption['Aircraft']],
                                 linestyle='--',
                                 color='orange', linewidth=2)  # Dashed orange line for disruption
                        plt.scatter([start_time, end_time], [disruption['Aircraft'], disruption['Aircraft']],
                                    color='orange',
                                    marker='x', s=100)  # Markers for disruption start and end

                    elif (disruption["Type"] == 'Delay' and
                          flight_nr in [f['Flightnr'] for f in aircraft_state['flights']]):
                        delay_minutes = int(disruption['Delay'])
                        if flight_nr in [f['Flightnr'] for f in aircraft_state['flights']]:
                            ADT = flight['ADT']
                            AAT = flight['AAT']
                            if ADT and AAT:
                                plt.plot([ADT, ADT + pd.Timedelta(minutes=delay_minutes)], [aircraft_id, aircraft_id],
                                         linestyle='--',
                                         color='red', linewidth=2)  # Dashed red line for Delay disruption
                                plt.scatter([ADT, ADT + pd.Timedelta(minutes=delay_minutes)],
                                            [aircraft_id, aircraft_id], color='red',
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
        plt.title(f'Aircraft Flight Schedule: step: {step} - action: {action}')
        plt.grid(True)

        # Format x-axis to show only time
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)  # Rotate x-axis labels to 45 degrees

        plt.legend()
        plt.show()

    def print_states(self):
        print('STATES:')
        for step in self.steps:
            print(f'step {step}')
            for aircraft_id, aircraft_state in self.states[step].items():
                print(f'\t{aircraft_id}: {aircraft_state}')
        print()

    def print_state(self, state_key):
        state = self.states[state_key]
        print(f'-------- Step {state['t']} -----------')
        for aircraft_id in self.aircraft_ids:
            print(f'\t{aircraft_id}: {state[aircraft_id]}')
        print(f'\tValue = {state['value'][-1]}')
        print(f'\tIteration {state['iteration'][-1]}')
        print()


if __name__ == '__main__':
    # folder = 'A01_small'
    # folder = 'A01_small2'
    folder = 'A01_mini'
    aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)
    m = VFA_ADP(aircraft_data, flight_data, disruptions, recovery_start, recovery_end)
    initial_state = m.initial_state(m.initial_value)
    for state_key, state in m.states.items():
        m.print_state(state_key)
    print(m.states)
    m.solve_with_vfa()
    # m.plot_schedule(initial_state)