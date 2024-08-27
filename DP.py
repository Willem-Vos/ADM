import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import copy
from itertools import product
from preprocessing import *
from environment import *

class DisruptionRealization:
    def __init__(self, aircraft_data, flight_data, disruptions, recovery_start, recovery_end):
        self.aircraft_data = aircraft_data
        self.flight_data = flight_data
        self.disruptions = disruptions
        self.aircraft_ids = [aircraft['ID'] for aircraft in self.aircraft_data]
        self.recovery_start = recovery_start
        self.recovery_end = recovery_end

        # Store the initial order of aircraft IDs
        self.aircraft_order = [aircraft['ID'] for aircraft in aircraft_data]

        # Set time interval for consequent steps
        interval = 60 # minutes
        self.intervals = pd.date_range(start=recovery_start, end=recovery_end, freq= str(interval)+'T')
        self.periods = {i: start for i, start in enumerate(self.intervals)}
        self.steps = [i for i in self.periods.keys()]
        self.last_step = self.steps[-1]
        self.period_length = pd.Timedelta(minutes=interval)

        self.y = 1 # discount factor

        # States
        self.states = dict()
        self.initial_states()
        # copy of states to trace back policy and visualize solution:
        self.states_copy = copy.deepcopy(self.states)

    def initial_states(self):
        for step in self.steps:
            self.states[step] = {}
            for aircraft in self.aircraft_data:
                aircraft_dict = {}
                aircraft_id = aircraft['ID']
                available = 1
                ass_flights = self.ass_flights(aircraft_id, step)
                conflict_next = self.initialize_conflict_at_step(aircraft_id, step)

                aircraft_dict = {'id': aircraft_id, 'conflict_next': conflict_next, 'flights':ass_flights}
                self.states[step][aircraft_id] = aircraft_dict


# INITIALIZATION FUNCTIONS:
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
        if step > self.last_step:
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
        next_time = self.periods[step + 1] if step + 1 < self.last_step else None

        # If there's no next time (i.e., at the last step), return False
        if not next_time:
            return False, None

        aircraft = get_ac_dict(self.aircraft_data, aircraft_id)
        for flight_nr in aircraft['AssignedFlights']:
            flight = get_flight_dict(flight_nr, self.flight_data)
            if current_time <= flight['ADT'] < next_time:
                return True, flight

        return False, None


######### HELPER FUNCTIONS:
    def get_flight(self, flight_nr, step):
        for aircraft_id, aircraft_state in self.states[step].items():
            for flight in aircraft_state['flights']:
                if flight['Flightnr'] == flight_nr:
                    return flight

    def unavailable_for_flight(self, flight, aircraft_id):
        """ Checks if the aircraft is unavailable due to disruption at Departure time of flight"""
        for disruption in self.disruptions:
            if disruption["Type"] == "AU" and disruption['Aircraft'] == aircraft_id:
                return (disruption['StartTime'] <= flight['ADT'] <= disruption['EndTime'] or
                        flight['ADT'] <= disruption['StartTime'] <= flight['AAT'])

    def has_flight_in_next_period(self, aircraft_id, step):
        """Check if the aircraft departs a flight in the next period. Return True/False and corresponding flight"""
        current_time = self.periods[step]
        next_time = self.periods[step + 1] if step + 1 < len(self.steps) else None

        # If there's no next time (i.e., at the last step), return False
        if not next_time:
            return False, None

        for flight in self.states[step][aircraft_id]['flights']:
            if current_time <= flight['ADT'] < next_time:
                return True, flight

        return False, None

    def conflict_at_step(self, aircraft_id, step):
        conflict = False
        has_flight, flight = self.has_flight_in_next_period(aircraft_id, step)
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

    def get_individual_actions(self, aircraft_id, step):
        """ Returns a list of all possible actions for one aircraft, no restrictions yet"""
        possible_actions = [('none', 'none', 'none')]
        aircraft_state = self.states[step][aircraft_id]
        current_flights = aircraft_state['flights']

        # Iterate over all flights of the current aircraft
        for flight in current_flights:
            flight_nr = flight['Flightnr']

            # Consider swapping this flight to every other aircraft
            for other_aircraft_id in self.aircraft_ids:
                if other_aircraft_id != aircraft_id:  # Exclude swapping to the same aircraft
                    possible_actions.append(('swap', flight_nr, other_aircraft_id))
        return possible_actions

    def get_valid_action_sets(self, step):
        if step == self.last_step:
            return [tuple((('none', 'none', 'none') for _ in range(len(self.aircraft_ids))))]

        all_individual_actions = {}

        # Generate possible actions for each aircraft
        for aircraft_id in self.aircraft_ids:
            all_individual_actions[aircraft_id] = self.get_individual_actions(aircraft_id, step)

        # Generate all possible combinations of actions
        all_action_sets = list(product(*all_individual_actions.values()))

        valid_action_sets = []

        for action_set in all_action_sets:
            # Validate each action set
            if self.is_valid_action_set(action_set, step):
                valid_action_sets.append(action_set)

        return valid_action_sets

    def is_valid_action_set(self, action_set, step):
        swapped_flights = {}

        for action in action_set:
            action_type, flight_nr, new_aircraft_id = action

            if action_type == 'swap':
                flight_to_swap = self.get_flight(flight_nr, step)
                new_aircraft_flights = self.states[step][new_aircraft_id]['flights']

                # 1. Check for overlap with existing flights on the new aircraft
                for assigned_flight in new_aircraft_flights:
                    if self.overlaps(flight_to_swap, assigned_flight):
                        return False

                # 2. Check if the new aircraft can perform the flight
                if self.unavailable_for_flight(flight_to_swap, new_aircraft_id):
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

    def create_hashable_state_key(self, state_dict):
        state_key = []
        for aircraft_id, aircraft_state in state_dict.items():
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
                aircraft_state['conflict_next'],
                flights_tuple
            )

            # Add the aircraft state key to the list of keys
            state_key.append((aircraft_id, aircraft_state_key))

        # Convert the list of aircraft state keys to a tuple (making it hashable)
        return tuple(state_key)

    def create_state_dict_from_hashable_key(self, state_key):
        state_dict = {}

        for aircraft_id, aircraft_state_key in state_key:
            aircraft_id, conflict_next, flights_tuple = aircraft_state_key

            # Convert the flights tuple back into a list of flight dictionaries
            flights_list = [
                dict(flight_tuple) for flight_tuple in flights_tuple
            ]

            # Reconstruct the aircraft state dictionary
            aircraft_state = {
                'id': aircraft_id,
                'conflict_next': conflict_next,
                'flights': flights_list
            }

            # Add the aircraft state back into the state_dict
            state_dict[aircraft_id] = aircraft_state

        return state_dict

    def compute_total_reward(self, current_state, action_set):
        total_reward = 0
        for aircraft_id, action in zip(current_state, action_set):
            aircraft_state = current_state[aircraft_id]
            individual_reward = self.compute_individual_reward(aircraft_state, action)
            total_reward += individual_reward
        return total_reward

    def compute_individual_reward(self, aircraft_state, action):
        reward = 0

        aircraft_id, conflict_next, flights = aircraft_state.values()
        action_type, flight_nr, new_aircraft_id = action

        if action_type == 'swap':
            reward += -10               # cost of swapping
            if conflict_next == 1:
                reward += 0             # conflict averted

            elif conflict_next == 0:
                reward += 0             # Avoid unnecessary swaps

        elif action_type == 'none':
            if conflict_next == 1:      # No action is taken when in conflict at next step
                reward -= 1000

            elif conflict_next == 0:    #
                reward += 0

        # if aircraft_state['id'] == 'B777#1' and action == ('swap', '4', 'B767#1') and conflict_next == 1:
        #     print('REWARD', reward)

        return reward

############## SOLVE
    def solve_dp(self):

        print(f'\n\n\n')
        # Initialize the value function for the last step (base case)
        self.value_function = {}
        self.policy = {}

        # Track the most up-to-date state after each step is processed
        last_processed_state = None

        for step in reversed(self.steps):
            current_state = copy.deepcopy(self.states[step])
            print()
            print(f'STATES PRE-ACTION - STEP {step}:')
            self.print_state(step)

            # Convert the dictionary into a tuple of (key, value) pairs
            current_state_key = self.create_hashable_state_key(current_state)

            # Initialize the value for this state
            if not current_state_key in self.value_function:
                self.value_function[current_state_key] = 0 if step == 0 else float('-inf')
            best_action_set = None

            # Get all possible action combinations for this state
            possible_action_sets = self.get_valid_action_sets(step)

            print(f'\tTotal values, actions:')
            for action_set in possible_action_sets:

                # Evaluate the action set (this will simulate the next state without modifying the actual state)
                total_value = self.evaluate_action_set(current_state_key, action_set, step)
                print(f'\t{total_value} <> {action_set}')

                # Update the value function and best action if this action set is better
                if total_value > self.value_function[current_state_key]:
                    self.value_function[current_state_key] = total_value
                    best_action_set = action_set
                    best_value = total_value

            if not best_action_set:
                print(f'\tNo improvement')
                best_action_set = (('none', 'none', 'none'),
                                   ('none', 'none', 'none'),
                                   ('none', 'none', 'none'),
                                   ('none', 'none', 'none'))
            print()
            print(f'\tBEST ACTION SET AT STEP {step}')
            print(f'\t----->>>>>>{best_action_set}')

            # Store the best action in the policy
            self.policy[step] = best_action_set

            if step > 0:
                # apply swaps to next state
                self.apply_swaps_to_states(step, best_action_set)

                # Check new assignments for conflicts in next (previous) step and current step
                for aircraft_id, aircraft_state in self.states[step-1].items():
                    aircraft_state['conflict_next'] = self.conflict_at_step(aircraft_state['id'], step-1)

            elif step == 0:
                # apply swaps to next state
                self.apply_swaps_to_states(step, best_action_set)

                # Check new assignments for conflicts in next (previous) step and current step
                for aircraft_id, aircraft_state in self.states[step].items():
                    aircraft_state['conflict_next'] = self.conflict_at_step(aircraft_state['id'], step)

            print()
            print(f'STATES POST-ACTION - STEP {step}:')
            self.print_state(step)
        print(f'\n\n\n')

    def get_future_value(self, next_state_key, action_set):
        # Return the stored value for the next state from the value function
        # If the next state has not been evaluated yet, return 0 as its value
        return self.value_function.get(next_state_key, 0)

    def simulate_action_set_for_evaluation(self, state_key, action_set, step):
        # Create a deep copy of the current system state to simulate changes without affecting the actual state
        current_system_state = self.create_state_dict_from_hashable_key(state_key)
        temp_system_state = copy.deepcopy(current_system_state)

        # Collect all changes that need to be evaluated simultaneously
        changes = []

        for action, aircraft_id in zip(action_set, self.aircraft_ids):
            change = self.collect_action_change(temp_system_state[aircraft_id], action, temp_system_state, step)
            if change:
                changes.append(change)

        # Apply all changes to the temporary state simultaneously
        for change in changes:
            self.apply_collected_change(temp_system_state, change, step)

        # Return the new state key representing the updated system state after evaluation
        return self.create_hashable_state_key(temp_system_state)

    def evaluate_action_set(self, state_key, action_set, step):
        if step == self.last_step:
            # If at the last step, there's no future value to add
            total_value = 0
        else:
            # Simulate the action set on a temporary state and return the resulting state key
            next_state_key = self.simulate_action_set_for_evaluation(state_key, action_set, step)
            state = self.create_state_dict_from_hashable_key(state_key)
            reward = self.compute_total_reward(state, action_set)

            # Compute future value for the state key obtained from the simulation
            future_value = self.get_future_value(next_state_key, action_set)
            total_value = reward + self.y * future_value

        return total_value

    def simulate_action_set(self, state, action_set, step):
        """
        Simulates applying the given action set to the current state and returns the resulting state.
        This modifies the state as the actions are applied.
        """
        next_system_state = copy.deepcopy(state)

        # Apply each action simultaneously
        changes = []
        for action, aircraft_id in zip(action_set, self.aircraft_ids):
            change = self.collect_action_change(next_system_state[aircraft_id], action, next_system_state, step)
            if change:
                changes.append(change)

        # Apply all changes simultaneously
        for change in changes:
            self.apply_collected_change(next_system_state, change, step)

        return next_system_state

    def apply_action_set_to_states(self, step, action_set):
        """
        Applies the action set to the state at the given step, updating the state for the next step.
        """
        # Deep copy the current state to modify it for the next step
        current_state = copy.deepcopy(self.states[step])

        # Collect all changes that need to be applied simultaneously
        changes = []
        for action, aircraft_id in zip(action_set, self.aircraft_ids):
            change = self.collect_action_change(current_state[aircraft_id], action, current_state, step)
            if change:
                changes.append(change)

        # Apply all changes to the state simultaneously
        for change in changes:
            self.apply_collected_change(current_state, change, step)

        # Update the states for the next step in backward (so previous step)
        next_step = step -1
        if next_step >= 0:
            self.states[next_step] = current_state

    def apply_swaps_to_states(self, step, action_set):
        """
        Applies the action set of swaps to the state at the given step
        """
        # Deep copy the current state to modify it for the next step
        pre_decision_state = copy.deepcopy(self.states[step])
        post_decision_state = copy.deepcopy(self.states[step])

        # Apply the action set by modifying the flight assignments
        for action, aircraft_id in zip(action_set, self.aircraft_ids):
            action_type, flight_to_swap_nr, new_aircraft_id = action
            if action_type == 'swap':
                # Find the flight in the current aircraft's state
                flight_to_swap = next(
                    flight for flight in post_decision_state[aircraft_id]['flights'] if
                    flight['Flightnr'] == flight_to_swap_nr
                )
                # Remove the flight from the current aircraft
                post_decision_state[aircraft_id]['flights'].remove(flight_to_swap)
                # Add the flight to the new aircraft
                post_decision_state[new_aircraft_id]['flights'].append(flight_to_swap)

        # Update the states for the previous and current step
        if step >= 1:
            next_step = step - 1
            self.states[next_step] = post_decision_state
            self.states[step] = post_decision_state
        else:
            self.states[step] = post_decision_state


    def check_conflicts(self, current_state,  step):
        # Check for conflicts in the new assignments
        for aircraft_id in self.aircraft_ids:
            current_state[aircraft_id]['conflict_next'] = self.conflict_at_step(aircraft_id, step)

    def collect_action_change(self, aircraft_state, action, next_system_state, step):
        action_type, flight_to_swap_nr, new_aircraft_id = action
        old_aircraft_id = aircraft_state['id']

        if action_type == 'swap':
            # Find the flight in the current aircraft's state
            flight_to_swap = next(
                flight for flight in aircraft_state['flights'] if flight['Flightnr'] == flight_to_swap_nr)

            # Prepare the changes
            return {
                'remove_from': old_aircraft_id,
                'add_to': new_aircraft_id,
                'flight': flight_to_swap
                # Only include next_step if not at the last step
            }

        elif action_type == 'none':
            # At the last step, there's no next step, so handle accordingl
                return {
                    'update_step': old_aircraft_id
                }

    def apply_collected_change(self, next_system_state, change, step):
        if 'remove_from' in change and 'add_to' in change:
            # Get states
            old_aircraft_state = next_system_state[change['remove_from']]
            new_aircraft_state = next_system_state[change['add_to']]
            flight_to_swap = change['flight']

            # Remove the flight from the old aircraft and add it to the new aircraft
            old_aircraft_state['flights'].remove(flight_to_swap)
            new_aircraft_state['flights'].append(flight_to_swap)


            # Update conflict status for both aircraft
            if step == self.last_step:
                old_aircraft_state['conflict_next'] = 0
                new_aircraft_state['conflict_next'] = 0
            else:
                old_aircraft_state['conflict_next'] = self.conflict_at_step(old_aircraft_state['id'], step)
                new_aircraft_state['conflict_next'] = self.conflict_at_step(new_aircraft_state['id'], step)

        elif 'update_step' in change:
            # For 'none', just increment the step and update the conflict status
            aircraft_state = next_system_state[change['update_step']]

            if step == self.last_step:
                aircraft_state['conflict_next'] = 0
            else:
                aircraft_state['conflict_next'] = self.conflict_at_step(aircraft_state['id'], step)


###### VISUALIZATION #######
    def store_state(self, step, state):
        """Stores the current state for a given step."""
        if not hasattr(self, 'stored_states'):
            self.stored_states = {}  # Initialize a dictionary to store states

        # Store the state with the corresponding step as the key
        self.stored_states[step] = copy.deepcopy(state)

    def plot_schedule(self, step, action=None):
        # Plotting
        plt.figure(figsize=(10, 5))

        # Get the states for the specified step
        state = self.states_copy[step]

        # Plot flights based on the stored order
        for aircraft_id in self.aircraft_order:
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

    def visualize_policy(self):
        # Start from the initial step and iterate through all steps in the policy
        for step in self.steps:
            # Get the action set from the policy for the current step
            action_set = self.policy.get(step, None)

            # If there is an action set defined in the policy for this step, apply it
            if action_set:
                # Apply the action set to update the states
                self.apply_action_set_to_state_copy(step, action_set)

            # Plot the schedule after applying the action
            action_description = f'{action_set}' if action_set else 'None'
            self.plot_schedule(step, action_description)

    def apply_action_set_to_state_copy(self, step, action_set):
        """
        Applies the action set to the state at the given step, updating the state for the next step.
        """
        # Deep copy the current state to modify it for the next step
        current_state = copy.deepcopy(self.states_copy[step])

        # Collect all changes that need to be applied simultaneously
        changes = []
        for action, aircraft_id in zip(action_set, self.aircraft_ids):
            change = self.collect_action_change(current_state[aircraft_id], action, current_state, step)
            if change:
                changes.append(change)

        # Apply all changes to the state simultaneously
        for change in changes:
            self.apply_collected_change(current_state, change, step)

        # Update the states for the next step
        next_step = step + 1
        if next_step <= self.last_step:
            self.states_copy[next_step] = current_state

    def print_states(self):
        print('STATES:')
        for step in self.steps:
            print(f'step {step}')
            for aircraft_id, aircraft_state in self.states[step].items():
                print(f'\t{aircraft_id}: {aircraft_state}')
        print()

    def print_state(self, step):
        print(f'-------- Step {step} -----------')
        for aircraft_id, aircraft_state in self.states[step].items():
            print(f'\t{aircraft_id}: {aircraft_state}')
        print()


if __name__ == '__main__':
    folder = 'A01_small'
    # folder = 'A01_small2'
    # folder = 'A01_mini'
    aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)
    adm = DisruptionRealization(aircraft_data, flight_data, disruptions, recovery_start, recovery_end)
    print(adm.flight_data)
    print(adm.aircraft_data)
    adm.print_states()
    adm.solve_dp()
    print()
    print(adm.policy)
    adm.visualize_policy()
    print()
    print()
    print()
    for step, action in adm.policy.items():
        print(f'state: {step}')
        print(f'actions: {action}')