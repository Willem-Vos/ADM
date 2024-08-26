import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import copy
from preprocessing import *


class AircraftRecovery(gym.Env):
    def __init__(self, aircraft_data, flight_data, disruptions, recovery_start, recovery_end):
        super(AircraftRecovery, self).__init__()

        self.flight_data = flight_data
        self.aircraft_data = aircraft_data
        self.disruptions = disruptions

        # store initial data:
        self.initial_flight_data = flight_data
        self.intial_disruptions = disruptions
        self.initial_aircraft_data = aircraft_data

        # Recovery Window
        self.recovery_start = recovery_start
        self.recovery_end = recovery_end

        # Define action space sizes for swaps and delays:
        # Calculate the total action space size
        self.swap_action_space_size = len(flight_data) * len(aircraft_data)
        # delay options in minutes:
        self.delay_options = [20, 40, 60, 80, 100, 120, 140, 160, 180]
        self.delay_action_space_size = len(flight_data) * len(self.delay_options)
        # +1 for do nothing

        # Calculate the total action space size
        # self.action_space_size = self.swap_action_space_size + self.delay_action_space_size # SWAPS + DELAYS
        self.action_space_size = self.swap_action_space_size + 1  # ONLY SWAPS
        self.action_space = spaces.Discrete(self.action_space_size)
        # Observation space: has_ac_assigned, can_depart, ADT, AAT for each flight
        self.observation_space = spaces.Dict({
            'flights': spaces.Box(low=0, high=1, shape=(len(flight_data), 4), dtype=np.float32)
        })

        # Initialize state
        self.state = self._get_initial_state()


    def _get_initial_state(self):
        state = []
        for flight in self.flight_data:
            has_ac_assigned = 1.0 if flight['AssignedAircraft'] != 'None' else 0.0
            can_depart = self._can_depart(flight)
            if can_depart == 0.0:
                print(f'Flight {flight["Flightnr"]} with AC {flight['AssignedAircraft']} cannot depart')
            # ADT = flight['ADT'].hour + flight['ADT'].minute / 60.0 # convert time to fractional hours
            # AAT = flight['AAT'].hour + flight['AAT'].minute / 60.0 # convert time to fractional hours
            ADT = flight["ADT"]
            AAT = flight["AAT"]

            state.append([has_ac_assigned, can_depart, ADT, AAT])
        # return np.array(state, dtype=np.float32)
        return np.array(state)

    def step(self, action):

        # DO NOTHING:
        if action == self.action_space_size:
            reward = 0
            done = np.all(self.state[:, 1] == 1.0)  # Done if all flights can depart
            info = {}
            return self.state, reward, done, info

        #SWAP
        elif action < self.swap_action_space_size:
            action_type, flight_idx, aircraft_idx = self.decode_action(action)

            # Swap aircraft assignment for the selected flight
            selected_flight = self.flight_data[flight_idx] # flight_dict
            selected_aircraft = self.aircraft_data[aircraft_idx] # aircraft_dict

            previous_assigned_aircraft_id = selected_flight['AssignedAircraft'] # ac id string
            previous_ac_index = ac_index(self.aircraft_data, previous_assigned_aircraft_id)
            previous_ac = self.aircraft_data[previous_ac_index]  # aircraft dict

            # If aircraft is already assigned to this flight, do nothing
            if selected_flight['AssignedAircraft'] == selected_aircraft['ID']:
                reward = -1  # Penalize no-op action
            else:
                # Reassign aircraft and update aircraft and flight data
                selected_flight['AssignedAircraft'] = selected_aircraft['ID'] # Update the flight's assigned aircraft
                previous_ac['AssignedFlights'].remove(selected_flight['Flightnr'])  # remove flight from previous ac assigned flights
                selected_aircraft['AssignedFlights'].append(selected_flight['Flightnr']) # Add flight to new ac's assigned flights

                # Update state
                self.state[flight_idx, 0] = 1.0  # has_ac_assigned
                self.state[flight_idx, 1] = self._can_depart(selected_flight)  # can_depart

                # Reward for successful reassignment
                if self.state[flight_idx, 1] == 1.0:  # if flight can depart
                    reward = 10  # Positive reward for fixing the disruption
                else:
                    reward = -5  # Negative reward if the disruption is not fixed

            done = np.all(self.state[:, 1] == 1.0)  # Done if all flights can depart
            info = {}

            return self.state, reward, done, info

        # DELAY
        elif action < self.swap_action_space_size + self.delay_action_space_size:
            action_type, flight_idx, delay_idx = self.decode_action(action)
            delay_minutes = self.delay_options[delay_idx]
            flight_to_delay = self.flight_data[flight_idx]
            flight_to_delay_nr = flight_to_delay['Flightnr']

            # Delay flight:
            flight_to_delay['ADT'] = flight_to_delay['ADT'] + pd.Timedelta(minutes=delay_minutes)
            flight_to_delay['AAT'] = flight_to_delay['AAT'] + pd.Timedelta(minutes=delay_minutes)

            # Update state
            self.state[flight_idx, 2] =  flight_to_delay['ADT']  # ADT
            self.state[flight_idx, 3] =  flight_to_delay['AAT'] # AAT

            ######################### EDIT ######################################
            reward = 0
            done = np.all(self.state[:, 1] == 1.0)  # Done if all flights can depart
            info = {}
            return self.state, reward, done, info
            ######################### EDIT ######################################

    def reset(self):
        self.state = self._get_initial_state()
        return self.state

    def render(self, mode='human'):
        print("Current State:")
        for i, flight in enumerate(self.flight_data):
            print(f"Flight {flight['Flightnr']} - Aircraft: {flight['AssignedAircraft']}, State: {self.state[i]}")

    def translate_action(self, action):
        if action < self.swap_action_space_size:
            flight_idx = action // len(self.aircraft_data)
            aircraft_idx = action % len(self.aircraft_data)
            flight_nr = self.flight_data[flight_idx]["Flightnr"]
            aircraft_id = self.aircraft_data[aircraft_idx]['ID']
            return 'swap', flight_nr, aircraft_id

        elif action < self.swap_action_space_size + self.delay_action_space_size:
            delay_action = action - self.swap_action_space_size
            flight_idx = delay_action // len(self.delay_options)
            delay_idx = delay_action % len(self.delay_options)
            flight_nr = self.flight_data[flight_idx]["Flightnr"]
            delay_minutes = self.delay_options[delay_idx]
            return 'delay', flight_nr, delay_minutes
        else:
            return 'none', None, None

    def decode_action(self, action):
        if action < self.swap_action_space_size:
            flight_idx = action // len(self.aircraft_data)
            aircraft_idx = action % len(self.aircraft_data)
            return 'swap', flight_idx, aircraft_idx
        elif action < self.swap_action_space_size + self.delay_action_space_size:
            delay_action = action - self.swap_action_space_size
            flight_idx = delay_action // len(self.delay_options)
            delay_idx = delay_action % len(self.delay_options)
            return 'delay', flight_idx, delay_idx
        else:
            return 'none', None, None

    def encode_action(self, flight_idx=None, aircraft_idx=None, delay_idx=None):
        if flight_idx is None and aircraft_idx is None and delay_idx is None:
            # Return the "Do nothing" action
            return self.swap_action_space_size + self.delay_action_space_size

        if delay_idx is not None:
            # Encode a delay action
            return self.swap_action_space_size + (flight_idx * len(self.delay_options)) + delay_idx
        else:
            # Encode a swap action
            return flight_idx * len(self.aircraft_data) + aircraft_idx

    def _can_depart(self, flight):
        # Check if the flight can depart based on disruptions
        for disruption in self.disruptions:
            if (disruption["Type"] == "AU" and disruption['Aircraft'] == flight['AssignedAircraft']
                    and disruption['StartTime'] <= flight['ADT'] <= disruption['EndTime']):
                return 0.0
        return 1.0

    def overlaps(self, flight_to_swap, flight):
        """Checks if two flight times overlap"""
        return (flight_to_swap['ADT'] <= flight['ADT'] <= flight_to_swap['AAT'] or
                       flight_to_swap['ADT'] <= flight['AAT'] <= flight_to_swap['AAT'] or
                       (flight_to_swap['ADT'] <= flight['ADT'] and flight_to_swap['AAT'] >= flight['AAT']) or
                       (flight_to_swap['ADT'] >= flight['ADT'] and flight_to_swap['AAT'] <= flight['AAT'])
                       )

    def is_unavailable(self, flight_to_swap, aircraft_id):
        """ Checks if the aircraft is unavailable for flight due to disruption"""
        for disruption in self.disruptions:
            if disruption["Type"] == "AU" and disruption['Aircraft'] == aircraft_id:
                return (flight_to_swap['ADT'] <= disruption['StartTime'] <= flight_to_swap['AAT'] or
                        flight_to_swap['ADT'] <= disruption['EndTime'] <= flight_to_swap['AAT'] or
                        (flight_to_swap['ADT'] <= disruption['StartTime'] and flight_to_swap['AAT'] >= disruption[
                            'EndTime']) or
                        (flight_to_swap['ADT'] >= disruption['StartTime'] and flight_to_swap['AAT'] <= disruption[
                            'EndTime']))

    def can_be_delayed(self, flight_to_swap, flight):
        if flight_to_swap['AAT'] >= flight["ADT"]:
            overlap_time = flight_to_swap['AAT'] - flight['AAT']

    def has_delay_conflicts(self, flight_to_delay, delay_minutes):
        has_conflict = False
        flight_to_delay_nr = flight_to_delay['Flightnr']

        # Simulate delay
        flight_to_delay['ADT'] = flight_to_delay['ADT'] + pd.Timedelta(minutes=delay_minutes)
        flight_to_delay['AAT'] = flight_to_delay['AAT'] + pd.Timedelta(minutes=delay_minutes)

        # Check for recovery horizon:
        if flight_to_delay['AAT'] > self.recovery_end:
            has_conflict = True
            print(f'    Flight {flight_to_delay_nr} cannot be delayed beyond recovery window')
        else:
            assigned_aircraft = get_aircraft_dict(flight_to_delay['Flightnr'], self.aircraft_data)  # Get assigned aircraft dict
            for flight_nr in assigned_aircraft['AssignedFlights']:
                flight = get_flight_dict(flight_nr, self.flight_data)

                # Check if the delayed flight would conflict with any other assigned flights for the same aircraft
                if flight_nr != flight_to_delay_nr and self.overlaps(flight_to_delay, flight):
                    print(f'    Flight {flight_to_delay_nr} overlaps with Flight {flight_nr}')
                    has_conflict = True
                    print(f'    Flight {flight_to_delay_nr} with aircraft {assigned_aircraft['ID']} cannot be delayed by {delay_minutes} minutes')

                # Check if aircraft is available at new delayed times
                elif flight != flight_to_delay and self.is_unavailable(flight_to_delay, assigned_aircraft['ID']):
                    has_conflict = True
                    print(f'    Flight {flight_to_delay_nr} with aircraft {assigned_aircraft['ID']} cannot be delayed by {delay_minutes} minutes')

        return has_conflict

    def valid_actions(self):
        # DO NOTHING:
        valid_actions = []
        # valid_actions = [self.action_space_size]  # "do nothing" action is always valid

        for action in range(self.action_space_size - 1):  # -1 to exclude the "do nothing" action
            action_type, flight_idx, idx = self.decode_action(action)
            # SWAP
            if action_type == 'swap':
                flight_to_swap = self.flight_data[flight_idx]  # flight dict
                new_aircraft = self.aircraft_data[idx]  # aircraft dict

                # CHECK FOR CONFLICTS:
                has_overlapping_flights = False
                is_unavailable = False

                # 1. potential new aircraft has no overlapping flights with to be swapped flight
                for flight_nr in new_aircraft['AssignedFlights']:
                    flight = get_flight_dict(flight_nr, self.flight_data)  # Get flight info dict from flight number
                    if self.overlaps(flight_to_swap, flight):
                        has_overlapping_flights = True

                # 2. potential new aircraft is not unavailable for flight
                is_unavailable = self.is_unavailable(flight_to_swap, new_aircraft["ID"])

                if not has_overlapping_flights and not is_unavailable:
                    valid_actions.append(action)

            # DELAY
            elif action_type == 'delay':
                delay_minutes = self.delay_options[idx]
                flight_to_delay = copy.deepcopy(self.flight_data[flight_idx])  # create copy to check for feasibility first before changing ariival and departure times

                has_conflict = self.has_delay_conflicts(flight_to_delay, delay_minutes)
                if not has_conflict:
                    valid_actions.append(action)
                    print(f'>>>> flight {flight_to_delay['Flightnr']} can be delayed with {delay_minutes} minutes')

            # CANCEL
            elif action_type == 'cancel':
                pass

        return valid_actions



if __name__ == '__main__':
    folder = 'A01_small'

    # read data and update for disruptions:
    aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder) # read data
    aircraft_data, flight_data = update_data_post_disruptions(aircraft_data, flight_data, disruptions) # update for disruptions

    # Save copies of initial data:
    initial_aircraft_data, initial_flight_data,  initial_rotations_data, initial_disruptions,  recovery_start, recovery_end  = read_data(folder)
    initial_aircraft_data, initial_flight_data = update_data_post_disruptions(initial_aircraft_data, initial_flight_data, disruptions)

    # Create environment:
    env = AircraftRecovery(aircraft_data, flight_data, disruptions, recovery_start, recovery_end)
    state = env.reset()
    done = False
    env.render()
    iteration = 0
    plot_schedule(aircraft_data, flight_data, disruptions, iteration)
    while not done and iteration < 25:
        iteration += 1
        print(f'\n\n\nITERATION: {iteration}')
        print(env.aircraft_data)
        valid_actions = env.valid_actions()
        print(f'VALID ACTIONS: {valid_actions}')
        action = random.choice(valid_actions)
        print(f'Take action: {action} - {env.translate_action(action)}')
        state, reward, done, info = env.step(action)
        env.render()
        plot_schedule(aircraft_data, flight_data, disruptions, iteration)

    print(f'INITIAL DATA')
    print(initial_flight_data)
    print(initial_aircraft_data)
    print(f'RECOVERED DATA')
    print(env.flight_data)
    print(env.aircraft_data)
    plot_schedule(env.aircraft_data, env.flight_data, env.disruptions, iteration='Final')
