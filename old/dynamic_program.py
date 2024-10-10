from itertools import product
from old.environment import *

class DisruptionRealization:
    def __init__(self, aircraft_data, flight_data, disruptions, recovery_start, recovery_end):
        self.aircraft_data = aircraft_data
        self.flight_data = flight_data
        self.disruptions = disruptions
        self.aircraft_ids = sorted([aircraft['ID'] for aircraft in self.aircraft_data])
        self.recovery_start = recovery_start
        self.recovery_end = recovery_end

        # Store the initial order of aircraft IDs
        self.aircraft_order = [aircraft['ID'] for aircraft in aircraft_data]

        # Set time interval for consequent steps
        interval = 60 # minutes
        self.steps = pd.date_range(start=recovery_start, end=recovery_end, freq= str(interval)+'T')
        self.periods = {i: start for i, start in enumerate(self.steps)}
        self.period_length = pd.Timedelta(minutes=interval)

        self.y = 1 # discount factor

        # States
        self.states = dict()
        self.initial_state()

    def initial_state(self):
        step = 0
        for aircraft in self.aircraft_data:
            aircraft_id = aircraft['ID']
            available = 1 if self.is_available(aircraft_id, step) else 0
            next_flights = tuple(self.next_flights(aircraft_id, step))  # Convert list to tuple
            conflict = 1 if self.conflict_at_step(aircraft_id, step+1) else 0

            # State key: Tuple of initial state variables
            state_key = (aircraft_id, step, available, next_flights, conflict)
            self.states[state_key] = {
                'aircraft_id': aircraft_id,
                'step': step,
                'is_available': available,
                'next_flights': next_flights,
                'conflict_at_next_t': conflict,
                'iteration': [0],
                'value': [0]  # Initialize value function
            }

    def add_state(self, state_tuple, initial_value=0):
        """
                This functions adds a state to the realization

                :param state_tuple: the (aircraft_id, is_available, next_flights, conflicting_flight_at_next_t )-tuple identifies the state
                :param initial_value: the initial value of this (post decision) state
                :return: none
                """

        self.states[state_tuple] = dict(
            ID=state_tuple[0],
            step=state_tuple[1],
            next_flights=state_tuple[2],
            conflicting_flight_at_next_t=state_tuple[3],
            iteration=[0],  # list of numbers representing the iterations
            value=[initial_value] # list of numbers representing the value of the state after the corresponding iteration
        )
        """
        Example: 
            iteration=[0, 1, 2]
            value=[0, 20, 25]
            The value of the state is estimated with 0, 20, and 25 after iterations 0, 1, and 2. 
            Note that iteration "0" represents the initial value.
        """

    def get_state_key(self, aircraft_id, step):
        """
        Returns the key for the current state of the aircraft at a given step.
        """
        available = 1 if self.is_available(aircraft_id, step) else 0
        next_flights = tuple(self.next_flights(aircraft_id, step))
        conflict = 1 if self.conflict_at_step(aircraft_id, step+1) else 0
        return (aircraft_id, step, available, next_flights, conflict)

    def get_possible_actions(self, state_key):
        """
        Returns a list of possible actions for the given state.
        Actions are represented as (flight_to_swap, new_aircraft_id).
        """
        aircraft_id, step, available, next_flights, conflict = state_key
        possible_actions = [('none','none', 'none')]

        # SWAP:
        for flight_nr in next_flights:
            for new_aircraft in self.aircraft_data:
                new_aircraft_id = new_aircraft['ID']
                if new_aircraft_id != aircraft_id and self.can_swap(flight_nr, new_aircraft_id):
                    possible_actions.append(('swap', flight_nr, new_aircraft_id))

        # print(f'STATE: {state_key}')
        # for action in possible_actions:
        #     print(f'    {action}')
        return possible_actions

    def get_all_possible_actions(self, step):
        all_possible_actions = []

        # Generate all combinations of actions for all aircraft at the current step
        aircraft_actions = [self.get_possible_actions(self.get_state_key(aircraft['ID'], step)) for aircraft in
                            self.aircraft_data]

        # Use itertools.product to generate all possible combinations of actions
        for action_set in product(*aircraft_actions):
            if not self.has_overlapping_swapped_flights(action_set, step):
                all_possible_actions.append(action_set)

        return all_possible_actions

    def modify_state(self, state_key, step, states):
            state = states[state_key]
            aircraft_id = state['ID']

            state['next_flights'] = tuple(self.next_flights(aircraft_id, step))
            state['is_available'] = self.is_available(aircraft_id, step)
            state['conflict_at_next_t'] = self.conflict_at_step(aircraft_id, step)

    def get_new_state_keys(self, state_key, action):
        """
        Simulate the outcome of an action and returns the new state keys resulting from the action.
        """
        # Create a deep copy of the current states and data to avoid modifying the original state
        aircraft_data = copy.deepcopy(self.aircraft_data)
        flight_data = copy.deepcopy(self.flight_data)
        temp_states = copy.deepcopy(self.states)

        aircraft_id, step, available, next_flights, conflict = state_key
        action_type, flight_to_swap_nr, new_aircraft_id = action

        # SWAP:
        if action_type == 'swap':
            # Update flight assignment in the copied data
            new_aircraft = get_ac_dict(aircraft_data, new_aircraft_id)
            old_aircraft = get_ac_dict(aircraft_data, aircraft_id)

            # Remove the flight from the old aircraft and add it to the new aircraft
            old_aircraft['AssignedFlights'].remove(flight_to_swap_nr)
            new_aircraft['AssignedFlights'].append(flight_to_swap_nr)
            flight = get_flight_dict(flight_to_swap_nr, flight_data)
            flight['AssignedAircraft'] = new_aircraft_id

            old_aircraft_state_key = self.get_state_key(aircraft_id, step + 1)
            if old_aircraft_state_key in temp_states:
                # self.modify_state(old_aircraft_state_key, step + 1, temp_states)
                old_aircraft_state = temp_states[old_aircraft_state_key]
                old_aircraft_state['next_flights'] = tuple(self.next_flights(aircraft_id, step + 1))
                old_aircraft_state['is_available'] = self.is_available(aircraft_id, step + 1)
                old_aircraft_state['conflict_at_next_t'] = self.conflict_at_step(aircraft_id, step + 2)

            # Create the new state for the new aircraft in the copied data
            new_aircraft_state_key = self.get_state_key(new_aircraft_id, step + 1)
            if new_aircraft_state_key in temp_states:
                new_aircraft_state = temp_states[new_aircraft_state_key]
                new_aircraft_state['next_flights'] = tuple(self.next_flights(new_aircraft_id, step + 1))
                new_aircraft_state['is_available'] = self.is_available(new_aircraft_id, step + 1)
                new_aircraft_state['conflict_at_next_t'] = self.conflict_at_step(new_aircraft_id, step + 2)

            return old_aircraft_state_key, new_aircraft_state_key

        elif action_type == 'none':
            # Create the new state for the do nothing action
            new_state_key = self.get_state_key(aircraft_id, step + 1)
            if new_state_key in temp_states:
                new_state = temp_states[new_state_key]
                new_state['next_flights'] = tuple(self.next_flights(aircraft_id, step + 1))
                new_state['is_available'] = self.is_available(aircraft_id, step + 1)
                new_state['conflict_at_next_t'] = self.conflict_at_step(aircraft_id, step + 2)

            return new_state_key, None

    def select_best_action(self, state_key):
        """
        Selects the best action based on the current value function estimates.
        """
        possible_actions = self.get_possible_actions(state_key)
        best_action = None
        best_value = float('-inf')

        print(f'ACTIONS')
        for action in possible_actions:
            # action tuples:
            # ('swap', flight_nr, aircraft_id)
            # ('delay', flight_nr, delay_minutes)
            # ('none', 'none', 'none')
            if action[0] == 'swap':
                new_state_key1, new_state_key2 = self.get_new_state_keys(state_key, action)
                # Sum the values of both affected states
                action_value1 = self.states[new_state_key1]['value'][-1]
                action_value2 = self.states[new_state_key2]['value'][-1]
                action_value = action_value1 + action_value2

            elif action[0] == 'none':
                new_state_key1, _ = self.get_new_state_keys(state_key, action)
                action_value = self.states[new_state_key1]['value'][-1]

            if action_value > best_value:
                best_value = action_value
                best_action = action
                if best_action[0] == 'swap':
                    print(f'BEST ACTION TO SWAP {best_action}')
                    print(f'ACTION VALUE {action_value}')

        return best_action

    def update_value_function(self, state_key, action, reward):
        """
        Updates the value function for a given state-action pair.
        """
        step_size = 0.1  # Step size for value function update
        if state_key in self.states:
            current_value = self.states[state_key]['value'][-1]
            new_value = (1 - step_size) * current_value + step_size * reward
            self.states[state_key]['value'].append(new_value) # stores all iterations of values in the state

    def get_next_state_value(self, simulated_state):
        total_next_state_value = 0
        for state in simulated_state.values():
            total_next_state_value += state['value'][-1]  # Most recent value of each state

        return total_next_state_value

    def compute_reward_for_action_set(self, current_step, action_set):
        total_reward = 0
        for aircraft, action in zip(self.aircraft_data, action_set):
            aircraft_id = aircraft['ID']
            state_key = self.get_state_key(aircraft_id, current_step)  # Use current state, not simulated state
            reward = self.compute_reward(state_key, action)
            total_reward += reward

        return total_reward

    def compute_reward(self, state_key, action):
        """
        Compute the immediate reward of taking the given action from the given state.
        This might involve resolving a conflict, avoiding a delay, etc.
        """
        aircraft_id, step, available, next_flights, conflict_next = state_key
        reward = 0

        if action is not None:
            action_type, flight_to_swap_nr, new_aircraft_id = action

            if action_type == 'swap':
                if conflict_next == 1:
                    reward += 10000         # # reward conflict aversion

                elif conflict_next == 0:
                    reward += 0             # Avoid unnecessary swaps

            elif action_type == 'none':
                if conflict_next == 1:      # No action is taken when in conflict at next step
                    reward -= 10000

                elif conflict_next == 0:    #
                    reward += 0

        return reward

    def compute_reward(self, state_key, action):
        """
        Compute the immediate reward of taking the given action from the given state.
        This might involve resolving a conflict, avoiding a delay, etc.
        """
        aircraft_id, step, available, next_flights, conflict_next = state_key
        reward = 0

        if action is not None:
            action_type, flight_to_swap_nr, new_aircraft_id = action

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

        return reward

    def simulate_action_in_temp_state(self, temp_states, aircraft_data, flight_data, state_key, action):
        # Similar to simulate_action but operates on temp data
        aircraft_id, step, available, next_flights, conflict = state_key
        action_type, flight_to_swap_nr, new_aircraft_id = action

        if action_type == 'swap':
            # Update flight assignment in the copied data
            new_aircraft = get_ac_dict(aircraft_data, new_aircraft_id)
            old_aircraft = get_ac_dict(aircraft_data, aircraft_id)

            # Remove the flight from the old aircraft and add it to the new aircraft
            old_aircraft['AssignedFlights'].remove(flight_to_swap_nr)
            new_aircraft['AssignedFlights'].append(flight_to_swap_nr)
            flight = get_flight_dict(flight_to_swap_nr, flight_data)
            flight['AssignedAircraft'] = new_aircraft_id

            old_aircraft_state_key = self.get_state_key(aircraft_id, step + 1)
            if old_aircraft_state_key in temp_states:
                old_aircraft_state = temp_states[old_aircraft_state_key]
                old_aircraft_state['next_flights'] = tuple(self.next_flights(aircraft_id, step + 1))
                old_aircraft_state['is_available'] = self.is_available(aircraft_id, step + 1)
                old_aircraft_state['conflict_at_next_t'] = self.conflict_at_next_step(aircraft_id, step + 1)

            new_aircraft_state_key = self.get_state_key(new_aircraft_id, step + 1)
            if new_aircraft_state_key in temp_states:
                new_aircraft_state = temp_states[new_aircraft_state_key]
                new_aircraft_state['next_flights'] = tuple(self.next_flights(new_aircraft_id, step + 1))
                new_aircraft_state['is_available'] = self.is_available(new_aircraft_id, step + 1)
                new_aircraft_state['conflict_at_next_t'] = self.conflict_at_next_step(new_aircraft_id, step + 1)

    def simulate_action_set(self, step, action_set):
        # Make deep copies of the current aircraft and flight data
        aircraft_data = copy.deepcopy(self.aircraft_data)
        flight_data = copy.deepcopy(self.flight_data)
        temp_states = copy.deepcopy(self.states)

        for aircraft, action in zip(self.aircraft_data, action_set):
            aircraft_id = aircraft['ID']
            state_key = self.get_state_key(aircraft_id, step)
            self.simulate_action_in_temp_state(temp_states, aircraft_data, flight_data, state_key, action)

        return temp_states


# HELPER FUNCTIONS:
    def is_available(self, aircraft_id, step):
        available = True

        if self.is_disrupted(aircraft_id, step):
            available = False

        # Chcek if aircraft is in flight
        if self.is_in_flight(aircraft_id, step):
            available = False
        return available

    def is_disrupted(self, aircraft_id, step):
        # Check if aircraft is disrupted
        for disruption in self.disruptions:
            if (disruption["Type"] == 'AU' and
                    disruption["StartTime"] <= self.periods[step] < disruption["EndTime"] and
                    disruption["Aircraft"] == aircraft_id):
                return True

    def is_in_flight(self, aircraft_id, step):
        """Returns True if aircraft_id is in flight at given period"""
        in_flight = False

        # Check if aircraft is in flight
        for flight_nr in get_ac_dict(self.aircraft_data, aircraft_id)['AssignedFlights']:
            flight = get_flight_dict(flight_nr, self.flight_data)
            if flight['ADT'] <= self.periods[step] < flight['AAT']:
                in_flight = True
                return in_flight, flight
        return in_flight, None

    def unavailable_for_flight(self, flight_nr, aircraft_id):
        """ Checks if the aircraft is unavailable for a flight due to disruption"""
        flight = get_flight_dict(flight_nr, self.flight_data)
        for disruption in self.disruptions:
            if disruption["Type"] == "AU" and disruption['Aircraft'] == aircraft_id:
                return (flight['ADT'] <= disruption['StartTime'] <= flight['AAT'] or
                        flight['ADT'] <= disruption['EndTime'] <= flight['AAT'] or
                        (flight['ADT'] <= disruption['StartTime'] and flight['AAT'] >= disruption[
                            'EndTime']) or
                        (flight['ADT'] >= disruption['StartTime'] and flight['AAT'] <= disruption[
                            'EndTime']))

    def conflict_at_step(self, aircraft_id, step):
        conflict = False

        # no conflicts in last step
        if step == len(self.steps):
            return conflict

        in_flight, flight = self.is_in_flight(aircraft_id, step) # Check if in flight at t and which flight

        if flight is not None:
            unavailable_for_flight = self.unavailable_for_flight(flight['Flightnr'], aircraft_id) # check that flight for disruptins
        else:
            unavailable_for_flight = False


        # Check if the flight the aircraft is performing at t+1 is disrupted anywhere
        if in_flight and unavailable_for_flight:
            conflict = True
        return conflict

    def can_swap(self, flight_to_swap_nr, new_aircraft_id):
        # CHECK FOR OVERLAPPING FLIGHTS AND AIRCRAFT UNAVAILABILITY:
        has_overlapping_flights = False
        is_unavailable = False
        can_swap = False

        new_aircraft = get_ac_dict(self.aircraft_data, new_aircraft_id)
        flight_to_swap = get_flight_dict(flight_to_swap_nr, self.flight_data)

        # 1. potential new aircraft must not have overlapping flights with to be swapped flight
        for flight_nr in new_aircraft['AssignedFlights']:
            flight = get_flight_dict(flight_nr, self.flight_data)  # Get flight info dict from flight number
            if self.overlaps(flight_to_swap, flight):
                has_overlapping_flights = True

        # 2. potential new aircraft is not unavailable during to be swapped flight
        is_unavailable_for_flight = self.unavailable_for_flight(flight_to_swap["Flightnr"], new_aircraft_id)

        if not has_overlapping_flights and not is_unavailable_for_flight:
            can_swap = True

        return can_swap

    def overlaps(self, flight_to_swap, flight):
        """Checks if two flight times overlap"""
        return (flight_to_swap['ADT'] <= flight['ADT'] <= flight_to_swap['AAT'] or
                flight_to_swap['ADT'] <= flight['AAT'] <= flight_to_swap['AAT'] or
                (flight_to_swap['ADT'] <= flight['ADT'] and flight_to_swap['AAT'] >= flight['AAT']) or
                (flight_to_swap['ADT'] >= flight['ADT'] and flight_to_swap['AAT'] <= flight['AAT'])
                )

    def next_flights(self, aircraft_id, step):
        """Returns a list of the next flights to be flown by an aircraft at a given period"""
        next_flights = []
        for flight_nr in get_ac_dict(self.aircraft_data, aircraft_id)['AssignedFlights']:
            flight = get_flight_dict(flight_nr, self.flight_data)
            if self.periods[step] <= flight['ADT']:
                next_flights.append(flight_nr)
        return next_flights

    def has_overlapping_swapped_flights(self, action_set, step):
        # Dictionary to track flights assigned to each aircraft
        swapped_flights = {}

        for action in action_set:
            action_type, flight_to_swap_nr, new_aircraft_id = action

            if action_type == 'swap':
                # Get the flight details
                flight_to_swap = get_flight_dict(flight_to_swap_nr, self.flight_data)

                # Check if the new aircraft already has an assigned flight that overlaps
                if new_aircraft_id in swapped_flights:
                    for assigned_flight_nr in swapped_flights[new_aircraft_id]:
                        assigned_flight = get_flight_dict(assigned_flight_nr, self.flight_data)
                        if self.overlaps(flight_to_swap, assigned_flight):
                            return True  # Found an overlap

                # Add the flight to the list of assigned flights for the aircraft
                if new_aircraft_id not in swapped_flights:
                    swapped_flights[new_aircraft_id] = []
                swapped_flights[new_aircraft_id].append(flight_to_swap_nr)

        return False

    def is_valid_action_set(self, action_set, step):
        """Checks all conditions for action sets to be valid and returns True if action set is valid"""
        valid = True
        if self.has_overlapping_swapped_flights(action_set, step):
            valid = False

        return valid

# SOLVE
    def solve_with_vfa(self, num_iterations):
        """
        Solves the disruption management problem using Value Function Approximation.
        """
        for _ in range(num_iterations):
            for step in reversed(range(len(self.steps))):
                for aircraft in self.aircraft_data:
                    state_key = self.get_state_key(aircraft['ID'], step)
                    best_action = self.select_best_action(state_key)

                    if best_action[0] == 'swap':
                        # Simulate the swap and get the resulting state keys for both affected aircraft
                        new_state_key1, new_state_key2 = self.get_new_state_keys(state_key, best_action)

                        # Compute rewards for both states (if applicable)
                        reward1 = self.compute_reward(new_state_key1, best_action)
                        reward2 = self.compute_reward(new_state_key2, best_action)
                        reward = reward1 + reward2

                        # Update value function for both states
                        self.update_value_function(new_state_key1, best_action, reward)
                        self.update_value_function(new_state_key2, best_action, reward)

                    elif best_action[0] == 'none':
                        new_state_key1, = self.get_new_state_keys(state_key, best_action)
                        reward = self.compute_reward(new_state_key1, best_action)
                        # Update value function next state
                        self.update_value_function(new_state_key1, best_action, reward)

    def solve_to_optimality(self):
        self.policy = {}  # Store the best action set for each state set

        # Initialize values at the last period
        last_period = len(self.steps) - 1
        for aircraft in self.aircraft_data:
            state_key = self.get_state_key(aircraft['ID'], last_period)
            if state_key not in self.states:
                self.add_state(state_key)

            # Set final period value to the immediate reward (e.g., conflicts resolved)
            final_state = self.states[state_key]
            final_state['value'] = [self.compute_reward(state_key, ('none', 'none', 'none'))]  # No action at final step

        # Iterate backward through time
        for step in reversed(range(last_period)):
            action_values = {}

            # Centralized action evaluation
            all_possible_actions = self.get_all_possible_actions(step)  # Get all possible actions for all aircraft

            # Capture the full state set for this t
            state_set = tuple(self.get_state_key(aircraft['ID'], step) for aircraft in self.aircraft_data)

            for action_set in all_possible_actions:
                simulated_states = self.simulate_action_set(step, action_set)
                immediate_reward = self.compute_reward_for_action_set(step, action_set)

                if step == last_period - 1:
                    action_values[action_set] = immediate_reward
                else:
                    next_state_value = self.get_next_state_value(simulated_states)
                    action_values[action_set] = immediate_reward + self.y * next_state_value

            # Find the best set of actions for this step
            best_action_set = max(action_values, key=action_values.get, default=None)

            if best_action_set:
                # Store the best action set for this state set
                self.policy[state_set] = best_action_set

                # # Apply the best action set to update states
                # self.apply_action_set(best_action_set, step)

    def update_data(self, action, step):
        action_type, flight_nr, new_aircraft_id = action

        if action_type == 'swap':
            selected_flight = get_flight_dict(flight_nr, self.flight_data)
            new_aircraft = get_ac_dict(self.aircraft_data, new_aircraft_id)
            previous_aircraft_id = selected_flight['AssignedAircraft']  # Aircraft currently assigned to the flight
            previous_ac = get_ac_dict(self.aircraft_data, previous_aircraft_id)

            # Reassign aircraft and update aircraft and flight data
            selected_flight['AssignedAircraft'] = new_aircraft['ID']  # Update the flight's assigned aircraft
            previous_ac['AssignedFlights'].remove(flight_nr)  # Remove flight from the previous aircraft's list
            new_aircraft['AssignedFlights'].append(flight_nr)  # Add flight to the new aircraft's list

    def update_states(self, action, state_key):
        """Updates the self.states dictionary following chosen actions by reading the new states from the updated data"""
        action_type, flight_nr, new_aircraft_id = action
        old_aircraft_id, step = state_key[0], state_key[1]

        if action_type == 'swap':
            # Update the state for both the previous aircraft and the new aircraft
            new_aircraft = get_ac_dict(self.aircraft_data, new_aircraft_id)

            # Update state for the old aircraft
            old_ac_state_key = state_key
            if old_ac_state_key in self.states:
                previous_ac_state = self.states[old_ac_state_key]
                # Update states:
                previous_ac_state['next_flights'] = tuple(self.next_flights(old_aircraft_id, step))
                previous_ac_state['is_available'] = self.is_available(old_aircraft_id, step)
                previous_ac_state['conflict_at_next_t'] = self.conflict_at_step(old_aircraft_id, step+1)

            # Update state for the new aircraft
            new_ac_state_key = self.get_state_key(new_aircraft_id, step)
            if new_ac_state_key in self.states:
                new_state = self.states[new_ac_state_key]
                new_state['next_flights'] = tuple(self.next_flights(new_aircraft_id, step))
                new_state['is_available'] = self.is_available(new_aircraft_id, step)
                new_state['conflict_at_next_t'] = self.conflict_at_step(new_aircraft_id, step+1)

        elif action_type == 'none':
            pass
            # Only update the state of the current aircraft
            # if state_key in self.states:
            #     state_info = self.states[state_key]
            #     state_info['next_flights'] = tuple(self.next_flights(aircraft_id, step))
            #     state_info['is_available'] = self.is_available(aircraft_id, step)
            #     state_info['conflict_at_next_t'] = self.conflict_at_next_step(aircraft_id, step)

# RESULTS
    def retrieve_optimal_action_sequence(self):
        print(f'\nPOLICY:\n    {self.policy}')
        action_sequence = {}
        total_reward = 0  # Initialize the total reward

        # Loop through each step
        for step in range(len(self.steps)):
            # Capture the full state set for this t
            state_set = tuple(self.get_state_key(aircraft['ID'], step) for aircraft in self.aircraft_data)

            # Retrieve the best action set for the current state set
            best_action_set = self.policy.get(state_set, None)

            if best_action_set is not None:
                action_sequence[step] = best_action_set

                # Compute the reward for this action set and add it to the total reward
                reward = self.compute_reward_for_action_set(step, best_action_set)
                total_reward += reward
            else:
                print(f'    No actions available for state set at step {step}: {state_set}')

        return action_sequence, total_reward

    def plot_schedule(self, step):
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

                    if ADT and AAT:
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
                        if ADT and AAT:
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

    def simulate_solution(self, action_sequence):
        print(f'ACTION SEQUENCE:\n   {action_sequence}')

        # Iterate over each step in the action sequence
        for step in range(len(self.steps) - 1):
            # Get the actions for the current step
            actions = action_sequence.get(step, [])

            # Plot the schedule before applying the actions
            self.plot_schedule(step)

            # Apply each action in the current step
            for action in actions:
                self.update_data(action, step)

    def print_states(self):
        for state in self.states:
            print(f'\nID: {state[0]}')
            print(f'TIME: {str(self.periods[state[1]]).split(' ')[-1]}')
            print(f'{state[2]}, {state[3]}, {state[4]}')

if __name__ == '__main__':
    folder = 'A01_small'
    # folder = 'A01_mini'
    aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data(folder)
    adm = DisruptionRealization(aircraft_data, flight_data, disruptions, recovery_start, recovery_end)
    print(adm.flight_data)

    adm.plot_schedule(step=0)
    adm.solve_to_optimality()
    solution, objective_value = adm.retrieve_optimal_action_sequence()
    adm.simulate_solution(solution)
    adm.plot_schedule(step=len(adm.steps))
    print(f'OBJECTIVE VALUE: {objective_value}')
    print(f'SOLUTION')
    for t in solution:
        print(t, solution[t])

for state_key, state in adm.states.items():
    print(state_key[1], state)
