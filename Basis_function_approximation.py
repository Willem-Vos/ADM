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
from openpyxl import load_workbook
import os
from sklearn.linear_model import LinearRegression

state_values = []
aircraft_data, flight_data, rotations_data, disruptions, recovery_start, recovery_end = read_data('TEST1')
aircraft_ids = [aircraft['ID'] for aircraft in aircraft_data]

def basis_functions(state):
    features = []
    features.append(state['t'])  # global attribute

    # Loop through each aircraft's attributes
    for aircraft in aircraft_ids:
        aircraft_state = state[aircraft]
        # Linear terms
        features.append(aircraft_state['n_remaining_flights'])
        features.append(aircraft_state['n_remaining_conflicts'])
        features.append(aircraft_state['utilization'])

        # Interaction terms with global features (optional)
        features.append(aircraft_state['n_remaining_flights'] * state['time_elapsed'])
        features.append(aircraft_state['n_remaining_conflicts'] * state['time_elapsed'])

        # Quadratic terms (optional)
        features.append(aircraft_state['n_remaining_flights'] ** 2)
        features.append(aircraft_state['n_remaining_conflicts'] * aircraft_state['utilization'])
        # Add other aircraft-specific basis functions based on domain knowledge

    return features



# Assume state_values is a list of (state, value) tuples collected during training
# X = [basis_functions(state) for state, _ in state_values]
# y = [value for _, value in state_values]

X = [[1, 0], [0, 1], [1, 1], [0,0], [1, 0], [0, 1]]
y = [[10],   [2],    [6],    [2],   [9],   [1]]
# Fit a linear model to estimate theta
model = LinearRegression()
model.fit(X, y)

# theta coefficients are stored in model.coef_
theta = model.coef_

features = [0.5,0.5]
y_test = model.predict([features])[0]
print(y_test)
print(theta)