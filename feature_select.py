import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

def filter_correlation(X, y, data):
    # Calculate the correlation matrix
    corr_matrix = data.corr()
    correlation_threshold = 0.25

    # Plot heatmap for visualization
    # plt.figure(figsize=(16, 10))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # plt.show()

    # Get correlation with the target 'value'
    y_correlation = corr_matrix['value'].abs().sort_values(ascending=False)
    print("Correlation with target (value):\n", y_correlation)

    # Identify columns with NaN correlations
    nan_correlation_features = y_correlation[y_correlation.isna()].index.tolist()
    # print("Features with NaN correlation:\n", nan_correlation_features)

    # Identify features with absolute correlation below the threshold
    low_correlation_features = y_correlation[abs(y_correlation) < correlation_threshold].index.tolist()
    # print("Features below correlation threshold:\n", low_correlation_features)

    # Combine NaN correlation features and low correlation features
    dropped_features = nan_correlation_features + low_correlation_features

    # print("Dropped features:\n", dropped_features)

    # Filter selected features, excluding target and dropped features
    selected_features = [col for col in X.columns if col not in dropped_features]
    X_filtered = X[selected_features]

    # Drop rows with NaN values in X_filtered and y
    X_filtered = X_filtered.dropna()
    y_filtered = data['value'].loc[X_filtered.index]  # Ensure alignment between X and y

    return X_filtered, y_filtered, dropped_features

def test_model(X, y, model):
    # Step 2: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Step 4: Evaluate model performance on the test set
    y_pred = model.predict(X_test)
    # print(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Squared Error (MSE) on test set: {mse}")
    print(f"Mean Absolute Error (MAE) on test set: {mae}, {model = }\n")
    return mse, mae

def optimize_RFR(X, y, n_estimator_range, depth_range):
    maes = {}
    mae_depth = {}

    for e in n_estimator_range:
        for d in depth_range:
            model = RandomForestRegressor(n_estimators=e, max_depth=d, random_state=42)
            mse, mae = test_model(X, y, model)
            maes[(e, d)] = mae

    best = min(maes, key=maes.get)
    print()
    print(best)
    print(maes[best])


