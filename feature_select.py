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
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from sklearn.model_selection import KFold, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def filter_correlation(X, y, data):
    # Calculate the correlation matrix
    corr_matrix = data.corr()
    correlation_threshold = 0.0

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


def bayesian_optimization(X, y, model=RandomForestRegressor(random_state=42)):
    # Define the parameter space for Bayesian optimization
    param_grid = {
        'n_estimators': Integer(10, 250),           # Number of trees in the forest
        'max_depth': Integer(1, 30),                # Maximum depth of the tree
        'min_samples_split': Integer(2, 20),        # Minimum samples for a split
        'min_samples_leaf': Integer(1, 20),         # Minimum samples per leaf
        'max_features': Real(0.3, 1.0, 'uniform'),  # Max features for splitting
    }

    # Define Bayesian Optimization with cross-validation
    bayes_cv = BayesSearchCV(
        estimator=model,
        search_spaces=param_grid,
        n_iter=30,  # Number of parameter evaluations
        cv=5,  # Cross-validation folds
        scoring='neg_mean_absolute_error',  # Optimization objective
        random_state=42,
        verbose=0  # Set to 1 or 2 for detailed logging
    )

    # Perform the search
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    bayes_cv.fit(X_train, y_train)

    print("Best Parameters:", bayes_cv.best_params_)
    print("Best Score (Negative MAE):", bayes_cv.best_score_)


    # Predict on test set
    y_pred = bayes_cv.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print()
    print("Bayes test Set MSE:", mse)
    print("Bayes test Set MAE:", mae)

    best_model = bayes_cv.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print("Bayes test Set Score (R^2):", test_score)

    return best_model


def test_model(X, y, model):
    """
    Test the model using a single train-test split.

    Parameters:
        X: Features (DataFrame or array-like)
        y: Target (Series or array-like)
        model: Machine learning model (e.g., RandomForestRegressor)

    Returns:
        mse: Mean Squared Error on the test set
        mae: Mean Absolute Error on the test set
        rmae: Relative Mean Absolute Error on the test set
        r2: R-squared score on the test set
    """
    # Step 1: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Fit the model
    model.fit(X_train, y_train)

    # Step 3: Predict on the test set
    y_pred = model.predict(X_test)

    # Step 4: Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Step 5: Compute Relative MAE (rMAE)
    mean_target = np.mean(y_test)  # Use test set mean for a fair comparison
    rmae = mae / mean_target

    # Print results
    # print(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
    # print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")
    # print(f"Relative Mean Absolute Error (rMAE) on test set: {rmae:.4f}")
    # print(f"R-squared (R2) on test set: {r2:.4f}, {model = }\n")

    return mse, mae, rmae, r2


def test_model_with_kfold(X, y, model, k=5):
    """
    Test the model using k-fold cross-validation.

    Parameters:
        X: Features (DataFrame or array-like)
        y: Target (Series or array-like)
        model: Machine learning model (e.g., RandomForestRegressor)
        k: Number of folds for k-fold cross-validation (default is 5)

    Returns:
        avg_mse: Average Mean Squared Error across folds
        avg_mae: Average Mean Absolute Error across folds
        avg_rmae: Average Relative Mean Absolute Error across folds
        avg_r2: Average R-squared score across folds
    """
    # Step 1: Initialize k-fold
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # Step 2: Define scoring metrics
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    r2_scorer = make_scorer(r2_score, greater_is_better=False)

    # Step 3: Perform cross-validation for MSE, MAE, and R2
    mse_scores = cross_val_score(model, X, y, cv=kfold, scoring=mse_scorer)
    mae_scores = cross_val_score(model, X, y, cv=kfold, scoring=mae_scorer)
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring=r2_scorer)

    # Step 4: Calculate average MSE, MAE, and R2
    avg_mse = -np.mean(mse_scores)  # Negate because scorers are negative
    avg_mae = -np.mean(mae_scores)  # Negate because scorers are negative
    avg_r2 = -np.mean(r2_scores)    # Negate because scorers are negative

    # Step 5: Compute Relative MAE (rMAE)
    mean_target = np.mean(y)  # Calculate the mean of the target variable
    avg_rmae = avg_mae / mean_target  # Relative MAE

    # Print results
    # print(f"Mean Squared Error (MSE) across {k} folds: {avg_mse:.4f}")
    # print(f"Mean Absolute Error (MAE) across {k} folds: {avg_mae:.4f}")
    # print(f"Relative Mean Absolute Error (rMAE) across {k} folds: {avg_rmae:.4f}")
    # print(f"R-squared (R2) across {k} folds: {avg_r2:.4f}")

    return avg_mse, avg_mae, avg_rmae, avg_r2




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


