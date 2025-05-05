import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ONLY use pandas for initial data loading
# All other operations should use NumPy

# Part 1: Data Preparation
def load_and_preprocess_data(file_path):
    """
    Load the dataset and perform preprocessing

    Args:
        file_path: Path to the dataset CSV file

    Returns:
        X_train: Training features
        y_train: Training target values
        X_test: Testing features
        y_test: Testing target values
    """
    # Load data using pandas
    df = pd.read_csv(file_path)

    # Select features for prediction (5 relevant features)
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront']
    X = df[features].values
    y = df['price'].values

    # Handle missing values if necessary
    # (Assuming no missing values in this dataset, but we would handle them here)

    # Normalize/standardize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std

    # Split data into training (80%) and testing (20%) sets
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(X_normalized))
    split_point = int(len(X_normalized) * 0.8)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    X_train = X_normalized[train_indices]
    y_train = y[train_indices]
    X_test = X_normalized[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


# Part 2: Basic Linear Regression with Gradient Descent
def predict(X, weights, bias):
    """
    Make predictions using the linear model: y = X*w + b

    Args:
        X: Features
        weights: Model weights
        bias: Model bias

    Returns:
        Predicted values
    """
    return np.dot(X, weights) + bias


def compute_cost(X, y, weights, bias):
    """
    Compute the Mean Squared Error cost function

    Args:
        X: Features
        y: Target values
        weights: Model weights
        bias: Model bias

    Returns:
        Mean Squared Error
    """
    m = len(y)
    predictions = predict(X, weights, bias)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def gradient_descent(X, y, learning_rate, num_iterations):
    """
    Implement gradient descent algorithm for linear regression

    Args:
        X: Features
        y: Target values
        learning_rate: Learning rate alpha
        num_iterations: Number of iterations to run

    Returns:
        weights: Optimized weights
        bias: Optimized bias
        cost_history: History of cost values during optimization
        weights_history: History of weights during optimization
        bias_history: History of bias during optimization
    """
    # Initialize parameters
    m, n = X.shape  # m = number of samples, n = number of features
    weights = np.zeros(n)
    bias = 0

    # Arrays to store history
    cost_history = np.zeros(num_iterations)
    weights_history = np.zeros((num_iterations, n))
    bias_history = np.zeros(num_iterations)

    # Implement gradient descent algorithm
    for i in range(num_iterations):
        # Make predictions
        predictions = predict(X, weights, bias)

        # Calculate gradients
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)

        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Store history
        cost_history[i] = compute_cost(X, y, weights, bias)
        weights_history[i] = weights
        bias_history[i] = bias

    return weights, bias, cost_history, weights_history, bias_history


# Part 3: RIDGE Regression (L2 Regularization)
def compute_cost_ridge(X, y, weights, bias, lambda_param):
    """
    Compute the Mean Squared Error cost function with L2 regularization

    Args:
        X: Features
        y: Target values
        weights: Model weights
        bias: Model bias
        lambda_param: Regularization parameter

    Returns:
        Mean Squared Error with L2 regularization
    """
    m = len(y)
    predictions = predict(X, weights, bias)

    # Standard MSE
    mse_cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

    # L2 regularization term (don't regularize bias)
    l2_cost = (lambda_param / (2 * m)) * np.sum(weights ** 2)

    return mse_cost + l2_cost


def gradient_descent_ridge(X, y, learning_rate, num_iterations, lambda_param):
    """
    Implement gradient descent algorithm for RIDGE regression

    Args:
        X: Features
        y: Target values
        learning_rate: Learning rate alpha
        num_iterations: Number of iterations to run
        lambda_param: Regularization parameter

    Returns:
        weights: Optimized weights
        bias: Optimized bias
        cost_history: History of cost values during optimization
        weights_history: History of weights during optimization
        bias_history: History of bias during optimization
    """
    # Initialize parameters
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    # Arrays to store history
    cost_history = np.zeros(num_iterations)
    weights_history = np.zeros((num_iterations, n))
    bias_history = np.zeros(num_iterations)

    # Implement gradient descent algorithm with RIDGE regularization
    for i in range(num_iterations):
        # Make predictions
        predictions = predict(X, weights, bias)

        # Calculate gradients with L2 regularization
        dw = (1 / m) * np.dot(X.T, (predictions - y)) + (lambda_param / m) * weights
        db = (1 / m) * np.sum(predictions - y)  # Bias not regularized

        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Store history
        cost_history[i] = compute_cost_ridge(X, y, weights, bias, lambda_param)
        weights_history[i] = weights
        bias_history[i] = bias

    return weights, bias, cost_history, weights_history, bias_history


# Part 4: LASSO Regression (L1 Regularization)
def compute_cost_lasso(X, y, weights, bias, lambda_param):
    """
    Compute the Mean Squared Error cost function with L1 regularization

    Args:
        X: Features
        y: Target values
        weights: Model weights
        bias: Model bias
        lambda_param: Regularization parameter

    Returns:
        Mean Squared Error with L1 regularization
    """
    m = len(y)
    predictions = predict(X, weights, bias)

    # Standard MSE
    mse_cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

    # L1 regularization term (don't regularize bias)
    l1_cost = (lambda_param / m) * np.sum(np.abs(weights))

    return mse_cost + l1_cost


def gradient_descent_lasso(X, y, learning_rate, num_iterations, lambda_param):
    """
    Implement gradient descent algorithm for LASSO regression

    Args:
        X: Features
        y: Target values
        learning_rate: Learning rate alpha
        num_iterations: Number of iterations to run
        lambda_param: Regularization parameter

    Returns:
        weights: Optimized weights
        bias: Optimized bias
        cost_history: History of cost values during optimization
        weights_history: History of weights during optimization
        bias_history: History of bias during optimization
    """
    # Initialize parameters
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    # Arrays to store history
    cost_history = np.zeros(num_iterations)
    weights_history = np.zeros((num_iterations, n))
    bias_history = np.zeros(num_iterations)

    # Implement gradient descent algorithm with LASSO regularization
    for i in range(num_iterations):
        # Make predictions
        predictions = predict(X, weights, bias)

        # Calculate standard gradient
        dw_mse = (1 / m) * np.dot(X.T, (predictions - y))

        # Add L1 regularization term
        dw_l1 = np.zeros(n)
        for j in range(n):
            if weights[j] > 0:
                dw_l1[j] = lambda_param / m
            elif weights[j] < 0:
                dw_l1[j] = -lambda_param / m
            else:
                dw_l1[j] = 0  # If weight is exactly 0, subdifferential is in [-λ/m, λ/m]

        # Combined gradient
        dw = dw_mse + dw_l1

        # Bias gradient (not regularized)
        db = (1 / m) * np.sum(predictions - y)

        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Store history
        cost_history[i] = compute_cost_lasso(X, y, weights, bias, lambda_param)
        weights_history[i] = weights
        bias_history[i] = bias

    return weights, bias, cost_history, weights_history, bias_history


# Part 5: Visualization and Analysis Functions
def plot_cost_history(cost_history, title):
    """
    Plot the cost history over iterations

    Args:
        cost_history: History of cost values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.grid(True)
    plt.show()


def plot_coefficients(feature_names, basic_weights, ridge_weights, lasso_weights):
    """
    Plot the coefficients from different models for comparison

    Args:
        feature_names: Names of the features
        basic_weights: Weights from basic linear regression
        ridge_weights: Weights from RIDGE regression
        lasso_weights: Weights from LASSO regression
    """
    plt.figure(figsize=(12, 8))
    x = np.arange(len(feature_names))
    width = 0.25

    plt.bar(x - width, basic_weights, width, label='Basic')
    plt.bar(x, ridge_weights, width, label='RIDGE')
    plt.bar(x + width, lasso_weights, width, label='LASSO')

    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Comparison of Model Coefficients')
    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()


def evaluate_model(X, y, weights, bias, model_name):
    """
    Evaluate the model on the provided data

    Args:
        X: Features
        y: Target values
        weights: Model weights
        bias: Model bias
        model_name: Name of the model for printing

    Returns:
        Mean Squared Error
    """
    y_pred = predict(X, weights, bias)
    mse = np.mean((y_pred - y) ** 2)
    print(f"{model_name} MSE: {mse:.4f}")
    return mse


# Main execution
if __name__ == "__main__":
    # File path to the dataset
    file_path = "kc_house_data.csv"  # Update with the correct path

    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data(file_path)

    # Get feature names for later visualization
    feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront']

    # Hyperparameters
    learning_rate = 0.01  # You may need to adjust this
    num_iterations = 1000
    ridge_lambda = 0.1  # You may need to adjust this
    lasso_lambda = 0.01  # You may need to adjust this

    print("Training Basic Linear Regression model...")
    basic_weights, basic_bias, basic_cost_history, _, _ = gradient_descent(
        X_train, y_train, learning_rate, num_iterations
    )

    print("Training RIDGE Regression model...")
    ridge_weights, ridge_bias, ridge_cost_history, _, _ = gradient_descent_ridge(
        X_train, y_train, learning_rate, num_iterations, ridge_lambda
    )

    print("Training LASSO Regression model...")
    lasso_weights, lasso_bias, lasso_cost_history, _, _ = gradient_descent_lasso(
        X_train, y_train, learning_rate, num_iterations, lasso_lambda
    )

    # Evaluate models on test set
    print("\nEvaluation on Test Set:")
    basic_mse = evaluate_model(X_test, y_test, basic_weights, basic_bias, "Basic Linear Regression")
    ridge_mse = evaluate_model(X_test, y_test, ridge_weights, ridge_bias, "RIDGE Regression")
    lasso_mse = evaluate_model(X_test, y_test, lasso_weights, lasso_bias, "LASSO Regression")

    # Plot cost history
    plot_cost_history(basic_cost_history, "Basic Linear Regression Cost History")
    plot_cost_history(ridge_cost_history, "RIDGE Regression Cost History")
    plot_cost_history(lasso_cost_history, "LASSO Regression Cost History")

    # Plot coefficients for comparison
    plot_coefficients(feature_names, basic_weights, ridge_weights, lasso_weights)

    # Check feature selection by LASSO
    print("\nFeature Selection by LASSO:")
    for i, (name, coef) in enumerate(zip(feature_names, lasso_weights)):
        if abs(coef) > 1e-10:  # Non-zero coefficients
            print(f"Feature {name}: {coef:.6f}")
        else:
            print(f"Feature {name}: 0 (eliminated)")

