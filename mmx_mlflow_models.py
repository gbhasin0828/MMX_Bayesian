"""# **Using a Basic Model**"""

from google.colab import files

# Upload a file
uploaded = files.upload()

# Display the uploaded file names
for filename in uploaded.keys():
    print(f'Uploaded file: {filename}')

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

df = pd.read_csv("data.csv")

X_columns = []

for i in df.columns:
  if i.startswith("mdsp_"):
    X_columns.append(i)



X = df[X_columns].values
y = df["sales"].values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


model_basic = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

model_basic.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

"""# ***Model with important transformations ***"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")

X_columns = []

for i in df.columns:
  if i.startswith("mdsp_"):
    X_columns.append(i)


X = df[X_columns].values
y = df["sales"].values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the Adstock transformation function
def adstock_transform(x, alpha):
    x = np.array(x).flatten()  # Ensure x is a 1D array
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]

    for t in range(1, len(x)):
        adstocked[t] = x[t] + alpha * adstocked[t-1]
    return adstocked

# Define the function to find optimal alpha for each media channel
def find_optimal_alpha_per_channel(X_media, y_sales, alpha_range=np.arange(0.2, 0.9, 0.1)):
    best_alphas = []

    # Ensure X_media is a 2D array
    X_media = np.array(X_media)
    y_sales = np.array(y_sales).flatten()

    if len(X_media.shape) != 2:
        raise ValueError("X_media must be a 2D array with shape (samples, channels).")

    for channel_idx in range(X_media.shape[1]):  # Loop through each media channel
        media_channel = X_media[:, channel_idx].flatten()  # Extract as a 1D array
        best_alpha = None
        best_mse = float('inf')  # Set the initial best MSE to infinity

        for alpha in alpha_range:
            # Apply Adstock transformation
            X_media_adstocked = adstock_transform(media_channel, alpha)

            # Fit a linear regression model
            model = LinearRegression()
            model.fit(X_media_adstocked.reshape(-1, 1), y_sales)
            y_pred = model.predict(X_media_adstocked.reshape(-1, 1))

            # Calculate MSE
            mse = mean_squared_error(y_sales, y_pred)

            # Update the best alpha if the current MSE is better
            if mse < best_mse:
                best_alpha = alpha
                best_mse = mse

        best_alphas.append(best_alpha)  # Add the best alpha for this channel to the list

    return best_alphas

# Example usage
# Ensure X_train and y_train are numpy arrays with the correct shapes
optimal_alphas = find_optimal_alpha_per_channel(X_train, y_train)
print("Optimal alphas for each channel:", optimal_alphas)

X_adstocked = np.array([adstock_transform(X[:, i], alpha) for i, alpha in enumerate(optimal_alphas)]).T

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_adstocked_scaled = scaler.fit_transform(X_adstocked)

model_advanced = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

model_advanced.fit(X_adstocked_scaled, y)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

"""# **MLFlow Model Experimentation**"""

!pip install mlflow

import mlflow
import mlflow.sklearn

mlflow.set_experiment("Random_Forest_Model_Experiments")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def log_experiment(name, model, X_train, X_test, y_train, y_test, additional_params = None):
  with mlflow.start_run(run_name = name):

    if hasattr(model, 'n_estimators'):
      mlflow.log_param("n_estimators", model.n_estimators)

    if hasattr(model, 'max_depth'):
      mlflow.log_param("max_depth", model.max_depth)

    if additional_params:
      for key, value in additional_params.items():
        mlflow.log_param(key, value)


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("Mean Squared Error", mse)
    mlflow.log_metric("Mean Absolute Error", mae)
    mlflow.log_metric("R-squared", r2)

    # Log the model
    mlflow.sklearn.log_model(model, artifact_path = "model")

log_experiment(
    name="Basic_Model",
    model=RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),  # Fixed: Added missing closing parenthesis
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

import mlflow

print("Tracking URI:", mlflow.get_tracking_uri())

import mlflow

# Set the tracking URI to the mlruns directory in your project
mlflow.set_tracking_uri("file:///Users/gauravbhasin/Desktop/DevOps_and_MLOps/MLOps/Test_Project/mlops/mlruns")

print("Updated Tracking URI:", mlflow.get_tracking_uri())

mlflow.set_experiment("Test_Experiment")

with mlflow.start_run():
    mlflow.log_param("param1", 123)
    mlflow.log_metric("metric1", 0.89)
    print("Run logged successfully.")

import mlflow

# Set tracking URI
mlflow.set_tracking_uri("file:///Users/gauravbhasin/Desktop/DevOps_and_MLOps/MLOps/Test_Project/mlops/mlruns")

# Set experiment name
mlflow.set_experiment("Test_Experiment")

# Log a simple run
with mlflow.start_run():
    mlflow.log_param("param1", 123)
    mlflow.log_metric("metric1", 0.89)
    print("Run logged successfully.")

# Set the correct tracking URI for MLflow
mlflow.set_tracking_uri("file:///Users/gauravbhasin/Desktop/DevOps_and_MLOps/MLOps/Test_Project/mlops/mlruns")

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def log_experiment(name, model, X_train, X_test, y_train, y_test, additional_params=None):
    """
    Logs an experiment to MLflow with the given model and data.

    Args:
        name (str): Name of the experiment/run.
        model (object): Machine learning model (e.g., RandomForestRegressor).
        X_train (array-like): Training features.
        X_test (array-like): Test features.
        y_train (array-like): Training labels.
        y_test (array-like): Test labels.
        additional_params (dict, optional): Additional parameters to log.

    Returns:
        None
    """
    try:
        # Set the correct tracking URI for MLflow
        mlflow.set_tracking_uri("file:///Users/gauravbhasin/Desktop/DevOps_and_MLOps/MLOps/Test_Project/mlops/mlruns")

        # Ensure the experiment exists or create it
        mlflow.set_experiment(name)

        # Start an MLflow run
        with mlflow.start_run(run_name=name):
            # Log model-specific parameters
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", model.n_estimators)

            if hasattr(model, 'max_depth'):
                mlflow.log_param("max_depth", model.max_depth)

            # Log additional parameters
            if additional_params:
                for key, value in additional_params.items():
                    mlflow.log_param(key, value)

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("Mean Squared Error", mse)
            mlflow.log_metric("Mean Absolute Error", mae)
            mlflow.log_metric("R-squared", r2)

            # Log the model as an artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"Experiment '{name}' logged successfully.")
    except Exception as e:
        print(f"Error logging experiment '{name}': {e}")

# Example Usage
# Assuming X_train, X_test, y_train, y_test are defined
log_experiment(
    name="Basic_Model",
    model=RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Set the tracking URI
mlflow.set_tracking_uri("file:///Users/gauravbhasin/Desktop/DevOps_and_MLOps/MLOps/Test_Project/mlops/mlruns")

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment
mlflow.set_experiment("Basic_Model")

# Start a run
with mlflow.start_run(run_name="Test_Run"):
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log metrics
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("Mean Squared Error", mse)

    # Log the model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Experiment logged successfully!")

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import logging

# Enable MLflow debug logs
logging.getLogger("mlflow").setLevel(logging.DEBUG)

# Set the correct tracking URI
mlflow.set_tracking_uri("file:///Users/gauravbhasin/Desktop/DevOps_and_MLOps/MLOps/Test_Project/mlops/mlruns")

# Check tracking URI
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Set experiment name
mlflow.set_experiment("Basic_Model")

# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run(run_name="Debug_Run"):
    # Log a model
    model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Experiment logged successfully!")

cat /Users/gauravbhasin/Desktop/DevOps_and_MLOps/MLOps/Test_Project/mlops/mlruns/0/meta.yaml

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set the correct tracking URI
mlflow.set_tracking_uri("file:///Users/gauravbhasin/Desktop/DevOps_and_MLOps/MLOps/Test_Project/mlops/mlruns")

# Set experiment to Basic_Model
experiment_name = "Basic_Model"
mlflow.set_experiment(experiment_name)

# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log the experiment
with mlflow.start_run(run_name="Run_Test"):
    # Log model parameters
    model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("max_depth", model.max_depth)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Log metrics
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    # Log the model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Experiment and run logged successfully!")

experiments = mlflow.list_experiments()
for exp in experiments:
    print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}, Location: {exp.artifact_location}")
