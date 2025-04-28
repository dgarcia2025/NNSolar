import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Multiply, Lambda, LSTM, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from keras.saving import register_keras_serializable
from google.colab import drive
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import time

# Reading the .csv s (Substitute with your paths)
route_gen = 'Generation_solar_Belgium.csv'
route_angles = 'Solar_angle.csv'
route_weather = 'weather_data.csv'

solar_generation_df = pd.read_csv(route_gen, sep = ';')
weather_df = pd.read_csv(route_weather, sep= ',')
solar_angles_df = pd.read_csv(route_angles, sep=',')
weather_df.fillna(0)

# Preparing the data
# Since the genaration at night will be 0, we can simply take the values from 06:00 a.m to 22:00 p.m
solar_generation, capacity, timestamps, competition = [], [], [], []
for i, date in enumerate(solar_generation_df['DateTime']):
    hour = int(date[11]+date[12])
    if hour >= 6 and hour <= 21:
        solar_generation.append(solar_generation_df['Real-time Upscaled Measurement [MW]'][i])
        capacity.append(solar_generation_df['Monitored Capacity [MWp]'][i])
        timestamps.append(solar_generation_df['DateTime'][i])
        competition.append(solar_generation_df['Day-Ahead forecast (11h00) [MW]'][i])

solar_generation = np.array(solar_generation, dtype=str)
solar_generation = np.char.replace(solar_generation, ',', '.')
solar_generation = solar_generation.astype(float)
capacity = np.array(capacity, dtype=str)
capacity = np.char.replace(capacity, ',', '.')
capacity = capacity.astype(float)
competition = np.array(competition, dtype=str)
competition = np.char.replace(competition, ',', '.')
competition = competition.astype(float)

inputs_df = pd.DataFrame({'capacity': capacity})

for ii in range(5):
    tempertaure_2m, cloud_cover, cloud_cover_low, cloud_cover_high, cloud_cover_mid, weather_code = [], [], [], [], [], []

    for i, date in enumerate(weather_df['date']):
        hour = int(str(date)[11]+str(date)[12])
        if hour >= 6 and hour <= 21:
            tempertaure_2m.append(weather_df['temperature_2m_' + str(ii + 1)][i])
            cloud_cover.append(weather_df['cloud_cover_' + str(ii + 1)][i])
            cloud_cover_low.append(weather_df['cloud_cover_low_' + str(ii + 1)][i])
            cloud_cover_high.append(weather_df['cloud_cover_high_' + str(ii + 1)][i])
            cloud_cover_mid.append(weather_df['cloud_cover_mid_' + str(ii + 1)][i])
            weather_code.append(weather_df['weather_code_' + str(ii + 1)][i])

    tempertaure_2m = np.array(tempertaure_2m, dtype=float)
    cloud_cover = np.array(cloud_cover, dtype=float)
    cloud_cover_low = np.array(cloud_cover_low, dtype=float)
    cloud_cover_mid = np.array(cloud_cover_mid, dtype=float)
    cloud_cover_high = np.array(cloud_cover_high, dtype=float)
    weather_code = np.array(weather_code, dtype=int)

    weathers_df = pd.DataFrame({
        'temperature_2m_' + str(ii + 1) : np.repeat(tempertaure_2m, 4),
        'cloud_cover_' + str(ii + 1) : np.repeat(cloud_cover, 4),
        'cloud_cover_low_' + str(ii + 1) : np.repeat(cloud_cover_low, 4),
        'cloud_cover_mid_' + str(ii + 1) : np.repeat(cloud_cover_mid, 4),
        'cloud_cover_high_' + str(ii + 1) : np.repeat(cloud_cover_high, 4),
        'weather_code_' + str(ii + 1) : np.repeat(weather_code, 4)
    })

    inputs_df = inputs_df.join(weathers_df, how='inner')
    # inputs_df = pd.join([inputs_df, weathers_df], axis=1)

solar_angles = []

for i, date in enumerate(solar_angles_df['Time']):
    hour = int(str(date)[11]+str(date)[12])
    if hour >= 6 and hour <= 21:
        solar_angles.append(solar_angles_df['Altitude (degrees)'][i])
solar_angles = np.array(solar_angles, dtype=float)

# inputs_df = inputs_df.join(solar_angles_df.iloc[:, -2:], how='inner')
inputs_df['Altitude (degrees)'] = solar_angles

x_test = inputs_df.tail(int(len(inputs_df) * 0.15) + (1 + 4 * 11))
y_test = solar_generation[int(len(solar_angles) * 0.85) - (4 * 11):]
competition_test = competition[int(len(solar_angles) * 0.85) - (4 * 11):]
timestamps_test = timestamps[int(len(solar_angles) * 0.85) - (4 * 11):]

test_len_15 = int(0.2 * len(solar_angles) - 1) - 6*4

# First model, Feedforward

def build_neural_network(X_df, y_array, ignore_columns=None, hidden_layers=[64, 32],
                         activation='relu', learning_rate=0.001, epochs=100, batch_size=32,
                         verbose=1, random_state=42):
    """
    Build and train a neural network model based on input dataframe and NumPy target array

    Parameters:
    -----------
    X_df : pandas DataFrame
        Input dataframe containing features
    y_array : numpy.ndarray
        Target values as a NumPy array
    ignore_columns : list, optional
        List of column names to ignore
    hidden_layers : list, optional
        List of neurons in each hidden layer
    activation : str, optional
        Activation function for hidden layers
    learning_rate : float, optional
        Learning rate for Adam optimizer
    epochs : int, optional
        Number of training epochs
    batch_size : int, optional
        Batch size for training
    verbose : int, optional
        Verbosity mode (0, 1, or 2)
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing model, history, and data splits
    """
    # Make a copy of the dataframe to avoid modifying the original
    data = X_df.copy()

    # Ensure y is a numpy array
    y = np.array(y_array)

    # Remove ignored columns if specified
    if ignore_columns:
        data = data.drop(columns=[col for col in ignore_columns if col in data.columns])

    # Store original column names and their types for future prediction
    column_info = {col: str(data[col].dtype) for col in data.columns}
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Check for non-numeric columns and handle them
    for col in categorical_columns:
        print(f"Converting categorical column '{col}' to one-hot encoding")
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
        data = pd.concat([data, dummies], axis=1)
        data = data.drop(columns=[col])

    # Convert features to numpy array
    X = data.values

    # Split data into train, validation, and test sets (70/15/15)
    # X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=random_state)
    # Note: 0.1765 is approximately 15/85, which gives us a 15% validation set from the remaining 85% after test split

    n = len(X)
    test_size = int(n * 0.15)
    val_size = int(n * 0.15)
    train_size = n - test_size - val_size

    # Manual split preserving order
    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]


    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Determine input dimension
    input_dim = X_train.shape[1]

    # Build the model
    model = Sequential()

    # Input layer
    model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
    model.add(Dropout(0.2))

    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(0.2))

    # Output layer (linear activation for regression)
    model.add(Dense(1))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose
    )

    # Evaluate the model on test data
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Plot training history
    plot_training_history(history)

    # Create model metadata
    model_metadata = {
        'feature_names': data.columns.tolist(),
        'categorical_columns': categorical_columns,
        'column_info': column_info,
        'ignore_columns': ignore_columns
    }

    # Return the model, history, and data
    return {
        'model': model,
        'history': history,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'metadata': model_metadata
    }

def plot_training_history(history):
    """
    Plot the training and validation loss and MAE

    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        History object returned from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True)

    # Plot training & validation MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
