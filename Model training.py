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
route_gen = 'Generation_solar.csv'
route_angles = 'Solar_angle.csv'
route_weather = 'weather.csv'

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
class feedforwardNN:
    def __init__(self,
                 ignore_columns=None,
                 hidden_layers=None,
                 activation='relu',
                 learning_rate=0.001,
                 epochs=100,
                 batch_size=32,
                 verbose=1,
                 random_state=42):
        """
        Initialize the feedforward neural network parameters.
        """
        self.ignore_columns = ignore_columns or []
        self.hidden_layers = hidden_layers or [64, 32]
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

        # Attributes to be set during fit
        self.scaler = None
        self.model = None
        self.history = None
        self.metadata = {}
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

    def _prepare_data(self, X_df, y_array):
        # Copy and drop ignored columns
        data = X_df.copy()
        if self.ignore_columns:
            cols_to_drop = [col for col in self.ignore_columns if col in data.columns]
            data.drop(columns=cols_to_drop, inplace=True)

        # Record metadata
        self.metadata['column_info'] = {col: str(data[col].dtype) for col in data.columns}
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        self.metadata['categorical_columns'] = categorical_cols
        self.metadata['ignore_columns'] = self.ignore_columns

        # One-hot encode categoricals
        for col in categorical_cols:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            data.drop(columns=[col], inplace=True)

        # Features and target
        X = data.values
        y = np.array(y_array)
        n = len(X)
        test_size = int(n * 0.15)
        val_size = int(n * 0.15)
        train_size = n - test_size - val_size

        # Split (preserving order)
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        self.X_val = X[train_size: train_size + val_size]
        self.y_val = y[train_size: train_size + val_size]
        self.X_test = X[train_size + val_size:]
        self.y_test = y[train_size + val_size:]

        # Standardize
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

    def _build_model(self, input_dim):
        model = Sequential()
        # Input layer + dropout
        model.add(Dense(self.hidden_layers[0], activation=self.activation, input_dim=input_dim))
        model.add(Dropout(0.2))
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation=self.activation))
            model.add(Dropout(0.2))
        # Output layer
        model.add(Dense(1))  # Linear for regression

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def fit(self, X_df, y_array):
        """
        Prepare data, build model, and train.
        """
        # Data prep
        self._prepare_data(X_df, y_array)

        # Build model
        input_dim = self.X_train.shape[1]
        self.model = self._build_model(input_dim)

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        # Train
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=self.verbose
        )
        # Evaluate
        loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss (MSE): {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        return {'loss': loss, 'mae': mae}

    def plot_history(self):
        """
        Plot training & validation loss and MAE.
        """
        if self.history is None:
            raise ValueError("No training history. Call fit() first.")
        hist = self.history.history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(hist['loss'], label='Train Loss')
        ax1.plot(hist['val_loss'], label='Val Loss')
        ax1.set_title('Loss (MSE)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(hist['mae'], label='Train MAE')
        ax2.plot(hist['val_mae'], label='Val MAE')
        ax2.set_title('Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def save(self, model_path, scaler_path, metadata_path):
        """
        Save the trained model, scaler, and metadata to disk.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not found. Train the network before saving.")
        # Save Keras model
        self.model.save(model_path)
        # Save scaler and metadata
        import joblib
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.metadata, metadata_path)

# Second model, LSTM
class LSTMNN:
    def __init__(self,
                 sequence_length,
                 exclude_columns=None,
                 epochs=50,
                 batch_size=32,
                 verbose=1,
                 random_state=None):
        """
        Initialize LSTM neural network parameters.
        """
        self.sequence_length = sequence_length
        self.exclude_columns = exclude_columns or []
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

        # To be set during fit
        self.X_scaler = None
        self.y_scaler = None
        self.model = None
        self.history = None
        self.feature_cols = []
        self.metadata = {}
        self.data_splits = None

    def _create_sequences(self, X, y):
        xs, ys = [], []
        for i in range(len(X) - self.sequence_length):
            xs.append(X[i:i + self.sequence_length])
            ys.append(y[i + self.sequence_length])
        return np.array(xs), np.array(ys)

    def _build_model(self, input_shape, output_dim):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim))
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, df, y_array):
        """
        Prepare data, build LSTM model, and train.
        """
        # Ensure y is 2D
        y = y_array.reshape(-1, 1) if y_array.ndim == 1 else y_array
        if len(df) != len(y):
            raise ValueError(f"Feature rows {len(df)} != target rows {len(y)}")

        # Exclude columns
        self.feature_cols = [c for c in df.columns if c not in self.exclude_columns]
        X = df[self.feature_cols].values

        # Scale
        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)

        # Sequence generation
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        # Split (70/15/15)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False, random_state=self.random_state)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False, random_state=self.random_state)
        self.data_splits = (X_train, X_val, X_test, y_train, y_val, y_test)

        # Build & train
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = y_train.shape[1]
        self.model = self._build_model(input_shape, output_dim)
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        # Evaluate
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {loss}")
        return {'loss': loss}

    def plot_history(self):
        if self.history is None:
            raise ValueError("Call fit() before plotting.")
        h = self.history.history
        plt.figure(figsize=(12, 6))
        plt.plot(h['loss'], label='Train Loss')
        plt.plot(h['val_loss'], label='Val Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_predictions(self, which='test'):
        if self.model is None or self.data_splits is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_splits
        if which == 'test':
            X, y_true = X_test, y_test
            title = 'Test Set: Actual vs Predicted'
        elif which == 'val':
            X, y_true = X_val, y_val
            title = 'Validation Set: Actual vs Predicted'
        else:
            raise ValueError("which must be 'test' or 'val'.")
        y_pred_scaled = self.model.predict(X)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        y_actual = self.y_scaler.inverse_transform(y_true)
        plt.figure(figsize=(12, 6))
        plt.plot(y_actual, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save(self, model_path, X_scaler_path, y_scaler_path):
        if self.model is None:
            raise ValueError("Train model before saving.")
        # Save model and scalers
        self.model.save(model_path)
        joblib.dump(self.X_scaler, X_scaler_path)
        joblib.dump(self.y_scaler, y_scaler_path)
        
# Model 6 - Informers

class InformerModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def build_informer(self, input_shape, output_shape,
                       head_size=32, num_heads=1, ff_dim=64, num_transformer_blocks=2,
                       mlp_units=[32, 16], dropout=0.1):
        """
        Build an Informer-based model

        # ... (other arguments remain the same) ...
        """
        inputs = Input(shape=input_shape)
        x = inputs

        # First apply a dense layer to project the input
        x = Dense(head_size * num_heads, activation="relu")(x)

        # Informer uses distilling attention mechanism
        for _ in range(num_transformer_blocks):
            # Multi-head attention with simpler configuration
            attention_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=head_size,
                # Add the following line to explicitly set the attention axis
                attention_axes=1
            )(x, x)

            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(attention_output + x)

            # Feed Forward
            ff_output = Dense(ff_dim, activation="relu")(x)
            ff_output = Dropout(dropout)(ff_output)
            ff_output = Dense(head_size * num_heads)(ff_output)

            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(ff_output + x)

        # MLP Head for prediction
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(dropout)(x)

        outputs = Dense(output_shape)(x)

        return Model(inputs, outputs)

    def train_model(self, df, y, drop_columns=None,
                    validation_split=0.15, test_split=0.15,
                    epochs=100, batch_size=32,
                    patience=10, verbose=1):
        """
        Train the Informer model for solar forecasting

        Args:
            df: DataFrame with input features
            y: Numpy array with target values
            drop_columns: List of columns to drop from df
            validation_split: Fraction of data for validation (default 0.15)
            test_split: Fraction of data for testing (default 0.15)
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            verbose: Verbosity level for training

        Returns:
            Dictionary with trained model, history, and evaluation metrics
        """
        # Drop specified columns
        if drop_columns:
            df = df.drop(columns=drop_columns)

        # Define the split points
        train_size = int(len(df) * (1 - validation_split - test_split))
        val_size = int(len(df) * validation_split)

        # Split the data
        X_train = df.iloc[:train_size].values
        y_train = y[:train_size]

        X_val = df.iloc[train_size:train_size+val_size].values
        y_val = y[train_size:train_size+val_size]

        X_test = df.iloc[train_size+val_size:].values
        y_test = y[train_size+val_size:]

        # Normalize the data
        X_train = self.scaler_X.fit_transform(X_train)
        X_val = self.scaler_X.transform(X_val)
        X_test = self.scaler_X.transform(X_test)

        # Convert to numpy arrays
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()

        # Reshape y for the scaler
        y_train_2d = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        y_val_2d = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
        y_test_2d = y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test

        # Scale the targets
        y_train_scaled = self.scaler_y.fit_transform(y_train_2d)
        y_val_scaled = self.scaler_y.transform(y_val_2d)
        y_test_scaled = self.scaler_y.transform(y_test_2d)

        # Flatten if the original was 1D
        if len(y_train.shape) == 1:
            y_train_scaled = y_train_scaled.flatten()
            y_val_scaled = y_val_scaled.flatten()
            y_test_scaled = y_test_scaled.flatten()

        # Build the model
        input_shape = (X_train.shape[1],)
        output_shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]

        self.model = self.build_informer(input_shape, output_shape)

        # Model summary
        self.model.summary()

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mae']
        )

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )

        # Train the model
        start_time = time.time()
        self.history = self.model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        training_time = time.time() - start_time

        # Evaluate the model
        val_predictions_scaled = self.model.predict(X_val)
        test_predictions_scaled = self.model.predict(X_test)

        # Reshape predictions for inverse transform if needed
        if len(y_train.shape) == 1:
            val_predictions_scaled = val_predictions_scaled.reshape(-1, 1)
            test_predictions_scaled = test_predictions_scaled.reshape(-1, 1)

        # Inverse transform predictions
        val_predictions = self.scaler_y.inverse_transform(val_predictions_scaled)
        test_predictions = self.scaler_y.inverse_transform(test_predictions_scaled)

        # Flatten predictions if needed
        if len(y_train.shape) == 1:
            val_predictions = val_predictions.flatten()
            test_predictions = test_predictions.flatten()

        # Calculate metrics
        val_mae = mean_absolute_error(y_val, val_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)

        # Plot training history
        self.plot_training_history()

        # Return results
        return {
            'model': self.model,
            'history': self.history,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'training_time': training_time,
            'predictions': {
                'validation': val_predictions,
                'test': test_predictions
            },
            'actual': {
                'validation': y_val,
                'test': y_test
            },
            'X_scaler': self.scaler_X,
            'y_scaler': self.scaler_y
        }

    def plot_training_history(self):
        """Plot the training and validation loss and MAE"""
        plt.figure(figsize=(15, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.legend()
        plt.grid(True)

        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('MAE over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def predict(self, X_new):
        """Make predictions with the trained model"""
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values

        X_new_scaled = self.scaler_X.transform(X_new)
        predictions_scaled = self.model.predict(X_new_scaled)

        # Reshape if needed for inverse_transform
        if len(predictions_scaled.shape) == 1 or predictions_scaled.shape[1] == 1:
            predictions_scaled = predictions_scaled.reshape(-1, 1)

        predictions = self.scaler_y.inverse_transform(predictions_scaled)

        # Flatten output if it's a single feature
        if predictions.shape[1] == 1:
            predictions = predictions.flatten()

        return predictions
