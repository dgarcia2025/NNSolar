# Code to evaluate the model

# Reading the datasets
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


# Function to evaluate a model
def evaluate_model(predictions, actual_generation):
    """
    Evaluate model predictions against actual values and return metrics

    Args:
        predictions: numpy array of predicted solar generation values
        actual_generation: numpy array of actual solar generation values

    Returns:
        dict: Evaluation metrics
    """
    # Calculate metrics
    mae = mean_absolute_error(actual_generation, predictions)
    rmse = np.sqrt(mean_squared_error(actual_generation, predictions))
    r2 = r2_score(actual_generation, predictions)
    mape = np.mean(np.abs((actual_generation - predictions) / (actual_generation + 1e-8))) * 100  # Adding small value to avoid division by zero

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'predictions': predictions,
        'actual': actual_generation
    }

def create_time_indices(data_length, time_increment=15):
    """
    Create mapping from hour to indices in the data

    Args:
        data_length: length of the data array
        time_increment: time increment in minutes (15 for 15-min data, 60 for hourly data)

    Returns:
        dict: Mapping from hour to list of indices
        list: Generated hour values for each data point
    """
    hour_indices = {}
    hours_list = []

    # Calculate how many data points per hour
    points_per_hour = 60 // time_increment

    # Calculate samples per day (from 6:00 to 21:45/21:00 inclusive)
    hours_per_day = 16  # 6:00 to 21:00/21:45
    samples_per_day = hours_per_day * points_per_hour

    for i in range(data_length):
        # Calculate day number and position within day
        day_num = i // samples_per_day
        position_in_day = i % samples_per_day

        # Calculate hour and minute
        hour_offset = position_in_day // points_per_hour
        current_hour = 6 + hour_offset

        # Only consider hours between 6 and 21
        if 6 <= current_hour <= 21:
            if current_hour not in hour_indices:
                hour_indices[current_hour] = []
            hour_indices[current_hour].append(i)
            hours_list.append(current_hour)

    return hour_indices, hours_list

def evaluate_by_hour(actual_generation, predictions, time_increment=15):
    """
    Evaluate model performance for each hour of the day

    Args:
        actual_generation: numpy array of actual values
        predictions: numpy array of predicted values
        time_increment: time increment in minutes (15 for 15-min data, 60 for hourly data)

    Returns:
        dict: Hourly evaluation metrics
    """
    hourly_metrics = {}

    # Create mapping from hour to indices
    hour_indices, _ = create_time_indices(len(actual_generation), time_increment)

    for hour in range(6, 22):  # From 6 to 21
        if hour in hour_indices:
            indices = hour_indices[hour]
            hour_actual = actual_generation[indices]
            hour_pred = predictions[indices]

            mae = mean_absolute_error(hour_actual, hour_pred)
            rmse = np.sqrt(mean_squared_error(hour_actual, hour_pred))
            r2 = r2_score(hour_actual, hour_pred) if len(np.unique(hour_actual)) > 1 else 0

            # Avoid division by zero in MAPE calculation
            mape = np.mean(np.abs((hour_actual - hour_pred) / (hour_actual + 1e-8))) * 100

            hourly_metrics[hour] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'count': len(indices),
                'mean_actual': np.mean(hour_actual),
                'mean_pred': np.mean(hour_pred)
            }

    return hourly_metrics

def plot_overall_performance(actual, predicted, title="Model Performance"):
    """
    Create scatter plot of predicted vs actual values

    Args:
        actual: numpy array of actual generation values
        predicted: numpy array of predicted values
        title: plot title

    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.scatter(actual, predicted, alpha=0.5)

    # Perfect prediction line
    max_val = max(np.max(actual), np.max(predicted))
    min_val = min(np.min(actual), np.min(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Actual Solar Generation')
    plt.ylabel('Predicted Solar Generation')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add metrics text
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    plt.figtext(0.15, 0.8, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    return plt.gcf()

def plot_hourly_metrics(hourly_metrics):
    """
    Create bar charts for hourly performance metrics

    Args:
        hourly_metrics: dictionary with hourly evaluation metrics

    Returns:
        matplotlib figure
    """
    hours = sorted(hourly_metrics.keys())
    metrics = ['MAE', 'RMSE', 'R2', 'MAPE']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [hourly_metrics[hour][metric] for hour in hours]

        axes[i].bar(hours, values)
        axes[i].set_xlabel('Hour of Day')
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'Hourly {metric}')
        axes[i].grid(True, alpha=0.3)

        # Add xticks for every hour
        axes[i].set_xticks(hours)

    plt.tight_layout()
    return fig

def plot_generation_profile(hourly_metrics, title="Average Generation Profile by Hour"):
    """
    Plot actual vs predicted average generation for each hour

    Args:
        hourly_metrics: dictionary with hourly evaluation metrics
        title: plot title

    Returns:
        matplotlib figure
    """
    hours = sorted(hourly_metrics.keys())
    actual_means = [hourly_metrics[hour]['mean_actual'] for hour in hours]
    pred_means = [hourly_metrics[hour]['mean_pred'] for hour in hours]

    plt.figure(figsize=(12, 6))

    plt.plot(hours, actual_means, 'o-', label='Actual')
    plt.plot(hours, pred_means, 's-', label='Predicted')

    plt.xlabel('Hour of Day')
    plt.ylabel('Average Solar Generation')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(hours)

    plt.tight_layout()
    return plt.gcf()

def plot_time_series(actual, predicted, time_increment=15, num_days=3):
    """
    Plot actual vs predicted values as a time series for the first few days

    Args:
        actual: numpy array of actual generation values
        predicted: numpy array of predicted values
        time_increment: time increment in minutes
        num_days: number of days to plot

    Returns:
        matplotlib figure
    """
    # Calculate samples per day (6:00 to 21:45/21:00)
    hours_per_day = 16  # 6:00 to 21:45/21:00
    points_per_hour = 60 // time_increment
    samples_per_day = hours_per_day * points_per_hour

    # Limit to specified number of days
    max_samples = num_days * samples_per_day
    if max_samples > len(actual):
        max_samples = len(actual)

    # Create figure
    plt.figure(figsize=(14, 6 * num_days))

    # Split into days for visualization
    days_to_plot = min(num_days, int(np.ceil(max_samples / samples_per_day)))

    for day in range(days_to_plot):
        plt.subplot(days_to_plot, 1, day + 1)

        start_idx = day * samples_per_day
        end_idx = min((day + 1) * samples_per_day, max_samples)

        # Extract data for this day
        day_actual = actual[start_idx:end_idx]
        day_pred = predicted[start_idx:end_idx]

        # Create x-axis labels (hours)
        day_minutes = np.arange(len(day_actual)) * time_increment
        day_hours = 6 + day_minutes / 60  # Starting from 6:00

        # Plot
        plt.plot(day_hours, day_actual, 'o-', label='Actual')
        plt.plot(day_hours, day_pred, 's-', label='Predicted')

        plt.title(f'Day {day + 1}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Solar Generation')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Set x-axis ticks to show hours
        plt.xticks(np.arange(6, 22, 1))

    plt.tight_layout()
    return plt.gcf()

def run_evaluation(predictions, actual_generation, time_increment=15): # 15 mins
    """
    Run the complete evaluation and generate all metrics and plots

    Args:
        predictions: numpy array of predicted solar generation values
        actual_generation: numpy array of actual solar generation values
        time_increment: time increment in minutes (15 for 15-min data, 60 for hourly data)

    Returns:
        dict: Complete evaluation results with metrics and figures
    """
    print("Evaluating model overall performance...")
    metrics = evaluate_model(predictions, actual_generation)

    print("Evaluating performance by hour...")
    hourly_metrics = evaluate_by_hour(actual_generation, predictions, time_increment)

    print("Generating plots...")
    overall_plot = plot_overall_performance(metrics['actual'], metrics['predictions'])
    hourly_plot = plot_hourly_metrics(hourly_metrics)
    profile_plot = plot_generation_profile(hourly_metrics)
    timeseries_plot = plot_time_series(metrics['actual'], metrics['predictions'], time_increment)

    print("\nOVERALL METRICS:")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R²: {metrics['R2']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    print("\nHOURLY METRICS SUMMARY:")
    for hour in sorted(hourly_metrics.keys()):
        print(f"Hour {hour}: MAE={hourly_metrics[hour]['MAE']:.4f}, "
              f"RMSE={hourly_metrics[hour]['RMSE']:.4f}, "
              f"R²={hourly_metrics[hour]['R2']:.4f}")

    return {
        'overall_metrics': metrics,
        'hourly_metrics': hourly_metrics,
        'plots': {
            'overall_performance': overall_plot,
            'hourly_metrics': hourly_plot,
            'generation_profile': profile_plot,
            'timeseries': timeseries_plot
        }
    }

# In case we want to apply some smoothing to the predictions

def smooth_array(array, strength=0.8):
    """
    Smooths an array by reducing sharp transitions between values.
    The function preserves the first and last values (assuming they're zeros).

    Parameters:
        array (list): The input array to smooth
        strength (float): Smoothing strength between 0 and 1 (higher = smoother)

    Returns:
        list: The smoothed array
    """
    if not 0 <= strength < 1:
        raise ValueError("Smoothing strength must be between 0 and 1")

    # Convert to numpy array for easier manipulation
    arr = np.array(array, dtype=float)
    smoothed = arr.copy()

    # Iteratively smooth the array
    for _ in range(10):  # Number of iterations affects smoothness
        # Create a temporary array to hold new values
        temp = smoothed.copy()

        # For each point (except first and last), adjust towards the average of neighbors
        for i in range(1, len(arr) - 1):
            # Calculate the average of the neighbors
            neighbor_avg = (smoothed[i-1] + smoothed[i+1]) / 2

            # Move the current value toward that average based on strength
            temp[i] = smoothed[i] * (1 - strength) + neighbor_avg * strength

        smoothed = temp

    return smoothed.tolist()

def smooth_forecast(forecasts, smoothing = 0.1, day_len = 22 - 6):
    for i in range(int(len(forecasts) / day_len)):
        forecasts[i * day_len * 4 : (i + 1) * day_len * 4] = smooth_array(forecasts[i * day_len * 4 : (i + 1) * day_len * 4], strength = smoothing)
    return forecasts

# Function used for prediction with model 1
def predict(df, model_dict):
    """
    Make predictions using the trained model

    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing features
    model_dict : dict, optional
        Dictionary containing model, scaler, and metadata
        (Either model_dict or model_dir must be provided)
    model_dir : str, optional
        Directory containing saved model components
        (Used if model_dict is not provided)

    Returns:
    --------
    numpy.ndarray
        Predicted values as a numpy array
    """

    # Get model components
    model = model_dict['model']
    scaler = model_dict['scaler']
    metadata = model_dict['metadata']

    # Make a copy of the input dataframe
    data = df.copy()

    # Remove ignored columns if specified
    if metadata['ignore_columns']:
        data = data.drop(columns=[col for col in metadata['ignore_columns'] if col in data.columns])

    # Process categorical columns
    for col in metadata['categorical_columns']:
        if col in data.columns:
            # Create dummies consistent with training data
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            data = data.drop(columns=[col])

    # Ensure all expected columns are present (fill missing with zeros)
    missing_cols = set(metadata['feature_names']) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    # Select only the columns that were used during training and in the same order
    data = data[metadata['feature_names']]

    # Convert to numpy array
    X = data.values

    # Scale the features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    # Return as a flat numpy array
    return predictions.flatten()

# Prediction function used for model 2
def predict_lstm(model, new_df, X_scaler, y_scaler, exclude_cols, sequence_length):
    """
    Make predictions using the trained LSTM model.

    Parameters:
    - model: trained LSTM model
    - new_df: pandas DataFrame containing features for prediction
    - X_scaler: fitted scaler for features
    - y_scaler: fitted scaler for target
    - exclude_cols: list of column names to exclude from features
    - sequence_length: sequence length used during training

    Returns:
    - numpy array of predictions
    """
    # Derive feature columns by excluding specified columns
    feature_cols = [col for col in new_df.columns if col not in exclude_cols]

    # Extract and scale features
    X = new_df[feature_cols].values
    X_scaled = X_scaler.transform(X)

    if len(X_scaled) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} samples to make a prediction, but got {len(X_scaled)}.")

    # Create sequences
    X_sequences = np.array([X_scaled[i:i + sequence_length] for i in range(len(X_scaled) - sequence_length + 1)])

    # Make predictions
    y_pred_scaled = model.predict(X_sequences)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    return y_pred

# Model 1
# Substitute this with actual paths to models
model_path = 'model1.keras'
model_w = keras.models.load_model(model_path)
scaler_path = 'scaler1.pkl'
scaler = joblib.load(scaler_path)
metadata_path = 'metadata1.pkl'
metadata = joblib.load(metadata_path)

model = {
    'model': model_w,
    'scaler': scaler,
    'metadata': metadata
}

# Executing evaluations

# As it is
# predictions = predict(x_test, model)

# Smoothed
# predictions = smooth_forecast(predict(x_test, model), smoothing=0.2)

# run_evaluation(predictions, y_test)


# Model 2
# Substitute with actual paths
# model_LSTM = load_model('model2.keras')
# scaler_x = joblib.load('scalerX2.pkl')
# scaler_y = joblib.load('scalery2.pkl')

# prediction_lstm = predict_lstm(model, x_test, X_scaler, y_scaler, [], 4 * (22 - 6))

# run_evaluation(predictions_lstm.T[0], y_test)


# # Model 3
# # Substitute with actual paths
# loaded_model = keras.models.load_model('informer_model.keras')
# loaded_attributes = joblib.load('informer_attributes.pkl')

# loaded_informer = InformerModel()
# loaded_informer.model = loaded_model
# loaded_informer.scaler_X = loaded_attributes['scaler_X']
# loaded_informer.scaler_y = loaded_attributes['scaler_y']
# loaded_informer.history = loaded_attributes['history']

# predictions = loaded_informer.predict(x_test)

# run_evaluation(predictions, y_test)
