# Code to evaluate the model

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
