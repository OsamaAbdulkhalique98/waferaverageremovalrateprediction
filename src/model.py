import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def train_model(model, inputs, outputs):
    """
    Train a machine learning model.

    Parameters:
    - model: The machine learning model to train.
    - inputs: Training input data.
    - outputs: Training output data.
    """
    model.fit(inputs, outputs)

    
def predict_test(model, test_inputs):
    """
    Make predictions on test data using a trained model.

    Parameters:
    - model: The trained machine learning model.
    - test_inputs: Test input data.

    Returns:
    - predictions: Predictions on the test data.
    """
    predictions = model.predict(test_inputs)
    return predictions


def inverse_transform_predictions(predictions, output_scaler):
    """
    Inverse transform scaled predictions to the original scale.

    Parameters:
    - predictions: Scaled predictions.
    - output_scaler: Scaler used for output data.

    Returns:
    - inv_predictions: Inverse transformed predictions on the original scale.
    """
    inv_predictions = output_scaler.inverse_transform(predictions.reshape(-1,1)).flatten()
    return inv_predictions


def calculate_errors(predictions, test_output_inv):
    """
    Calculate absolute errors and relative errors.

    Parameters:
    - predictions: Predictions.
    - test_output_inv: Actual values on the original scale.

    Returns:
    - errors: Absolute errors.
    - relative_errors: Relative errors.
    """
    errors = predictions - test_output_inv
    relative_errors = errors / np.maximum(np.abs(test_output_inv), 1e-8)
    return errors, relative_errors


def get_metrics(errors, predictions_inv, test_output_inv):
    """
    Calculate various evaluation metrics based on errors and predictions.

    Parameters:
    - errors: Absolute errors.
    - predictions_inv: Inverse transformed predictions.
    - test_output_inv: Actual values on the original scale.

    Returns:
    - metrics: Dictionary containing calculated metrics.
    """
    metrics = {
        'Mean of Error': np.mean(errors),
        'Std of Error': np.std(errors),
        'Max of Error': np.max(errors),
        'MAE': mean_absolute_error(test_output_inv, predictions_inv),
        'Mean Absolute Percentage Error': mean_absolute_percentage_error(test_output_inv, predictions_inv),
        'Max Absolute Percentage Error': np.max(np.abs(errors / test_output_inv)),
        'MSE': mean_squared_error(test_output_inv, predictions_inv),
        'SSE': np.sum(np.square(errors)),
        'R': np.corrcoef(test_output_inv, predictions_inv)[0, 1],
        'R2 Score': r2_score(test_output_inv, predictions_inv)
    }
    return metrics
