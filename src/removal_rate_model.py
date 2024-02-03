from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

class RemovalRateModel:
    def __init__(self):
        self.model_rf = RandomForestRegressor()
        self.model_svr = SVR()
        self.model_lasso = Lasso()

    def scale_data(self, inputs, outputs, test_inputs, test_outputs):
        """
        Scale inputs and outputs using MinMaxScaler.

        Parameters:
        - inputs: Training inputs
        - outputs: Training outputs
        - test_inputs: Test inputs
        - test_outputs: Test outputs

        Returns:
        - scaled_training_inputs: Scaled training inputs
        - scaled_test_inputs: Scaled test inputs
        - scaled_training_outputs: Scaled training outputs
        - scaled_test_outputs: Scaled test outputs
        """

        # Scaling inputs
        scaled_training_inputs = self.inputs_scaler.fit_transform(inputs)
        scaled_test_inputs = self.inputs_scaler.transform(test_inputs)

        # Scaling outputs. Reshape is used because fit_transform expects 2D array
        scaled_training_outputs = self.outputs_scaler.fit_transform(outputs.reshape(-1, 1)).flatten()
        scaled_test_outputs = self.outputs_scaler.transform(test_outputs.reshape(-1, 1)).flatten()

        return scaled_training_inputs, scaled_test_inputs, scaled_training_outputs, scaled_test_outputs

    def train_model(self, model, inputs, outputs):
        """Train the given model with the provided inputs and outputs."""
        model.fit(inputs, outputs)

    def predict_and_inverse_transform(self, model, test_inputs, output_scaler):
        """Predict on the test data and inverse transform the predictions."""
        predictions = model.predict(test_inputs)
        predictions_inv = output_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions_inv

    def evaluate_model(self, predictions, test_outputs_inv):
        """Calculate errors and additional metrics for the model."""
        errors = predictions - test_outputs_inv
        relative_errors = errors / np.maximum(np.abs(test_outputs_inv), 1e-8)

        metrics = {
            'Mean of Error': np.mean(errors),
            'Std of Error': np.std(errors),
            'Max of Error': np.max(errors),
            'MAE': mean_absolute_error(test_outputs_inv, predictions),
            'Mean Absolute Percentage Error': mean_absolute_percentage_error(test_outputs_inv, predictions),
            'Max Absolute Percentage Error': np.max(np.abs(errors / test_outputs_inv)),
            'MSE': mean_squared_error(test_outputs_inv, predictions),
            'SSE': np.sum(np.square(errors)),
            'R': np.corrcoef(test_outputs_inv, predictions)[0, 1],
            'R2 Score': r2_score(test_outputs_inv, predictions)
        }
        return metrics, relative_errors
    
    def summarize_result(self, metrics_rf, metrics_svr, metrics_lasso):
        results = pd.DataFrame({"Random Forest": metrics_rf,
                                "SVR": metrics_svr,
                                "Lasso": metrics_lasso})
        return results
    
    
    def plot_errors_by_wafer(self, errors_rf, errors_svr, errors_lasso):
        """Plot the errors of each model for each Wafer ID."""
        wafer_ids = range(len(errors_rf))  # Assuming errors_rf, errors_svr, and errors_lasso have the same length

        plt.figure(figsize=(18, 6))

        plt.plot(wafer_ids, errors_rf, alpha=0.5, label='RF Errors')
        plt.plot(wafer_ids, errors_svr, alpha=0.5, label='SVR Errors', color='orange')
        plt.plot(wafer_ids, errors_lasso, alpha=0.5, label='Lasso Errors', color='green')

        plt.title('Errors for Each Wafer ID')
        plt.xlabel('Wafer ID')
        plt.ylabel('Error')
        plt.legend()
        plt.grid()

        plt.show()

    def plot_errors_by_histogram_with_normal(self, errors_rf, errors_svr, errors_lasso):
        """Plot histograms of errors with normal distribution curve."""
        plt.figure(figsize=(18, 15))

        # Create a list of error data and corresponding labels
        error_data = [errors_rf, errors_svr, errors_lasso]
        labels = ['RF Errors', 'SVR Errors', 'Lasso Errors']
        colors = ['blue', 'orange', 'green']

        for i, errors in enumerate(error_data):
            plt.subplot(3, 3, i + 1)
            self.plot_histogram(errors, bins=30, alpha=0.7, label=labels[i], color=colors[i])

            plt.title(f'{labels[i]} vs Normal Distribution')
            plt.xlabel('Error')
            plt.ylabel('Density')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()

    def plot_histogram(self, data, bins, alpha, label, color):
        """Plot a histogram for the given data."""
        plt.hist(data, bins=bins, alpha=alpha, density=True, label=label, color=color)

    def perform_ks_test(self, errors, mean_error, std_error):
        """Perform KS-test between the error distribution and a normal distribution."""
        ks_test_result = ks_2samp(errors, np.random.normal(mean_error, std_error, size=len(errors)))
        return ks_test_result

