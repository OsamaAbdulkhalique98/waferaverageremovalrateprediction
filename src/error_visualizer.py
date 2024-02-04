import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import norm

class ErrorVisualizationAnalyzer:
    def summarize_result(self, metrics_rf, metrics_svr, metrics_lasso):
        """
        Summarize error metrics for different models in a DataFrame.

        Parameters:
        - metrics_rf: Error metrics for Random Forest model.
        - metrics_svr: Error metrics for SVR model.
        - metrics_lasso: Error metrics for Lasso model.

        Returns:
        - results: DataFrame summarizing error metrics for each model.
        """
        results = pd.DataFrame({"Random Forest": metrics_rf,
                                "SVR": metrics_svr,
                                "Lasso": metrics_lasso})
        return results.T

    def plot_errors_by_wafer(self, errors_rf, errors_svr, errors_lasso):
        """
        Plot the errors of each model for each Wafer ID.

        Parameters:
        - errors_rf: Errors for Random Forest model.
        - errors_svr: Errors for SVR model.
        - errors_lasso: Errors for Lasso model.
        """
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
        """
        Plot histograms of errors with normal distribution curve.

        Parameters:
        - errors_rf: Errors for Random Forest model.
        - errors_svr: Errors for SVR model.
        - errors_lasso: Errors for Lasso model.
        """
        plt.figure(figsize=(18, 15))

        # Create a list of error data and corresponding labels
        error_data = [errors_rf, errors_svr, errors_lasso]
        labels = ['RF Errors', 'SVR Errors', 'Lasso Errors']
        colors = ['blue', 'orange', 'green']

        for i, errors in enumerate(error_data):
            plt.subplot(3, 3, i + 1)
            self.plot_histogram(errors, bins=30, alpha=0.7, label=labels[i], color=colors[i])

            # Plot normal distribution curve
            mu, std = np.mean(errors), np.std(errors)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'r', linewidth=2)

            plt.title(f'{labels[i]} vs Normal Distribution')
            plt.xlabel('Error')
            plt.ylabel('Density')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()

    def plot_histogram(self, data, bins, alpha, label, color):
        """
        Plot a histogram for the given data.

        Parameters:
        - data: Error data.
        - bins: Number of bins for the histogram.
        - alpha: Transparency of the bars.
        - label: Label for the histogram.
        - color: Color of the bars.
        """
        plt.hist(data, bins=bins, alpha=alpha, density=True, label=label, color=color)

    def perform_ks_test(self, errors, mean_error, std_error):
        """
        Perform KS-test between the error distribution and a normal distribution.

        Parameters:
        - errors: Error data.
        - mean_error: Mean of the error distribution.
        - std_error: Standard deviation of the error distribution.

        Returns:
        - ks_test_result: Result of the KS-test.
        """
        ks_test_result = ks_2samp(errors, np.random.normal(mean_error, std_error, size=len(errors)))
        return ks_test_result
