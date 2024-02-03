import pandas as pd
from src.removal_rate_model import RemovalRateModel
from src.data_processing import read_and_combine, scale_data, prepare_data, extract_features
from sklearn.preprocessing import MinMaxScaler

# Use read_and_combine to combine CSV files into two DataFrames for training and test
training_data = read_and_combine()
test_data = read_and_combine("test")
training_inputs, training_outputs, test_inputs, test_outputs = prepare_data(training_data, test_data)

# print()
# print()
# print()


# # Use prepare_data to extract features, add output column, and create inputs/outputs for training and test sets
# training_inputs, training_outputs, test_inputs, test_outputs = prepare_data(training_data, test_data)
inputs_scaler = MinMaxScaler()
outputs_scaler = MinMaxScaler()

# # # Scaling inputs
scaled_training_inputs = inputs_scaler.fit_transform(training_inputs)
scaled_test_inputs = inputs_scaler.transform(test_inputs)

# # Scaling outputs. Reshape is used because fit_transform expects 2D array
scaled_training_outputs = outputs_scaler.fit_transform(training_outputs).flatten()
scaled_test_outputs = outputs_scaler.transform(test_outputs).flatten()
print(scaled_training_inputs)
# # # Scale the data
# # scaled_training_inputs, scaled_test_inputs, scaled_training_outputs, scaled_test_outputs = scale_data(
# #     training_inputs, training_outputs, test_inputs, test_outputs
# # )
# # # Create an instance of RemovalRateModel
# # model = RemovalRateModel()

# # # Train the Random Forest model
# # model.train_model(model.model_rf, scaled_training_inputs, scaled_training_outputs)

# # # Predict on the test data and inverse transform the predictions
# # predictions_rf = model.predict_and_inverse_transform(model.model_rf, scaled_test_inputs)

# # # Evaluate the model and get metrics
# # metrics_rf = model.evaluate_model(predictions_rf, scaled_test_outputs)

# # # Perform KS-test for the Random Forest model
# # ks_test_rf = model.perform_ks_test(predictions_rf - scaled_test_outputs, 0, 1)

# # # Print the metrics and KS-test results
# # print("Random Forest Metrics:")
# # print(metrics_rf)
# # print("\nRandom Forest KS-Test Result:")
# # print(ks_test_rf)

# # # Plot errors by Wafer ID
# # model.plot_errors_by_wafer(predictions_rf - scaled_test_outputs, [], [])

# # # Plot errors by histogram with normal distribution
# # model.plot_errors_by_histogram_with_normal(predictions_rf - scaled_test_outputs, [], [])
