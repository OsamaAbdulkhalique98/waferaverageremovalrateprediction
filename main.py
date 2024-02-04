from src.removal_rate_model import RemovalRateModel
from src.data_processing import read_and_combine, scale_data, prepare_data

# Use read_and_combine to combine CSV files into two DataFrames for training and test
training_data = read_and_combine()
test_data = read_and_combine("test")
outliers = True
training_inputs, training_outputs, test_inputs, test_outputs = prepare_data(training_data, test_data, outliers)

# Scale the data
scaled_training_inputs, scaled_test_inputs, scaled_training_outputs, scaled_test_outputs, output_scaler = scale_data(
    training_inputs, training_outputs, test_inputs, test_outputs
)
# Create an instance of RemovalRateModel for Random Forest
rf_model = RemovalRateModel()

# Train the Random Forest model
rf_model.train_model(rf_model.model_rf, scaled_training_inputs, scaled_training_outputs)

# Predict on the test data and inverse transform the predictions
predictions_rf = rf_model.predict_and_inverse_transform(rf_model.model_rf, scaled_test_inputs, output_scaler)

# Evaluate the model and get metrics
metrics_rf, relative_errors_rf = rf_model.evaluate_model(predictions_rf, scaled_test_outputs)

# Perform KS-test for the Random Forest model
ks_test_rf = rf_model.perform_ks_test(predictions_rf - scaled_test_outputs, 0, 1)


# Create an instance of RemovalRateModel for SVR Model
svr_model = RemovalRateModel()

# Train the SVR model
svr_model.train_model(svr_model.model_svr, scaled_training_inputs, scaled_training_outputs)

# Predict on the test data and inverse transform the predictions
predictions_svr = svr_model.predict_and_inverse_transform(svr_model.model_svr, scaled_test_inputs, output_scaler)

# Evaluate the model and get metrics
metrics_svr, relative_errors_svr = svr_model.evaluate_model(predictions_svr, scaled_test_outputs)

# Perform KS-test for the SVR model
ks_test_svr = svr_model.perform_ks_test(predictions_svr - scaled_test_outputs, 0, 1)


# Create an instance of RemovalRateModel for Lasso Model
lasso_model = RemovalRateModel()

# Train the SVR model
lasso_model.train_model(lasso_model.model_lasso, scaled_training_inputs, scaled_training_outputs)

# Predict on the test data and inverse transform the predictions
predictions_lasso = lasso_model.predict_and_inverse_transform(lasso_model.model_lasso, scaled_test_inputs, output_scaler)

# Evaluate the model and get metrics
metrics_lasso, relative_errors_lasso = lasso_model.evaluate_model(predictions_lasso, scaled_test_outputs)

# Perform KS-test for the Lasso model
ks_test_lasso = lasso_model.perform_ks_test(predictions_lasso - scaled_test_outputs, 0, 1)


# Calculate Errors for each model
errors_rf = predictions_rf - scaled_test_outputs
errors_svr = predictions_svr - scaled_test_outputs
errors_lasso = predictions_lasso -scaled_test_outputs
model_instance = RemovalRateModel()
results = model_instance.summarize_result(metrics_rf, metrics_svr, metrics_lasso)
print(results)
# Plot errors by Wafer ID
model_instance.plot_errors_by_wafer(errors_rf, errors_svr, errors_lasso)
# Plot errors by histogram with normal distribution
model_instance.plot_errors_by_histogram_with_normal(errors_rf, errors_svr, errors_lasso)
