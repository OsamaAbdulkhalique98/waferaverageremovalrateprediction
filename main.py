from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from src.error_visualizer import ErrorVisualizationAnalyzer
from src.data_processing import read_and_combine, scale_data, prepare_data
from src.model import train_model, predict_test, inverse_transform_predictions, calculate_errors, get_metrics

# Use read_and_combine to combine CSV files into two DataFrames for training and test
print("Reading and combining the training and test data files....")
training_data = read_and_combine()
test_data = read_and_combine("test")
print("Combined successfully!")

outliers = input("Would you like to keep the outliers? (yes or no): ")
while outliers.lower() not in ["yes", "no"]:
    print("I only accept yes or no!")
    outliers = input("Would you like to remove the outliers? (yes or no): ")
if outliers.lower() == "yes":    
    want_outliers = False
else:
    want_outliers = True

stage_choice = input("Select stage (a, b, or both): ")
while stage_choice.lower() not in ["a", "b", "both"]:
    print("I only accept a, b, or both!")
    stage_choice = input("Select stage (a, b, or both): ")
    
if stage_choice.lower() == "both":
    stage = None
else:
    stage = stage_choice.upper()

training_inputs, training_outputs, test_inputs, test_outputs = prepare_data(training_data, test_data, want_outliers, stage)

# Scale the data
scaled_training_inputs, scaled_test_inputs, scaled_training_outputs, scaled_test_outputs, output_scaler = scale_data(
    training_inputs, training_outputs, test_inputs, test_outputs
)

# Random Forest Model
rf_model = RandomForestRegressor()
train_model(rf_model, scaled_training_inputs, scaled_training_outputs)
predictions_rf = predict_test(rf_model, scaled_test_inputs)
predictions_rf_inv = inverse_transform_predictions(predictions_rf, output_scaler)
test_outputs_inv = inverse_transform_predictions(scaled_test_outputs,output_scaler)
errors_rf, relative_errors_rf = calculate_errors(predictions_rf_inv, test_outputs_inv)
metrics_rf = get_metrics(errors_rf,predictions_rf_inv,  test_outputs_inv)


# SVR Model
svr_model = SVR()
train_model(svr_model, scaled_training_inputs, scaled_training_outputs)
predictions_svr = predict_test(svr_model, scaled_test_inputs)
predictions_svr_inv = inverse_transform_predictions(predictions_svr, output_scaler)
test_outputs_inv = inverse_transform_predictions(scaled_test_outputs,output_scaler)
errors_svr, relative_errors_svr = calculate_errors(predictions_svr_inv, test_outputs_inv)
metrics_svr = get_metrics(errors_svr,predictions_svr_inv,  test_outputs_inv)


# Lasso Model
lasso_model = Lasso()
train_model(lasso_model, scaled_training_inputs, scaled_training_outputs)
predictions_lasso = predict_test(lasso_model, scaled_test_inputs)
predictions_lasso_inv = inverse_transform_predictions(predictions_lasso, output_scaler)
test_outputs_inv = inverse_transform_predictions(scaled_test_outputs,output_scaler)
errors_lasso, relative_errors_lasso = calculate_errors(predictions_lasso_inv, test_outputs_inv)
metrics_lasso = get_metrics(errors_lasso,predictions_lasso_inv,  test_outputs_inv)


# Model Instance for Plotting and Summarizing the results
visualize_model_instance = ErrorVisualizationAnalyzer()
results = visualize_model_instance.summarize_result(metrics_rf, metrics_svr, metrics_lasso)
print(results)

# Plot errors by Wafer ID
visualize_model_instance.plot_errors_by_wafer(errors_rf, errors_svr, errors_lasso)

# Plot errors by histogram with normal distribution
visualize_model_instance.plot_errors_by_histogram_with_normal(errors_rf, errors_svr, errors_lasso)
