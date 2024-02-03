# main.py
from src.data_processing import read_and_combine, extract_features, add_output_column, prepare_data, scale_data
from src.model import WaferRemovalRateModel

def main():
    data_folder = "data/"
    data_name = "training"

    # Data processing
    data_processing = DataProcessing(data_folder, data_name)
    training_set, test_set = data_processing.read_and_combine()

    features_data = data_processing.extract_features(training_set)
    features_data = data_processing.add_output_column(features_data)

    training_inputs, training_outputs, test_inputs, test_outputs = data_processing.prepare_data(features_data, data_name)
    scaled_training_inputs, scaled_test_inputs, scaled_training_outputs, scaled_test_outputs = data_processing.scale_data(training_inputs, training_outputs, test_inputs, test_outputs)

    # Model training and evaluation
    model = WaferRemovalRateModel(data_folder, data_name)
    model.train_models()

    predictions_rf = model.predict_and_inverse_transform(model.model_rf, scaled_test_inputs, model.outputs_scaler)
    predictions_svr = model.predict_and_inverse_transform(model.model_svr, scaled_test_inputs, model.outputs_scaler)
    predictions_lasso = model.predict_and_inverse_transform(model.model_lasso, scaled_test_inputs, model.outputs_scaler)

    metrics_rf = model.evaluate_model(predictions_rf, model.outputs_scaler.inverse_transform(scaled_test_outputs.reshape(-1, 1)).flatten())
    metrics_svr = model.evaluate_model(predictions_svr, model.outputs_scaler.inverse_transform(scaled_test_outputs.reshape(-1, 1)).flatten())
    metrics_lasso = model.evaluate_model(predictions_lasso, model.outputs_scaler.inverse_transform(scaled_test_outputs.reshape(-1, 1)).flatten())

    # Display results or other actions as needed

if __name__ == "__main__":
    main()
