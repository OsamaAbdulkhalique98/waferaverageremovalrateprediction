import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def read_and_combine(folder: str = "training"):
    # Path to the folder containing CSV files
    folder_path = "../data/" + folder
    # List to store DataFrames from each CSV file
    dataFrames = []   
    # Get the list of CSV files in the folder
    csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]  
    # Loop through all CSV files
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)      
        # Check if the file ends with '.csv'
        if filename.endswith('.csv'):
            # Read each CSV file into a DataFrame and append to the list
            dataframe = pd.read_csv(file_path)
            dataFrames.append(dataframe)  
    # Combine all DataFrames into one
    combined_dataframe = pd.concat(dataFrames, ignore_index=True)    
    # Save the combined DataFrame to a new CSV file
    combined_dataframe.to_csv('../data/combined_output.csv', index=False)
    # Display the combined DataFrame
    return combined_dataframe


def extract_features(data: pd.DataFrame, id_column: str = "WAFER_ID", non_extracted_columns: list = ["TIMESTAMP", "WAFER_ID", "CHAMBER"]):
    unique_ids = data[id_column].unique() 
    data_rows = []
    for wafer in unique_ids:
        wafer_data = data[data["WAFER_ID"]==wafer]
        # Iterate through each numerical column and calculate the features using numpy
        features_np = {}
        for column in wafer_data.select_dtypes(include='number').columns:
            if column in non_extracted_columns:
                continue
            col_data = wafer_data[column].values  # Convert the column to a numpy array
            features_np.update({
                f'{column}_Mean': np.mean(col_data),
                f'{column}_Median': np.median(col_data),
                f'{column}_StdDev': np.std(col_data, ddof=1),  # ddof=1 for sample standard deviation
                f'{column}_Variance': np.var(col_data, ddof=1),  # ddof=1 for sample variance
                f'{column}_Minimum': np.min(col_data),
                f'{column}_Maximum': np.max(col_data),
                f'{column}_Range': np.ptp(col_data),  # Peak-to-peak is a simpler way to compute range
                f'{column}_Skewness': pd.Series(col_data).skew(),  # Using pandas for skew as numpy does not have a direct function
                f'{column}_Kurtosis': pd.Series(col_data).kurt(),  # Using pandas for kurtosis as numpy does not have a direct function
                f'{column}_25thPercentile': np.percentile(col_data, 25),
                f'{column}_50thPercentile': np.percentile(col_data, 50),
                f'{column}_75thPercentile': np.percentile(col_data, 75)
            })
        # Convert the features dictionary to a DataFrame
        # Since we want all features in one row, we use pd.DataFrame and specify the index [0]
        feature_df = pd.DataFrame([features_np])
        feature_df.insert(0, "WAFER_ID", wafer)
        feature_df.insert(1, "STAGE", np.unique(wafer_data["STAGE"])[0])
        feature_df.insert(2, "CHAMBER", np.unique(wafer_data["CHAMBER"])[0])

        data_rows.append(feature_df)
    extracted_data = pd.concat(data_rows)
    return extracted_data


def add_output_column(data: pd.DataFrame(),
                      data_name: str = "training"):
    output_data = pd.read_csv("../data/CMP-" + data_name + "-removalrate.csv")
    data = pd.merge(data, output_data, on=['WAFER_ID', 'STAGE'])
    return data

def prepare_data(training_set, test_set):
    # Extract features and add output column to the training set
    training_features = extract_features(training_set)
    training_features = add_output_column(training_features)

    # Extract features and add output column to the test set
    test_features = extract_features(test_set)
    test_features = add_output_column(test_features, "test")
    
    training_features["STAGE"] = training_features["STAGE"].replace({"A": 0, "B": 1})
    test_features["STAGE"] = test_features["STAGE"].replace({"A": 0, "B": 1})
    # Create training inputs and outputs
    training_inputs = training_features.drop(columns=['WAFER_ID', 'AVG_REMOVAL_RATE']).values
    training_outputs = training_features['AVG_REMOVAL_RATE'].values

    # Create test inputs and outputs
    test_inputs = test_features.drop(columns=['WAFER_ID', 'AVG_REMOVAL_RATE']).values
    test_outputs = test_features['AVG_REMOVAL_RATE'].values

    return training_inputs, training_outputs, test_inputs, test_outputs


def scale_data(training_inputs, training_outputs, test_inputs, test_outputs):
    """
    Scale inputs and outputs using MinMaxScaler.

    Parameters:
    - training_inputs: Training inputs
    - training_outputs: Training outputs
    - test_inputs: Test inputs
    - test_outputs: Test outputs

    Returns:
    - scaled_training_inputs: Scaled training inputs
    - scaled_test_inputs: Scaled test inputs
    - scaled_training_outputs: Scaled training outputs
    - scaled_test_outputs: Scaled test outputs
    """

    # Initialize scalers
    inputs_scaler = MinMaxScaler()
    outputs_scaler = MinMaxScaler()

    # Scaling inputs
    scaled_training_inputs = inputs_scaler.fit_transform(training_inputs)
    scaled_test_inputs = inputs_scaler.transform(test_inputs)

    # Scaling outputs. Reshape is used because fit_transform expects 2D array
    scaled_training_outputs = outputs_scaler.fit_transform(training_outputs.reshape(-1, 1)).flatten()
    scaled_test_outputs = outputs_scaler.transform(test_outputs.reshape(-1, 1)).flatten()

    return scaled_training_inputs, scaled_test_inputs, scaled_training_outputs, scaled_test_outputs, outputs_scaler
