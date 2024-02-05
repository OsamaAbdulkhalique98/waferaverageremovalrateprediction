# Wafer Removal Rate Prediction

## Overview

This project predicts the average removal rate of wafers in a CMP (Chemical Mechanical Planarization) process. It employs machine learning models, including Random Forest, Support Vector Regression (SVR), and Lasso regression, to make accurate predictions.

## Project Structure

The project is organized into the following components:

- `data_processing.py`: Contains functions for reading and processing data.
- `model.py`: Contains different functions that handle the training and evaluation of machine learning models.
- `error_visualizer.py`: Contains a class named ErrorVisualizationAnalyzer, and is used to summarize the error metrics of each model and visualize the errors in different plots.
- `main.py`: The main script for running the project, including data processing, model training, and evaluation.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/OsamaAbdulkhalique98/waferaverageremovalrateprediction.git
   ```
2. **Execute the main.py file:**
   ```cmd
   python3 main.py
   ```
3. **Prompts:** You will be asked for some inputs such as:
  ```cmd
   Would you like to remove outliers (yes or no)? 
   Which STAGE would you like to select (A, B, or Both): 
  ```  
   The predictions vary based on these inputs. If you want the outliers to be removed, you will end up with higher accuracy.  
  
4. **Done!**
