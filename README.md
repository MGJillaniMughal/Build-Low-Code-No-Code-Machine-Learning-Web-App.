# Build-Low-Code-No-Code-Machine-Learning-Web-App.
# Predictive Analytics App

The Predictive Analytics App is a Streamlit web application that allows users to perform predictive analytics tasks on their CSV datasets. The app supports both classification and regression tasks using the PyCaret library for AutoML.

## Features

- Upload CSV data
- Select the type of task: Regression or Classification
- Data Cleaning: Impute missing values and encode categorical features using Label Encoding
- Exploratory Data Analysis (EDA): Generate a Pandas Profiling report
- Split data into training and test sets
- Run AutoML to automatically train and compare multiple machine-learning models
- Hyperparameter tuning for the best model
- Save and download the trained models

## How to Run

1. Install the required libraries listed in `requirements.txt` using pip:   
2. Run the Streamlit app:

3. 
3. The app will open in your default web browser. Upload your CSV data, select the target variable and features to drop, and proceed with the analysis.

4. For Classification Task:
- Click "Run AutoML" to train and compare multiple classification models.
- Evaluate the best model's accuracy and F1 score on the hold-out test set.
- Click "Tune Model" to perform hyperparameter tuning on the best model.
- Save the tuned model or download the AutoML model.

5. For Regression Task:
- Click "Run AutoML" to train and compare multiple regression models.
- Evaluate the best model's R-squared score on the hold-out test set.
- Click "Tune Model" to perform hyperparameter tuning on the best model.
- Save the tuned model or download the AutoML model.

## Data Format

The app expects the input data in CSV format. The target variable should be a single column in the dataset, and the rest of the columns are treated as features.

## Libraries Used

- pandas
- numpy
- streamlit
- pandas-profiling
- scikit-learn
- matplotlib
- seaborn
- pycaret

## Credits

This app was created by Jillani SoftTech.



