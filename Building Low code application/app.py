# app.py
import pandas as pd
import numpy as np
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, tune_model, plot_model, predict_model, save_model
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, tune_model as tune_model_reg, plot_model as plot_model_reg, predict_model as predict_model_reg, save_model as save_model_reg
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, r2_score

def upload_data():
    uploaded_file = st.file_uploader("Upload CSV data", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())
        return data
    return None

def select_task():
    task = st.selectbox("Select the type of task", ['Regression', 'Classification'])
    return task

def clean_data(df):
    st.write("Data Types:")
    st.write(df.dtypes)
    numeric_columns = df.select_dtypes(['int64', 'float64']).columns

    # Impute missing values
    for column in df.columns:
        if df[column].isna().sum() > 0:  # if there is a missing value in the column
            if column in numeric_columns:  # if the column is numeric
                df[column] = df[column].fillna(df[column].mean())  # fill with mean
                st.write(f"Filling missing values of {column} with mean {df[column].mean()}")
            else:  # if the column is categorical
                df[column] = df[column].fillna(df[column].mode()[0])  # fill with mode
                st.write(f"Filling missing values of {column} with mode {df[column].mode()[0]}")

    # Convert categorical columns to numerical (label encoding)
    label_encoders = {}  # to store label encoders for each column
    categorical_columns = df.select_dtypes(['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        st.write(f'Encoded {col} with values {list(le.classes_)} as {list(le.transform(le.classes_))}')

    st.write('Data after encoding:')
    st.write(df.head())

    return df

def perform_eda(df):
    pr = ProfileReport(df, explorative=True)
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)

def select_features(data):
    features_to_drop = st.multiselect("Select the features to drop", data.columns.tolist())
    target_variable = st.selectbox("Select the target variable", list(set(data.columns.tolist()) - set(features_to_drop)))
    return data.drop(features_to_drop, axis=1), target_variable

def handle_regression(data, target_variable, test_data_size):
    # Setting up PyCaret environment
    regression = setup_reg(data, target=target_variable, train_size=1-test_data_size, session_id=42, silent=True, verbose=False)

    best_model = None
    if st.button("Run AutoML"):
        best_model = compare_models_reg(sort='R2')  # You can change the optimization metric if needed
        st.write('AutoML Model:', best_model)

        # Display R-squared score and Feature Importance
        best_model_holdout_pred = predict_model_reg(best_model)

        # Adding 'Label' column to the predictions DataFrame
        best_model_holdout_pred['Label'] = data[target_variable]

        r2 = r2_score(best_model_holdout_pred['Label'], best_model_holdout_pred['Score'])

        st.write('R-squared score on hold-out: ', r2)

        plot_model_reg(best_model, plot='feature')
        st.pyplot(plt)

    # Hyperparameter tuning
    if st.button("Tune Model"):
        if best_model is not None:
            tuned_model = tune_model_reg(best_model, optimize='R2')
            st.write('Tuned Model:', tuned_model)

            # Display R-squared score and Feature Importance
            tuned_model_holdout_pred = predict_model_reg(tuned_model)
            r2_tuned = r2_score(tuned_model_holdout_pred['Label'], tuned_model_holdout_pred['Score'])
            st.write('R-squared score on hold-out after tuning: ', r2_tuned)

            plot_model_reg(tuned_model, plot='feature')
            st.pyplot(plt)

            # Display Tuned Model Parameters
            st.write('Tuned Model Parameters: ', tuned_model.get_params())

            # Save Model
            if st.button("Save Tuned Model"):
                model_name = st.text_input("Enter model name for the tuned model:")
                if model_name:
                    save_model_reg(tuned_model, model_name=model_name)
                    st.write(f"Tuned Model {model_name} saved successfully.")
        else:
            st.write("Please run AutoML before tuning the model.")

    if st.button("Download AutoML Model"):
        model_name = st.text_input("Enter model name for the AutoML model:")
        if model_name and best_model is not None:
            save_model_reg(best_model, model_name=model_name)
            st.write(f"AutoML Model {model_name} saved successfully.")

def handle_classification(data, target_variable, test_data_size):
    # Setting up PyCaret environment
    classification = setup(data, target=target_variable, train_size=1-test_data_size, session_id=42, silent=True, verbose=False)

    best_model = None
    if st.button("Run AutoML"):
        best_model = compare_models(sort='Accuracy')  # You can change the optimization metric if needed
        st.write('AutoML Model:', best_model)

        # Display Accuracy, F1 Score, and Feature Importance
        best_model_holdout_pred = predict_model(best_model)

        # Adding 'Label' column to the predictions DataFrame
        best_model_holdout_pred['Label'] = data[target_variable]

        accuracy = (best_model_holdout_pred['Label'] == best_model_holdout_pred['Score']).mean()
        f1 = f1_score(best_model_holdout_pred['Label'], best_model_holdout_pred['Score'], average='weighted')

        st.write('Accuracy on hold-out: ', accuracy)
        st.write('F1 Score on hold-out: ', f1)

        plot_model(best_model, plot='feature')
        st.pyplot(plt)

    # Hyperparameter tuning
    if st.button("Tune Model"):
        if best_model is not None:
            tuned_model = tune_model(best_model, optimize='F1')
            st.write('Tuned Model:', tuned_model)

            # Display Accuracy, F1 Score, and Feature Importance
            tuned_model_holdout_pred = predict_model(tuned_model)
            accuracy_tuned = (tuned_model_holdout_pred['Label'] == tuned_model_holdout_pred['Score']).mean()
            f1_tuned = f1_score(tuned_model_holdout_pred['Label'], tuned_model_holdout_pred['Score'], average='weighted')

            st.write('Accuracy on hold-out after tuning: ', accuracy_tuned)
            st.write('F1 Score on hold-out after tuning: ', f1_tuned)

            plot_model(tuned_model, plot='feature')
            st.pyplot(plt)

            # Display Tuned Model Parameters
            st.write('Tuned Model Parameters: ', tuned_model.get_params())

            # Save Model
            if st.button("Save Tuned Model"):
                model_name = st.text_input("Enter model name for the tuned model:")
                if model_name:
                    save_model(tuned_model, model_name=model_name)
                    st.write(f"Tuned Model {model_name} saved successfully.")
        else:
            st.write("Please run AutoML before tuning the model.")

    if st.button("Download AutoML Model"):
        model_name = st.text_input("Enter model name for the AutoML model:")
        if model_name and best_model is not None:
            save_model(best_model, model_name=model_name)
            st.write(f"AutoML Model {model_name} saved successfully.")

def main():
    st.title('Predictive Analytics App')

    data = upload_data()

    if data is not None:
        data, target_variable = select_features(data)

        st.write('Cleaning Data...')
        data = clean_data(data)
        st.write('Data after cleaning:')
        st.write(data.head())

        perform_eda(data)

        test_data_size = st.slider("Select the test data size", 0.1, 0.9, 0.3)  # default test size is 0.3

        task = select_task()
        if task == 'Regression':
            handle_regression(data, target_variable, test_data_size)
        elif task == 'Classification':
            handle_classification(data, target_variable, test_data_size)

if __name__ == "__main__":
    main()
