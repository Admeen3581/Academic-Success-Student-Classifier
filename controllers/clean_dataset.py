"""
clean_dataset.py
Description: Cleans the dataset preserving key features & deleting NaN values.
Data comes in as Label Encoding via Integer values.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
#Imports
from controllers.csv_controller import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np


def fix_features():
    """
    Removes features that have no meaning towards the goal of the model.

    Dropped features relate to application processes, which are not related to the outcome.
    Renames features to be more descriptive.
    :return:
    """
    if os.path.exists('./data/processed_csv'):
        print("Dataset already cleaned. Skipping Cleaning.")
        print_warning("If you experience issues, delete the directory and try again.")
        return

    os.makedirs('./data/processed_csv', exist_ok=True)

    dataframe_train, dataframe_test = get_csv_data()
    dataframe_train.drop(
        columns = ['id', 'Application mode', 'Application order'],
        inplace = True
    )

    dataframe_train = dataframe_train.rename(columns={
        'Course' : 'Course Code',
        'Nacionality' : 'Nationality',
        'Displaced' : 'Relocated',
        'Curricular units 1st sem (credited)' : 'Course Credit Sem1 (transferred)',
        'Curricular units 2nd sem (credited)' : 'Course Credit Sem2 (transferred)',
        'Curricular units 1st sem (evaluations)' : 'Course Credit Sem1 (exams taken)',
        'Curricular units 2nd sem (evaluations)' : 'Course Credit Sem2 (exams taken)',
        'Curricular units 1st sem (approved)' : 'Course Credit Sem1 (passed)',
        'Curricular units 2nd sem (approved)' : 'Course Credit Sem2 (passed)',
        'Curricular units 1st sem (without evaluations)' : 'Course Credit Sem1 (skipped exams)',
        'Curricular units 2nd sem (without evaluations)' : 'Course Credit Sem2 (skipped exams)',
    })

    #Save to file
    dataframe_train.to_csv('./data/processed_csv/train.csv', index=False)

    dataframe_test.drop(
        columns = ['id', 'Application mode', 'Application order'],
        inplace = True
    )

    dataframe_test = dataframe_test.rename(columns={
        'Course' : 'Course Code',
        'Nacionality' : 'Nationality',
        'Displaced' : 'Relocated',
        'Curricular units 1st sem (credited)' : 'Course Credit Sem1 (transferred)',
        'Curricular units 2nd sem (credited)' : 'Course Credit Sem2 (transferred)',
        'Curricular units 1st sem (evaluations)' : 'Course Credit Sem1 (exams taken)',
        'Curricular units 2nd sem (evaluations)' : 'Course Credit Sem2 (exams taken)',
        'Curricular units 1st sem (approved)' : 'Course Credit Sem1 (passed)',
        'Curricular units 2nd sem (approved)' : 'Course Credit Sem2 (passed)',
        'Curricular units 1st sem (without evaluations)' : 'Course Credit Sem1 (skipped exams)',
        'Curricular units 2nd sem (without evaluations)' : 'Course Credit Sem2 (skipped exams)',
    })

    # Save to file
    dataframe_test.to_csv('./data/processed_csv/test.csv', index=False)

    print_succ("Dataset cleaned successfully!")

def get_split_dataset():
    """
    Split dataset from get_csv_data() into train, validate, and test sets.
    :return X_train, y_train, X_validate, y_validate, X_test, y_test: Metrics describing the model.
    """

    dataframe_train, _ = get_csv_data()

    X = dataframe_train.drop(columns=['Target'])

    y = dataframe_train['Target']

    X_train, y_train, X_temp, y_temp = train_test_split(X, y, test_size=0.2)
    X_validate, y_validate, X_test, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.fit_transform(X_validate)
    X_test_scaled = scaler.fit_transform(X_test)

    y_train = np.ravel(y_train)
    y_validate = np.ravel(y_validate)
    y_test = np.ravel(y_test)

    return X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test

def get_unsplit_dataset():
    """
    Obtains the dataset without splitting it, returning features and target variables.

    This function loads the training data using the `get_csv_data` function, separates
    features and target from the dataset, and then formats it for further usage.

    :return: A tuple containing the feature matrix and the flattened target array.
    :rtype: Tuple[pd.DataFrame, np.ndarray]
    """

    dataframe_train, _ = get_csv_data()

    X = dataframe_train.drop(columns=['Target'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = dataframe_train['Target']

    return X_scaled, np.ravel(y)