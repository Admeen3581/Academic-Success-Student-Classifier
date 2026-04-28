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
from controllers.data_receiver import *
from controllers.csv_controller import *


def fix_features():
    '''
    Removes features that have no meaning towards the goal of the model.

    Dropped features relate to application processes, which are not related to the outcome.
    Renames features to be more descriptive.
    :return:
    '''
    dataframe_train = get_csv_data(frametype='train')
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
    os.makedirs('./data/processed_csv', exist_ok=True)
    dataframe_train.to_csv('./data/processed_csv/train.csv', index=False)

    dataframe_test = get_csv_data(frametype='test')
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

    print("Dataset cleaned successfully!")
