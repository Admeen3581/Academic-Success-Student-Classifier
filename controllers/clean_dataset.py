"""
main.py
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


def fix_outlier_features():
    '''
    Removes features that have no meaning towards the goal of the model.

    Dropped features relate to application processes, which are not related to the outcome.
    Renames features to be more descriptive.
    :return:
    '''
    dataframe = get_csv_data(frametype='train')
    dataframe.drop(
        columns = ['Application_mode', 'Application_type'],
        inplace = True
    )

    dataframe.rename(columns={
        'Course' : 'Course Code',
        'Nacionality' : 'Nationality',
        'Displaced' : 'Relocated',
        'Curricular units 1st sem (credited)' : 'Curricular units 1st sem (transferred)',
        'Curricular units 2nd sem (credited)' : 'Curricular units 2nd sem (transferred)',
        'Curricular units 1st sem (evaluations)' : 'Curricular units 1st sem (exams taken)',
        'Curricular units 2nd sem (evaluations)' : 'Curricular units 2nd sem (exams taken)',
        'Curricular units 1st sem (approved)' : 'Curricular units 1st sem (passed)',
        'Curricular units 2nd sem (approved)' : 'Curricular units 2nd sem (passed)',
        'Curricular units 1st sem (without evaluations)' : 'Curricular units 1st sem (skipped exams)',
        'Curricular units 2nd sem (without evaluations)' : 'Curricular units 2nd sem (skipped exams)',
    })
