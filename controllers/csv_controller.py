"""
csv_controller.py
Description: Central controller for CSV files & data.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import pandas as pd


def get_csv_data():
    '''
    Returns the cleaned CSV data.

    :type frametype: str
    :return: Pandas dataframe. First return is 'train.csv', second is 'test.csv'
    :raises FileNotFoundError: if the dataset is not found.
    '''
    try:
        return pd.read_csv(f'./data/processed_csv/train.csv'), pd.read_csv(f'./data/processed_csv/test.csv')
    except FileNotFoundError:
        print("Dataset not found. Check 'controllers/clean_dataset.py/fix_features()'.")
        exit(-1)

def get_raw_csv_data():
    '''
    Gets raw CSV data.

        :type frametype: str
        :return: Pandas dataframe. First return is 'train.csv', second is 'test.csv'
        :raises FileNotFoundError: if the dataset is not found.
    '''
    try:
        return pd.read_csv(f'./data/raw_csv/train.csv'), pd.read_csv(f'./data/raw_csv/test.csv')
    except FileNotFoundError:
        print("Dataset not found. Ensure the dataset was downloaded via 'controllers/data_receiver.py'/download_dataset()'.")
        exit(-1)