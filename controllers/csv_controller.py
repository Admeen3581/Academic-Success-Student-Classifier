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


def get_csv_data(frametype='train'):
    '''
    :type frametype: str
    :param frametype: 'train' or 'test' depending on CSV data required. 'train' is default.
    :return: Pandas dataframe
    '''
    return pd.read_csv(f'./data/raw_csv/{type}.csv')