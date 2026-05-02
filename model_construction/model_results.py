"""
model_results.py
Description: Assembles CSV files containing detailed logs of model results.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
import os
import pandas as pd
from controllers.logs.color_logs import *


def log_model_result(results_df : pd.DataFrame, model_name : str):
    """
    Logs and saves model cross-validation results.

    :param model_name: Name of model being logged. (ex. KNN, Logistic Regression)
    :param results_df:
        A pandas DataFrame containing the cross-validation results for the model.
        The DataFrame must include columns such as 'params', 'mean_test_score',
        'std_test_score', 'rank_test_score', and any 'split' results from the model
        training process. This is given with 'grid_search.cv_results_'
    :return:
        None
    """

    log_columns = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    log_columns.extend([col for col in results_df.columns if 'split' in col])

    os.makedirs('./data/output_csv', exist_ok=True)

    results_df.to_csv(f'./data/output_csv/{model_name}_model_log.csv', index=False)
    print_succ(f"{model_name} Model results logged successfully!")


def log_to_ranking_list(model_name, params, score):
    """
    Updates the model ranking list by adding or replacing an entry for a specific model.
    Entries with the same model name are replaced with the new one.

    :param model_name: Name of the machine learning model to be logged
    :type model_name: str
    :param score: Best score attained by the model
    :type score: float
    :param params: Hyperparameters associated with the model's best performance
    :type params: dict
    :return: None
    :rtype: None
    :raises FileNotFoundError: If the specified ranking list file path does not exist
    """

    LIST_PATH = "./data/model_ranking.txt"

    if os.path.exists(LIST_PATH):
        with open(LIST_PATH, "r") as f:
            content = f.read()

        #Split content into entries using the dashed line as a delimiter
        entries = [e.strip() for e in content.split("-" * 30) if e.strip()]

        #Filter out the entry that matches the current model name
        filtered_entries = [e for e in entries if f"MODEL: {model_name}" not in e]
    else:
        filtered_entries = []

    new_entry = f"MODEL: {model_name}\nBEST SCORE: {score:.5f}\nBEST PARAMS: {params}"
    filtered_entries.append(new_entry)

    with open(LIST_PATH, "w") as f:
        for entry in filtered_entries:
            f.write("-" * 30 + "\n")
            f.write(entry + "\n")
            f.write("-" * 30 + "\n\n")

    print_succ(f"Model ranking list updated successfully!")