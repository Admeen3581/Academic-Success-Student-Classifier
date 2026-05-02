"""
model_constructor.py
Description: Outlines the fundamental steps of model construction.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from controllers.clean_dataset import get_unsplit_dataset
from controllers.logs.color_logs import print_succ
from model_construction.model_results import log_model_result, log_to_ranking_list


def train_model(model : BaseEstimator, model_name : str, param_grid : dict, folds : int) -> BaseEstimator:
    """
    General use machine learning model trainer using GridSearchCV for
    hyperparameter tuning and cross-validation.

    :param model: The machine learning model to be trained.
    :type model: BaseEstimator
    :param model_name: Name of the model for identification in logs.
    :type model_name: str
    :param param_grid: Dictionary with parameters names (`str`) as keys and lists
        of parameter settings to try as values.
    :type param_grid: dict
    :param folds: Number of cross-validation folds.
    :type folds: int
    :return: The best model estimator after GridSearchCV.
    :rtype: BaseEstimator
    """

    X_dataset, y_dataset = get_unsplit_dataset()

    # GridSearch is used because the sklearn cross_validation function doesn't return a complete model.

    # 6-Fold Cross Validation
    print(f"\n--- Performing {model_name} Model Execution... ---\n")
    grid_search = GridSearchCV(model, param_grid, cv=folds, scoring='accuracy', verbose=3, refit=True)

    # Fit the search
    grid_search.fit(X_dataset, y_dataset)

    best_model = grid_search.best_estimator_

    print("\n")
    log_model_result(pd.DataFrame(grid_search.cv_results_), model_name)
    log_to_ranking_list(model_name, grid_search.best_params_, grid_search.best_score_)

    print_succ(f"Best Average Score: {grid_search.best_score_:.5f}")

    return best_model