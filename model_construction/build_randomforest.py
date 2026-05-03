"""
build_randomforest.py
Description: Builds the Random Forest Model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
#Imports
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from model_construction.model_constructor import train_model

def build_forest_model(folds=6) -> BaseEstimator:
    """
    Builds and trains a Random Forest Tree model.

    This function initializes a Random Forest classifier, specifies a parameter grid for
    hyperparameter tuning, and trains the model using cross-validation.

    :param folds: Number of folds for cross-validation, default is 6.
    :type folds: int
    :return: Tuned Random Forest model after cross-validation.
    :rtype: Any
    """

    MODEL_NAME = "Random_Forest"

    student_forest_model = RandomForestClassifier()

    param_grid = {
        'criterion': ['gini'],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [3, 4, 5, 10, 20, 100],
        'min_samples_split' : [3, 6, 9, 12, 15],
        'ccp_alpha' : [0.001, 0.01, 0.1, 0.5, 1, 5],
        'n_estimators': [10, 25, 100],  # over 100 estimators is time consuming
    }

    best_model = train_model(student_forest_model, MODEL_NAME, param_grid, folds)

    return best_model