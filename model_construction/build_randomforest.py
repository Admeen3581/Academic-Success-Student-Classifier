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

    student_forest_model = RandomForestClassifier(max_features='sqrt', n_estimators=25, criterion='gini')
    # over 100 estimators is time consuming. No payoff. Shows no change in results.
    # max_features = sqrt is better due to less diminishing returns.

    param_grid = {
        'max_depth' : [5, 8, 10, 12, 15, 50],
        'min_samples_split' : [10, 50, 75, 100, 250, 500],
        'ccp_alpha' : [0.001, 0.01, 0.1, 0.5, 1, 5],
    }

    best_model = train_model(student_forest_model, MODEL_NAME, param_grid, folds)

    return best_model