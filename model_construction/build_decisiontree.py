"""
build_decisiontree.py
Description: Builds the Decision Tree Model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
#Imports
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from model_construction.model_constructor import train_model

def build_decision_model(folds=6) -> BaseEstimator:
    """
    Builds and trains a Decision Tree model.

    This function initializes a Decision Tree classifier, specifies a parameter grid for
    hyperparameter tuning, and trains the model using cross-validation.

    :param folds: Number of folds for cross-validation, default is 6.
    :type folds: int
    :return: Tuned Decision Tree model after cross-validation.
    :rtype: Any
    """

    MODEL_NAME = "Decision_Tree"

    student_decision_model = DecisionTreeClassifier()

    param_grid = {
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [5, 8, 10, 12, 15, 25, 100],
        'min_samples_split' : [10, 20, 40, 60, 80, 100],
    }

    best_model = train_model(student_decision_model, MODEL_NAME, param_grid, folds)

    return best_model