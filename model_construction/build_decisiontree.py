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

def build_lda_model(folds=6) -> BaseEstimator:
    """
    Builds and trains a Decision Tree model.

    This function initializes a Decision Tree classifier, specifies a parameter grid for
    hyperparameter tuning, and trains the model using cross-validation.

    :param folds: Number of folds for cross-validation, default is 6.
    :type folds: int
    :return: Tuned LDA model after cross-validation.
    :rtype: Any
    """

    MODEL_NAME = "Decision_Tree"

    student_decision_model = DecisionTreeClassifier()

    param_grid = {'criterion' : ['gini'], 'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'min_samples_split' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    best_model = train_model(student_decision_model, MODEL_NAME, param_grid, folds)

    return best_model