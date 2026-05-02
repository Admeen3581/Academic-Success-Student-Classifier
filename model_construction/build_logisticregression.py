"""
build_logisticregression.py
Description: Builds the Logistic Regression Model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
#Imports
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from model_construction.model_constructor import train_model


def build_logistic_model(folds=6) -> BaseEstimator:
    """
    Builds and returns a logistic regression model optimized using the specified number of
    folds for cross-validation.

    :param folds: Number of folds for cross-validation. Default is 6.
    :type folds: int
    :return: The best logistic regression model after training.
    :rtype: LogisticRegression
    """

    MODEL_NAME = "Logistic_Regression"

    student_logistic_model = LogisticRegression()

    # single parameter needed.
    param_grid = {'solver': ['saga']}

    best_model = train_model(student_logistic_model, MODEL_NAME, param_grid, folds)

    return best_model