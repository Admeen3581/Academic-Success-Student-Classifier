"""
build_gaussianbayes.py
Description: Builds the Gaussian Bayes Model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
#Imports
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from model_construction.model_constructor import train_model

def build_gaussian_model(folds=6) -> BaseEstimator:
    """
    Builds and trains a Gaussian Naive Bayes model.

    This function initializes a Gaussian Bayes classifier and trains the model using cross-validation.

    :param folds: Number of folds for cross-validation, default is 6.
    :type folds: int
    :return: Tuned Gaussian model after cross-validation.
    :rtype: Any
    """

    MODEL_NAME = "Gaussian_Bayes"

    student_gaussian_model = GaussianNB()

    #empty intentionally, Bayes doesn't have hyperparameters.
    param_grid = {}

    best_model = train_model(student_gaussian_model, MODEL_NAME, param_grid, folds)

    return best_model