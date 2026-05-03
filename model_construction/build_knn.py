"""
build_knn.py
Description: Builds the K-Nearest Neighbors Model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
#Imports
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from model_construction.model_constructor import train_model

def build_knn_model(folds=6) -> BaseEstimator:
    """
    Builds and trains a K-Nearest Neighbors (KNN) model.

    This function initializes a KNN classifier, specifies a parameter grid for
    hyperparameter tuning, and trains the model using cross-validation.

    :param folds: Number of folds for cross-validation, default is 6.
    :type folds: int
    :return: Tuned KNN model after hyperparameter tuning.
    :rtype: Any
    """

    MODEL_NAME = "KNN"

    student_knn_model = KNeighborsClassifier()

    param_grid = {'n_neighbors': [5, 10, 20, 30, 40, 50, 100, 1000]}

    best_model = train_model(student_knn_model, MODEL_NAME, param_grid, folds)

    return best_model