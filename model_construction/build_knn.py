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
from sklearn.neighbors import KNeighborsClassifier
from controllers.clean_dataset import *
from sklearn.model_selection import GridSearchCV
from model_construction.model_results import *
import pandas as pd


def build_knn_model(folds=6):
    """
    Builds and tunes a K-Nearest Neighbors (KNN) classification model
    using cross-validation and grid search. It leverages GridSearchCV for parameter
    optimization and evaluates the model performance on a given dataset.

    :raises ValueError: If the dataset for training has issues or is not suitable.
    :raises Exception: For unexpected issues during model training and validation steps.
    :param folds: Determines cross-validation folds. (Default: 6)
    :return pkl_model: The trained KNN model.
    """

    MODEL_NAME = "KNN"

    student_knn_model = KNeighborsClassifier()

    X_dataset, y_dataset = get_unsplit_dataset()

    #GridSearch is used because the sklearn cross_validation function doesn't return a complete model.
    param_grid = {'n_neighbors': [5, 10, 20, 30, 40, 50, 100, 200, 1000]}

    #6-Fold Cross Validation
    print(f"\n--- Performing {MODEL_NAME} Model Execution... ---\n")
    grid_search = GridSearchCV(student_knn_model, param_grid, cv=folds, scoring='accuracy', verbose=3, refit=True)

    #Fit the search
    grid_search.fit(X_dataset, y_dataset)

    best_model = grid_search.best_estimator_

    log_model_result(pd.DataFrame(grid_search.cv_results_), MODEL_NAME)
    log_to_ranking_list(MODEL_NAME, grid_search.best_params_, grid_search.best_score_)

    print_succ(f"Best K value: {grid_search.best_params_}")
    print_succ(f"Best Average Score: {grid_search.best_score_}")

    return best_model