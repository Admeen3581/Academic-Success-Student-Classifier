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
from sklearn.linear_model import LogisticRegression
from controllers.clean_dataset import *
from sklearn.model_selection import GridSearchCV
from model_construction.model_results import *
import pandas as pd


def build_logistic_model(folds=6):

    MODEL_NAME = "Logistic_Regression"

    student_logistic_model = LogisticRegression()

    X_dataset, y_dataset = get_unsplit_dataset()

    #GridSearch is used because the sklearn cross_validation function doesn't return a complete model.
    param_grid = {'solver': ['saga']}#single parameter needed.

    #6-Fold Cross Validation
    print(f"\n--- Performing {MODEL_NAME} Model Execution... ---\n")
    grid_search = GridSearchCV(student_logistic_model, param_grid, cv=folds, scoring='accuracy', verbose=3, refit=True)

    #Fit the search
    grid_search.fit(X_dataset, y_dataset)

    best_model = grid_search.best_estimator_

    print("\n")
    log_model_result(pd.DataFrame(grid_search.cv_results_), MODEL_NAME)
    log_to_ranking_list(MODEL_NAME, grid_search.best_params_, grid_search.best_score_)

    print_succ(f"Best Average Score: {grid_search.best_score_}")

    return best_model