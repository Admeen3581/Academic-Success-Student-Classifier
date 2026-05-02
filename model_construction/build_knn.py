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
import pandas as pd


def build_knn_model():


    student_knn_model = KNeighborsClassifier()

    X_dataset, y_dataset = get_unsplit_dataset()

    #GridSearch is used because the sklearn cross_validation function doesn't return a complete model.
    param_grid = {'n_neighbors': [5, 10, 20, 30, 40, 50, 100, 200, 1000]}

    #6-Fold Cross Validation
    print("\n--- Performing KNN Model Execution... ---\n")
    grid_search = GridSearchCV(student_knn_model, param_grid, cv=6, scoring='accuracy', verbose=3, refit=True)

    #Fit the search
    grid_search.fit(X_dataset, y_dataset)

    best_knn_model = grid_search.best_estimator_

    knn_model_result(pd.DataFrame(grid_search.cv_results_))

    print_succ(f"Best K value: {grid_search.best_params_}")
    print_succ(f"Best Score: {grid_search.best_score_}")

    return best_knn_model

def knn_model_result(results_df : pd.DataFrame):

    log_columns = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    log_columns.extend([col for col in results_df.columns if 'split' in col])

    print(results_df[log_columns].sort_values(by='rank_test_score'))

    os.makedirs('./data/output_csv', exist_ok=True)

    results_df.to_csv('knn_model_log.csv', index=False)