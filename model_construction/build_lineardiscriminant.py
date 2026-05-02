"""
build_lineardiscriminant.py
Description: Builds the Linear Discrimination Analysis Model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
#Imports
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from model_construction.model_constructor import train_model

def build_lda_model(folds=6) -> BaseEstimator:
    """
    Builds and trains a Linear Discriminant Analysis (LDA) model.

    This function initializes a LDA classifier, specifies a parameter grid for
    hyperparameter tuning, and trains the model using cross-validation.

    :param folds: Number of folds for cross-validation, default is 6.
    :type folds: int
    :return: Tuned LDA model after cross-validation.
    :rtype: Any
    """

    MODEL_NAME = "Linear_Discriminant_Analysis"

    student_lda_model = LinearDiscriminantAnalysis()

    #Different solvers are available. Which is fastest?
    param_grid = {'solver' : ['svd', 'lsqr', 'eigen']}

    best_model = train_model(student_lda_model, MODEL_NAME, param_grid, folds)

    return best_model