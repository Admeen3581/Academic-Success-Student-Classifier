"""
main.py
Description: Hello There :)

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
#Imports
from controllers.data_receiver import *
from controllers.logs.color_logs import *
from model_construction.build_decisiontree import build_decision_model
from model_construction.build_gaussianbayes import build_gaussian_model
from model_construction.build_lineardiscriminant import build_lda_model
from model_construction.build_logisticregression import *
from model_construction.build_knn import *
from controllers.clean_dataset import fix_features
from model_construction.build_randomforest import build_forest_model
from model_predict.kaggle_bonus import run_test
import joblib

#Constants
MODEL_PATH = "./model/student_success_model.pkl"

if __name__ == '__main__':
    print_blue("Hello There :)\nAcademic Success Classifier Ver.0.2\n\n\t---Initializing---\n\n")

    download_dataset()
    fix_features()

    build_knn_model()
    build_logistic_model()
    build_gaussian_model()
    build_lda_model()
    build_decision_model(folds=5)
    best_model = build_forest_model(folds=5)

    joblib.dump(best_model, MODEL_PATH)

    #Kaggle Bonus -- Top 2,000 competitors
    #https://www.kaggle.com/competitions/playground-series-s4e6/overview
    run_test()