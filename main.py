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

#Constants
MODEL_PATH = "./model/car_classifier.pt"

if __name__ == '__main__':
    print_blue("Hello There :)\nAcademic Success Classifier Ver.0.2\n\n\t---Initializing---\n\n")

    download_dataset()
    fix_features()

    build_knn_model()
    build_logistic_model()
    build_gaussian_model()
    build_lda_model()
    build_decision_model(folds=5)
    build_forest_model(folds=5)

    # dataset_init()
    #
    # if not os.path.exists(MODEL_PATH):
    #     #Best model was trained via AWS: Nvidia L40S GPU w/ 8 CPU cores.
    #     #25 epochs over 4 learning rate chunks off ResNet101 (see ModelTraining.py).
    #     #Training with the above setup typically takes ~1 hour.
    #     train_model(get_datasheet(), 4)
    # else:
    #     print("Model detected. Skipping training...")

    #validate_model(get_datasheet("./data/anno_test_filtered.csv"), 4, MODEL_PATH)