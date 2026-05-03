"""
kaggle_bonus.py
Description: For Kaggle submission.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
#Imports
import pandas as pd
import joblib
from controllers.csv_controller import get_csv_data

def run_test():
    """
    Processes test data, predicts outcomes using a pre-trained model, and saves the results
    to a CSV file.

    :return: None
    """
    _, test_csv = get_csv_data()
    sample_sub = pd.read_csv('./data/raw_csv/sample_submission.csv')

    scaler = joblib.load('./model/my_scaler.pkl')
    X_test_scaled = scaler.transform(test_csv)

    model = joblib.load("./model/student_success_model.pkl")
    predictions = model.predict(X_test_scaled)

    submission = pd.DataFrame({
        'id': sample_sub['id'],
        'Target': predictions
    })

    submission.to_csv('./data/final_submission.csv', index=False)