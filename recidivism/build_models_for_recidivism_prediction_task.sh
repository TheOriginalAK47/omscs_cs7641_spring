#!/bin/bash
echo "Running initial model dataset preparation script."
python3 recidivism_prediction_feature_prep.py Recidivism.csv recidivism_training_data.csv recidivism_test_data.csv

echo "Running script to find optimal model hyper-parameters, plot learning      curve, and complete model performance evaluation for the five different model   types on the Recidivism prediction use-case."
echo "Running process for Decision Tree model."
python3 recidivism_prediction_dt_classifier.py recidivism_training_data.csv  recidivism_test_data.csv plots/ models/recidivism_dt_classifier_model.joblib  recidivism_classification recidivism_flag
echo "Running process for Boosted model."
python3 recidivism_prediction_boosted_classifier.py recidivism_training_data.csv       recidivism_test_data.csv plots/ models/recidivism_boosted_classifier_model.joblib recidivism_classification recidivism_flag
echo "Running process for Neural Network model."
python3 recidivism_prediction_nn_classifier.py recidivism_training_data.csv       recidivism_test_data.csv plots/ models/recidivism_nn_classifier_model.joblib            recidivism_classification recidivism_flag
echo "Running process for K-Means model."
python3 recidivism_prediction_k_means_classifier.py recidivism_training_data.csv  recidivism_test_data.csv plots/ models/recidivism_k_means_classifier_model.joblib recidivism_classification recidivism_flag
echo "Running process for SVM model."
python3 recidivism_prediction_svm_classifier.py recidivism_training_data.csv       recidivism_test_data.csv plots/ models/recidivism_svm_classifier_model.joblib recidivism_classification recidivism_flag
