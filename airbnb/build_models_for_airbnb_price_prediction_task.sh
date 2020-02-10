#!/bin/bash
echo "Performing cleanup of raw dataset to produce training and test data."
python3 airbnb_price_prediction_feature_prep.py AB_NYC_2019.csv airbnb_training_data.csv airbnb_test_data.csv  

echo "Running script to find optimal model hyper-parameters, plot learning curve, and complete model performance evaluation for the five different model types on the Airbnb Rental price prediction case."
echo "Running process for Decision Tree model."
python3 airbnb_price_prediction_dt_classifier.py airbnb_training_data.csv airbnb_test_data.csv plots/ models/airbnb_dt_classifier_model.joblib airbnb_rental_price_classification expensive
echo "Running process for Boosted model."
python3 airbnb_price_prediction_boosted_classifier.py airbnb_training_data.csv       airbnb_test_data.csv plots/ models/airbnb_boosted_classifier_model.joblib airbnb_rental_price_classification expensive
echo "Running process for Neural Network model."
python3 airbnb_price_prediction_nn_classifier.py airbnb_training_data.csv       airbnb_test_data.csv plots/ models/airbnb_nn_classifier_model.joblib airbnb_rental_price_classification expensive
echo "Running process for K-Means model."
python3 airbnb_price_prediction_k_means_classifier.py airbnb_training_data.csv       airbnb_test_data.csv plots/ models/airbnb_k_means_classifier_model.joblib airbnb_rental_price_classification expensive
echo "Running process for SVM model."
python3 airbnb_price_prediction_svm_classifier.py airbnb_training_data.csv       airbnb_test_data.csv plots/ models/airbnb_svm_classifier_model.joblib airbnb_rental_price_classification expensive

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
