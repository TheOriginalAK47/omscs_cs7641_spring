import sys

import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split

from IPython.display import display, Image
import pydotplus
from sklearn import tree

from model_utils import plot_hyperparam_search, plot_hyperparam_search_helper


def perform_boosted_tree_hyperparameter_sweep(features, labels, problem_name, plot_dir):
    parameter_search_space = [[{'n_estimators': 25}, {'n_estimators': 50}, {'n_estimators': 75}, {'n_estimators': 100}], \
                                [{'learning_rate': 0.85}, {'learning_rate': 0.90}, {'learning_rate': 0.95}, {'learning_rate': 1.0}]]
    clf = AdaBoostClassifier()
    for param_set in parameter_search_space:
        plot_hyperparam_search_helper(clf, features, labels, param_set, problem_name, plot_dir)

features_and_labels_dataset_path = sys.argv[1]

plot_dir = sys.argv[2]

airbnb_features_data_df = pd.read_csv(features_and_labels_dataset_path)

sampled_dataset = airbnb_features_data_df

feature_cols = [x for x in sampled_dataset.columns if x != 'price' and x != 'expensive']

train, test = train_test_split(sampled_dataset, test_size=0.25, train_size=0.75)
train_features, train_labels = train[[col for col in feature_cols]], train[['expensive']]
test_features, test_labels = test[[col for col in feature_cols]], test[['expensive']]

clf = AdaBoostClassifier()

#pruned_ccp_alpha = prune_tree_model(train_features, train_labels, validation_features, validation_labels)

perform_boosted_tree_hyperparameter_sweep(train_features, train_labels, problem_name='Airbnb Rental Price Classification', plot_dir=plot_dir)