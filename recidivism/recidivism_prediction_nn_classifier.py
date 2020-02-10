import sys

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split

from IPython.display import display, Image
import pydotplus
from sklearn import tree

from scipy import stats

from model_utils import perf_random_search_for_best_hyper_params, plot_opt_model_perf
from model_utils import plot_complexity_curves_for_hyperparams, plot_learning_curve, plot_learning_curves_helper, store_model


def sweep_hyperparameter_space_and_plot_complexity_curves(clf, features, labels, problem_name, plot_dir):
    parameter_search_space = [[{'hidden_layer_sizes': (25,)}, {'hidden_layer_sizes': (50,)}, {'hidden_layer_sizes': (100,)}, {'hidden_layer_sizes': (150,)}, {'hidden_layer_sizes': (200,)}], \
                                [{'activation': 'identity'}, {'activation': 'logistic'}, {'activation': 'tanh'}, {'activation': 'relu'}]]
    for hyperparam_range in parameter_search_space:
        plot_complexity_curves_for_hyperparams(clf, train_features,train_labels, hyperparam_range, \
                                                problem_name, plot_dir)

training_dataset_path = sys.argv[1]

test_dataset_path = sys.argv[2]

plot_dir = sys.argv[3]

model_path = sys.argv[4]

problem_name = sys.argv[5]

label_col_name = sys.argv[6]

train = pd.read_csv(training_dataset_path)

test = pd.read_csv(test_dataset_path)

feature_cols = [x for x in train.columns if x != label_col_name]

train_features, train_labels = train[[col for col in feature_cols]], train[[label_col_name]]
test_features, test_labels = test[[col for col in feature_cols]], test[[label_col_name]]

clf = MLPClassifier()

sweep_hyperparameter_space_and_plot_complexity_curves(clf, train_features, train_labels, problem_name, plot_dir)

param_dist = {'hidden_layer_sizes': [(50,), (100,), (150,)], 'activation': ['logistic', 'tanh', 'relu']}

scoring_metric = 'f1'

opt_param_set_from_random_search = perf_random_search_for_best_hyper_params(clf, train_features, train_labels, scoring_metric, param_dist, n_iter_search=10, n_jobs=-1, cv=3)

clf.set_params(**opt_param_set_from_random_search)

plot_learning_curves_helper(clf, train_features, train_labels, scoring_metric, plot_dir, problem_name)

clf.fit(train_features, train_labels.values.ravel())

plot_opt_model_perf(clf, test_features, test_labels, [0, 1], problem_name, plot_dir)

store_model(clf, model_path)