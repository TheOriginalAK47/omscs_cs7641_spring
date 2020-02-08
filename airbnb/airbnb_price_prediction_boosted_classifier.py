import sys

import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split

from IPython.display import display, Image
import pydotplus
from sklearn import tree

from scipy import stats

from model_utils import perf_random_search_for_best_hyper_params, plot_opt_model_perf
from model_utils import plot_complexity_curves_for_hyperparams, plot_learning_curve, plot_learning_curves_helper, store_model


# param_dist = {'learning_rate': stats.uniform(0, 1), 'subsample': stats.uniform(0.7, 0.3)}

def sweep_hyperparameter_space_and_plot_complexity_curves(clf, features, labels, problem_name, plot_dir):
    parameter_search_space = [[{'n_estimators': 25}, {'n_estimators': 50}, {'n_estimators': 75}, {'n_estimators': 100}, {'n_estimators': 150}, \
    							{'n_estimators': 200}, {'n_estimators': 300}], \
                                [{'learning_rate': 0.80}, {'learning_rate': 0.825}, {'learning_rate': 0.85}, {'learning_rate': 0.875}, \
                                 {'learning_rate': 0.90}, {'learning_rate': 0.925}, {'learning_rate': 0.95}, {'learning_rate': 0.975}, {'learning_rate': 1.0}]]
    for hyperparam_range in parameter_search_space:
        plot_complexity_curves_for_hyperparams(clf, train_features,train_labels, hyperparam_range, \
                                                problem_name, plot_dir)

training_dataset_path = sys.argv[1]

test_dataset_path = sys.argv[2]

plot_dir = sys.argv[3]

model_path = sys.argv[4]

problem_name = 'airbnb_rental_price_classification'

train = pd.read_csv(training_dataset_path)

test = pd.read_csv(test_dataset_path)

feature_cols = [x for x in train.columns if x != 'price' and x != 'expensive']

train_features, train_labels = train[[col for col in feature_cols]], train[['expensive']]
test_features, test_labels = test[[col for col in feature_cols]], test[['expensive']]

clf = AdaBoostClassifier()

sweep_hyperparameter_space_and_plot_complexity_curves(clf, train_features, train_labels, problem_name, plot_dir)

param_dist = {'n_estimators': [25, 50, 75, 100, 150, 200, 300], 'learning_rate': stats.uniform(0.75, 0.25)}

opt_param_set_from_random_search = perf_random_search_for_best_hyper_params(clf, train_features, train_labels, param_dist, n_iter_search=20)

clf.set_params(**opt_param_set_from_random_search)

scoring_metric = 'f1'

plot_learning_curves_helper(clf, train_features, train_labels, scoring_metric, plot_dir, problem_name)

clf.fit(train_features, train_labels)

# generate_tree_config_diagram(clf, feature_cols)

plot_opt_model_perf(clf, test_features, test_labels, [0, 1], problem_name, plot_dir)

store_model(clf, model_path)