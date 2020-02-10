import sys

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split

from IPython.display import display, Image
import pydotplus
import matplotlib.pyplot as plt

from sklearn import tree

from model_utils import perf_random_search_for_best_hyper_params, plot_opt_model_perf
from model_utils import plot_complexity_curves_for_hyperparams, plot_learning_curve, plot_learning_curves_helper, store_model

def prune_tree_model(train_features, train_labels, test_features, test_labels):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5, min_samples_leaf=10)
    path = clf.cost_complexity_pruning_path(train_features, train_labels)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    clfs = []
    best_clf = []
    best_score = 0.0
    for ccp_alpha in ccp_alphas:
    	clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5, min_samples_leaf=10, random_state=0, ccp_alpha=ccp_alpha)
    	clf.fit(train_features, train_labels)
    	clfs.append(clf)
    	acc_score = clf.score(test_features, test_labels)
    	print("Score: " + str(acc_score))
    	if (acc_score > best_score):
    		best_clf = [clf]
    return best_clf[-1].ccp_alpha

def generate_tree_config_diagram(model, feature_names):
    model_type = model.__class__.__name__
    output_file_name = model_type + '.dot'
    dot_data = tree.export_graphviz(model, out_file = output_file_name,
                                 feature_names=feature_names, filled   = True
                                    , rounded  = True
                                    , special_characters = True)
    graph = pydotplus.graph_from_dot_file(output_file_name)  
    graph.write_png(model_type + "_diagram.png")

def sweep_hyperparameter_space_and_plot_complexity_curves(clf, features, labels, problem_name, plot_dir):
    parameter_search_space = [[{'max_depth': 5}, {'max_depth': 8}, {'max_depth': 12}, {'max_depth': 15}, \
                                             {'max_depth': 20}, {'max_depth': 25}, {'max_depth': 30}], \
                                [{'min_samples_split': 25}, {'min_samples_split': 50}, {'min_samples_split': 100}, \
                                 {'min_samples_split': 150}, {'min_samples_split': 200}, {'min_samples_split': 300}, {'min_samples_split': 400}, {'min_samples_split': 500}], \
                                [{'min_samples_leaf': 50}, {'min_samples_leaf': 100}, {'min_samples_leaf': 150}, {'min_samples_leaf': 200}, \
                                {'min_samples_leaf': 250}, {'min_samples_leaf': 325}, {'min_samples_leaf': 400}, {'min_samples_leaf': 500}]]
    for hyperparam_range in parameter_search_space:
        plot_complexity_curves_for_hyperparams(clf, features, labels, hyperparam_range, \
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

train_features, train_labels = train[feature_cols], train[[label_col_name]]
test_features, test_labels = test[feature_cols], test[[label_col_name]]

clf = DecisionTreeClassifier()

sweep_hyperparameter_space_and_plot_complexity_curves(clf, train_features, train_labels, problem_name, plot_dir)

param_dist = {'max_depth': [5, 8, 12, 15, 20, 25, 30], 'min_samples_split': [25, 50, 100, 150, 200, 300, 400, 500], \
              'min_samples_leaf': [50, 100, 150, 200, 250, 325, 400, 500]}

scoring_metric = 'f1'

opt_param_set_from_random_search = perf_random_search_for_best_hyper_params(clf, train_features, train_labels, scoring_metric, param_dist, n_iter_search=20, n_jobs=4, cv=5)

clf.set_params(**opt_param_set_from_random_search)


plot_learning_curves_helper(clf, train_features, train_labels, scoring_metric, plot_dir, problem_name)

clf.fit(train_features, train_labels.values.ravel())

generate_tree_config_diagram(clf, feature_cols)

plot_opt_model_perf(clf, test_features, test_labels, [0, 1], problem_name, plot_dir)

store_model(clf, model_path)