import sys

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split

from IPython.display import display, Image
import pydotplus
from sklearn import tree

from model_utils import plot_hyperparam_search, plot_hyperparam_search_helper, get_train_test_validation_split

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

# Borrowed from sklearn tutorial/example
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(20, 5))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")
    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    return plt

def plot_learning_curves_helper(model, features, labels):
	fig, axes = plt.subplots(3, 2, figsize=(10, 15))
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	plot = plot_learning_curve(model, "Decision Tree Classifier Learning Curves", \
						features, labels, axes=axes[:, 1], ylim=(0.5, 1.01), \
							cv=cv, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10))
	plt.savefig("airbnb_rental_classifier_tree_learning_curves.png")

def generate_tree_config_diagram(model, feature_names, model_name='base'):
    output_file_name = 'tree_' + model_name + '.dot'
    dot_data = tree.export_graphviz(model, out_file = output_file_name,
                                 feature_names=feature_names, filled   = True
                                    , rounded  = True
                                    , special_characters = True)
    graph = pydotplus.graph_from_dot_file(output_file_name)  
    graph.write_png('tree_' + model_name + '.png')

def perform_tree_hyperparameter_sweep(features, labels, problem_name, plot_dir):
    parameter_search_space = [[{'max_depth': 5}, {'max_depth': 8}, {'max_depth': 12}, {'max_depth': 15}], \
                                [{'min_samples_split': 100}, {'min_samples_split': 150}, {'min_samples_split': 200}, {'min_samples_split': 250}], \
                                [{'min_samples_leaf': 10}, {'min_samples_leaf': 25}, {'min_samples_leaf': 35}, {'min_samples_leaf': 50}]]
    clf = DecisionTreeClassifier()
    for param_set in parameter_search_space:
        plot_hyperparam_search_helper(clf, features, labels, param_set, problem_name, plot_dir)

def sweep_hyperparameter_space_and_plot_complexity_curves(features, labels, problem_name, plot_dir):
    parameter_search_space = [[{'max_depth': 5}, {'max_depth': 8}, {'max_depth': 12}, {'max_depth': 15}, \
                                             {'max_depth': 20}, {'max_depth': 25}, {'max_depth': 30}], \
                                [{'min_samples_split': 25}, {'min_samples_split': 50}, {'min_samples_split': 100}, \
                                 {'min_samples_split': 150}, {'min_samples_split': 200}, {'min_samples_split': 300}, {'min_samples_split': 400}, {'min_samples_split': 500}], \
                                [{'min_samples_leaf': 50}, {'min_samples_leaf': 100}, {'min_samples_leaf': 150}, {'min_samples_leaf': 200}, \
                                {'min_samples_leaf': 250}, {'min_samples_leaf': 325}, {'min_samples_leaf': 400}, {'min_samples_leaf': 500}]]
    for hyperparam_range in parameter_search_space:
        plot_complexity_curves_for_hyperparams(c, train_features,train_labels, hyperparam_range, \
                                                'Airbnb Rental Price Classification', '../../omscs_ml_project1/airbnb/plots/')



features_and_labels_dataset_path = sys.argv[1]

plot_dir = sys.argv[2]

airbnb_features_data_df = pd.read_csv(features_and_labels_dataset_path)

sampled_dataset = airbnb_features_data_df

feature_cols = [x for x in sampled_dataset.columns if x != 'price' and x != 'expensive']

# get_train_test_validation_split(feature_df, train_set_size=0.6, validation_set_size=0.1)
train, validation, test = get_train_test_validation_split(sampled_dataset, train_set_size=0.6, validation_set_size=0.2)
train_features, train_labels = train[[col for col in feature_cols]], train[['expensive']]
validation_features, validation_labels = validation[[col for col in feature_cols]], validation[['expensive']]
test_features, test_labels = test[[col for col in feature_cols]], test[['expensive']]

clf = DecisionTreeClassifier()

#pruned_ccp_alpha = prune_tree_model(train_features, train_labels, validation_features, validation_labels)

# perform_tree_hyperparameter_sweep(train_features, train_labels, problem_name='Airbnb Rental Price Classification', plot_dir=plot_dir)

"""
pruned_clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5, min_samples_leaf=10, ccp_alpha=pruned_ccp_alpha)

pruned_clf.fit(train_features, train_labels)

print(pruned_clf.score(train_features, train_labels))

prepruned_clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features="log2", min_samples_split=200, min_samples_leaf=35, max_leaf_nodes=125)

prepruned_clf.fit(train_features, train_labels)
print(prepruned_clf.score(test_features, test_labels))

print(pruned_clf.feature_importances_)
"""

prepruned_clf = DecisionTreeClassifier(max_depth=8, min_samples_split=200, min_samples_leaf=35)


