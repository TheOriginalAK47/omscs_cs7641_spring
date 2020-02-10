import sys

import time

from joblib import dump, load

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, f1_score, roc_curve, auc, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, plot_precision_recall_curve
from sklearn.model_selection import learning_curve, ShuffleSplit, KFold, RandomizedSearchCV
from sklearn.base import clone

from IPython.display import display, Image
import pydotplus
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.parasite_axes import host_subplot
from mpl_toolkits.axisartist.axislines import Axes

def store_model(model, model_path):
	dump(model, model_path)

def scorer(y_true, y_pred):
	return f1_score(y_true, y_pred)

def get_train_test_validation_split(feature_df, train_set_size=0.6, validation_set_size=0.1):
    train, validate, test = np.split(feature_df.sample(frac=1), [int(
        train_set_size * len(feature_df)), int((train_set_size + validation_set_size) * len(feature_df))])
    return train, validate, test

def plot_conf_matrix_wrapper(model, features, labels, label_names, ax):
	plot_obj = plot_confusion_matrix(model, features, labels, labels=label_names, normalize='true', ax=ax)

# Borrowed from sklearn tutorial/example
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring_metric=None):
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
    axes[0, 0].set_title(title)
    if ylim is not None:
        axes[0, 0].set_ylim(*ylim)
    axes[0, 0].set_xlabel("Training examples")
    axes[0, 0].set_ylabel("Score (F1)")
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, scoring=scoring_metric, n_jobs=n_jobs, \
                       train_sizes=train_sizes, \
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0, 0].grid()
    axes[0, 0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0, 0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0, 0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0, 0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0, 0].legend(loc="best")
    # Plot n_samples vs fit_times
    axes[0, 1].grid()
    axes[0, 1].plot(train_sizes, fit_times_mean, 'o-')
    axes[0, 1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[0, 1].set_xlabel("Training examples")
    axes[0, 1].set_ylabel("fit_times")
    axes[0, 1].set_title("Scalability of the model")
    # Plot fit_time vs score
    axes[1, 0].grid()
    axes[1, 0].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[1, 0].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[1, 0].set_xlabel("fit_times")
    axes[1, 0].set_ylabel("Score (F1)")
    axes[1, 0].set_title("Performance of the model")
    return plt

def plot_learning_curves_helper(model, features, labels, scoring_metric, plot_dir, problem_name):
	model_type = model.__class__.__name__
	fig, axes = plt.subplots(2, 2, figsize=(10, 15))
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	plot = plot_learning_curve(model, model_type + " Learning Curves", \
						features, labels, axes=axes, ylim=(0.4, 1.01), \
							cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10), \
							scoring_metric=scoring_metric)
	plt.savefig(plot_dir + problem_name + "_" + model_type + "_learning_curves.png")

def plot_model_complexity_graph(estimator, title_base_str, X, y, axes=None, ylim=None, cv=None, params_list=None):
	model_type = estimator.__class__.__name__
	validation_scores_list = []
	hyper_params_val_list = []
	time_complexities_list = []
	param_name = list(params_list[0].keys())[0]
	kf = KFold(n_splits=5)
	for param_setting in params_list:
		model = clone(estimator)
		model.set_params(**param_setting)
		param_setting_scores = []
		param_complexities_list = []
		for train_index, test_index in kf.split(X):
			train_features, test_features = X.iloc[train_index], X.iloc[test_index]
			train_labels, test_labels = y.iloc[train_index], y.iloc[test_index]
			model.fit(train_features, train_labels.values.ravel())
			start_time = time.time()
			model.predict(test_features)
			elapsed_time = (time.time() - start_time) / float(len(test_features.index))
			if (model_type == 'KMeans'):
				param_setting_scores.append(accuracy_score(test_labels, model.predict(test_features)))
			else:
				param_setting_scores.append(model.score(test_features, test_labels))
			param_complexities_list.append(elapsed_time)
		validation_scores_list.append(np.mean(param_setting_scores))
		param_value = list(param_setting.values())[0]
		hyper_params_val_list.append(param_value)
		time_complexities_list.append(np.mean(param_complexities_list))
	plt.figure(figsize=(9, 4.5))
	ax1 = plt.plot()
	line1 = plt.plot(hyper_params_val_list, validation_scores_list,'b-')
	plt.xlabel('Model Complexity (for parameter %s)' % param_name)
	plt.ylabel('Accuracy', color='b')
	ax2 = plt.twinx()
	line2 = ax2.plot(hyper_params_val_list, time_complexities_list, 'r-')
	plt.ylabel('Time (s)', color='r')
	plt.legend((line1[0], line2[0]), ('prediction acc.', 'latency'), loc='upper right')
	plt.title('Influence of Model Complexity - %s' % model_type)
	return plt

def plot_hyperparam_search_helper(model, features, labels, params_list, problem_name, plot_dir):
	cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
	model_type = model.__class__.__name__
	plot = plot_hyperparam_search(model, model_type + " Classifier Learning Curves", \
						features, labels, ylim=(0.4, 1.01), \
							cv=cv, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10), params_list=params_list)
	param_name = list(params_list[0].keys())[0]
	plt.savefig(plot_dir + problem_name + "_" + model_type + "_" + param_name + "_param_learning_curves.png")

def plot_complexity_curves_for_hyperparams(model, features, labels, params_list, problem_name, plot_dir):
	model_type = model.__class__.__name__
	plot = plot_model_complexity_graph(model, model_type + " Classifier Model Complexity Curves", \
						X=features, y=labels, ylim=(0.0, 1.01), params_list=params_list)
	param_name = list(params_list[0].keys())[0]
	plt.savefig(plot_dir + problem_name + "_" + model_type + "_" + param_name + "_param_complexity_curves.png")


def plot_opt_model_perf(model, features, labels, label_names, problem_name, plot_dir, axes=None):
	model_type = model.__class__.__name__
	if axes is None and model_type != 'KMeans':
		fig, axes = plt.subplots(2, 2, figsize=(10, 5))
		fig.subplots_adjust(hspace=0.3, wspace=0.2)
	else:
		plt.figure()
	if (model_type != 'SVC'):
		axes[0, 0].set_title(model_type + " ROC Curve")
		fpr, tpr, _ = roc_curve(labels, model.predict_proba(features)[:, 1])
		roc_auc = auc(fpr, tpr)
		lw = 2
		axes[0, 0].plot(fpr, tpr, color='darkorange',
					lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
		axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		axes[0, 0].set_xlim([0.0, 1.0])
		axes[0, 0].set_ylim([0.0, 1.05])
		axes[0, 0].set_xlabel('False Positive Rate')
		axes[0, 0].set_ylabel('True Positive Rate')
		axes[0, 0].legend(loc="lower right")
		rounded_auc = round(roc_auc, 5)
		conf_matrix_ax = axes[0, 1]
		axes[1, 0].axis('off')
		axes[1, 0].grid(b=None)
		prec_recall_curve_ax = axes[1, 1]
		plot_precision_recall_curve(model, features, labels, ax=prec_recall_curve_ax)
		plot_conf_matrix_wrapper(model, features, labels, label_names, conf_matrix_ax)
	else:
		rounded_auc = "N/A"
		cm = confusion_matrix(labels, model.predict(features), labels=label_names)
		fig, ax = plot_confusion_matrix(conf_mat=cm)
		plt.savefig(plot_dir + problem_name + "_" + model_type + "_conf_matrix.png")
	model_predictions = model.predict(features)
	model_acc = round(accuracy_score(labels, model_predictions), 5)
	precision = round(precision_score(labels, model_predictions), 5)
	recall = round(recall_score(labels, model_predictions), 5)
	f1 = round(f1_score(labels, model_predictions), 5)
	model_perf_array = [[str(model_acc), str(precision), str(recall), str(f1), str(rounded_auc)]]
	if (model_type != 'KMeans'):
		axes[1, 0].table(cellText=model_perf_array, cellLoc='center', rowLabels=None, colLabels=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC Score'], colLoc='center', loc='center')
	else:
		plt.table(cellText=model_perf_array, cellLoc='center', rowLabels=None, colLabels=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC Score'], colLoc='center', loc='center')
	plt.savefig(plot_dir + problem_name + "_" + model_type + "_test_set_performance.png")

# Utility function to report best scores
def report(results, n_top=3):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})"
				.format(results['mean_test_score'][candidate],
					results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate]))
			print("")

def perf_random_search_for_best_hyper_params(clf, features, labels, scoring_metric, param_dist, n_iter_search, n_jobs=None, cv=None):
	random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, scoring=scoring_metric, n_jobs=n_jobs, cv=cv)
	start = time.time()
	random_search.fit(features, labels.values.ravel())
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
		" parameter settings." % ((time.time() - start), n_iter_search))
	report(random_search.cv_results_)
	return random_search.best_params_

