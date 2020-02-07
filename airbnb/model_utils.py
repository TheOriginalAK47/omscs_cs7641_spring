import sys

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import learning_curve, ShuffleSplit, KFold
from sklearn.base import clone

from IPython.display import display, Image
import pydotplus
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.parasite_axes import host_subplot
from mpl_toolkits.axisartist.axislines import Axes

def get_train_test_validation_split(feature_df, train_set_size=0.6, validation_set_size=0.1):
    train, validate, test = np.split(feature_df.sample(frac=1), [int(
        train_set_size * len(feature_df)), int((train_set_size + validation_set_size) * len(feature_df))])
    #X_train, X_test, y_train, y_test = train_test_split(feature_df[feature_cols], feature_df['loan_outcome'], test_size=0.25)
    return train, validate, test

def plot_hyperparam_search(estimator, title_base_str, X, y, axes=None, ylim=None, cv=None,
							n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10), params_list=None):
	if axes is None:
		fig, axes = plt.subplots(2, 2, figsize=(15, 5))
		fig.subplots_adjust(hspace=0.6, wspace=0.2)
	first_title_str = title_base_str + "\n where " + list(params_list[0].keys())[0] + " = " + str(list(params_list[0].values())[0])
	second_title_str = title_base_str + "\n where " + list(params_list[1].keys())[0] + " = " + str(list(params_list[1].values())[0])
	third_title_str = title_base_str + "\n where " + list(params_list[2].keys())[0] + " = " + str(list(params_list[2].values())[0])
	fourth_title_str = title_base_str + "\n where " + list(params_list[3].keys())[0] + " = " + str(list(params_list[3].values())[0])
	axes[0, 0].set_title(first_title_str)
	axes[0, 1].set_title(second_title_str)
	axes[1, 0].set_title(third_title_str)
	axes[1, 1].set_title(fourth_title_str)
	if ylim is not None:
		axes[0, 0].set_ylim(*ylim)
	axes[0, 0].set_xlabel("Training examples")
	axes[0, 0].set_ylabel("Score")
	axes[0, 1].set_xlabel("Training examples")
	axes[0, 1].set_ylabel("Score")
	axes[1, 0].set_xlabel("Training examples")
	axes[1, 0].set_ylabel("Score")
	axes[1, 1].set_xlabel("Training examples")
	axes[1, 1].set_ylabel("Score")
	train_scores_avg_list = []
	train_scores_std_list = []
	test_scores_avg_list = []
	test_scores_std_list = []
	for param_setting in params_list:
		estimator.set_params(**param_setting)
		train_sizes, train_scores, test_scores, fit_times, _ = \
			learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
								train_sizes=train_sizes,
								return_times=True)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		train_scores_avg_list.append(train_scores_mean)
		train_scores_std_list.append(train_scores_std)
		test_scores_avg_list.append(test_scores_mean)
		test_scores_std_list.append(test_scores_std)
	#axes[0, 0].grid()
	axes[0, 0].fill_between(train_sizes, train_scores_avg_list[0] - train_scores_std[0],
							train_scores_avg_list[0] + train_scores_std[0], alpha=0.1,
							color="r")
	axes[0, 0].fill_between(train_sizes, test_scores_avg_list[0] - test_scores_std[0],
							test_scores_avg_list[0] + test_scores_std[0], alpha=0.1,
							color="g")
	axes[0, 1].fill_between(train_sizes, train_scores_avg_list[1] - train_scores_std[1],
							train_scores_avg_list[1] + train_scores_std[1], alpha=0.1,
							color="r")
	axes[0, 1].fill_between(train_sizes, test_scores_avg_list[1] - test_scores_std[1],
						test_scores_avg_list[1] + test_scores_std[1], alpha=0.1,
						color="g")
	axes[1, 0].fill_between(train_sizes, train_scores_avg_list[2] - train_scores_std[2],
						train_scores_avg_list[2] + train_scores_std[2], alpha=0.1,
						color="r")
	axes[1, 0].fill_between(train_sizes, test_scores_avg_list[2] - test_scores_std[2],
						test_scores_avg_list[2] + test_scores_std[2], alpha=0.1,
						color="g")
	axes[1, 1].fill_between(train_sizes, train_scores_avg_list[3] - train_scores_std[3],
						train_scores_avg_list[3] + train_scores_std[3], alpha=0.1,
						color="r")
	axes[1, 1].fill_between(train_sizes, test_scores_avg_list[3] - test_scores_std[3],
						test_scores_avg_list[3] + test_scores_std[3], alpha=0.1,
						color="g")
	axes[0, 0].plot(train_sizes, train_scores_avg_list[0], 'o-', color="r",
					label="Training score")
	axes[0, 0].plot(train_sizes, test_scores_avg_list[0], 'o-', color="g",
					label="Cross-validation score")
	axes[0, 1].plot(train_sizes, train_scores_avg_list[1], 'o-', color="r",
					label="Training score")
	axes[0, 1].plot(train_sizes, test_scores_avg_list[1], 'o-', color="g",
					label="Cross-validation score")
	axes[1, 0].plot(train_sizes, train_scores_avg_list[2], 'o-', color="r",
					label="Training score")
	axes[1, 0].plot(train_sizes, test_scores_avg_list[2], 'o-', color="g",
					label="Cross-validation score")
	axes[1, 1].plot(train_sizes, train_scores_avg_list[3], 'o-', color="r",
					label="Training score")
	axes[1, 1].plot(train_sizes, test_scores_avg_list[3], 'o-', color="g",
					label="Cross-validation score")
	axes[0, 0].legend(loc="best")
	axes[0, 1].legend(loc="best")
	axes[1, 0].legend(loc="best")
	axes[1, 1].legend(loc="best")
	return plt

def plot_model_complexity_graph(estimator, title_base_str, X, y, axes=None, ylim=None, cv=None, params_list=None):
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
			model.fit(train_features, train_labels)
			start_time = time.time()
			model.predict(test_features)
			elapsed_time = (time.time() - start_time) / float(len(test_features.index))
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
	model_type = model.__class__.__name__
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
						X=features, y=labels, ylim=(0.4, 1.01), params_list=params_list)
	param_name = list(params_list[0].keys())[0]
	plt.savefig(plot_dir + problem_name + "_" + model_type + "_" + param_name + "_param_complexity_curves.png")

def plot_opt_model_perf(model, features, labels, problem_name, plot_dir):


