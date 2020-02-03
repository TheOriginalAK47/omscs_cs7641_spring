import sys

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import learning_curve, ShuffleSplit

def get_train_test_validation_split(feature_df, train_set_size=0.6, validation_set_size=0.1):
	train, validate, test = np.split(feature_df.sample(frac=1), [int(
		train_set_size * len(feature_df)), int((train_set_size + validation_set_size) * len(feature_df))])
	return train, validate, test

def prune_tree_model(train_features, train_labels, test_features, test_labels):
	clf = DecisionTreeClassifier()
	path = clf.cost_complexity_pruning_path(train_features, train_labels)
	ccp_alphas, impurities = path.ccp_alphas, path.impurities
	clfs = []
	best_clf = []
	best_score = 0.0
	for ccp_alpha in ccp_alphas:
		clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
		clf.fit(train_features, train_labels)
		clfs.append(clf)
		acc_score = clf.score(test_features, test_labels)
		print("Score: " + str(acc_score))
		if (acc_score > best_score):
			best_clf = [clf]
	return best_clf[-1]

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
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
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
						features, labels, axes=axes[:, 1], ylim=(0.3, 1.01), \
							cv=cv, n_jobs=4)
	plt.savefig("airbnb_rental_classifier_tree_learning_curves.png")

features_and_labels_dataset_path = sys.argv[1]

airbnb_features_data_df = pd.read_csv(features_and_labels_dataset_path)

sampled_dataset = airbnb_features_data_df

feature_cols = [x for x in sampled_dataset.columns if x != 'price' and x != 'expensive']

train, validate, test = get_train_test_validation_split(sampled_dataset, 0.6, 0.1)
train_features, train_labels = train[[col for col in feature_cols]], train[['expensive']]
validation_features, validation_labels = validate[[col for col in feature_cols]], validate[['expensive']]
test_features, test_labels = test[[col for col in feature_cols]], test[['expensive']]

clf = DecisionTreeClassifier()


clf.fit(train_features, train_labels, test_features, test_labels)

print(clf.feature_importances_)