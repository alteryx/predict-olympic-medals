import featuretools.primitives as ftypes
import featuretools as ft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
import itertools


def remove_low_information_features(feature_matrix, features=None):
    '''
    Select features that have at least 2 unique values and that are not all null
    Args:
        feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are
            feature names and rows are instances
        features (list[:class:`featuretools.PrimitiveBase`] or list[str], optional):
            List of features to select
    '''
    keep = [c for c in feature_matrix
            if (feature_matrix[c].nunique(dropna=False) > 1 and
                feature_matrix[c].dropna().shape[0] > 0)]
    feature_matrix = feature_matrix[keep]
    if features is not None:
        features = [f for f in features
                    if f.get_name() in feature_matrix.columns]
        return feature_matrix, features
    return feature_matrix


def feature_importances_as_df(fitted_est, columns):
    return (pd.DataFrame({
        'Importance': fitted_est.steps[-1][1].feature_importances_,
        'Feature': columns
    }).sort_values(['Importance'], ascending=False))


def build_seed_features(es):
    # Baseline 1
    total_num_medals = ftypes.Count(es['medals_won']['medal_id'], es['countries'])
    count_num_olympics = ftypes.NUnique(
        es['countries_at_olympic_games']['Olympic Games ID'], es['countries'])
    mean_num_medals = (
        total_num_medals / count_num_olympics).rename("mean_num_medals")

    # Number of medals in each Olympics
    olympic_id = ft.Feature(es['countries_at_olympic_games']['Olympic Games ID'],
                            es['medals_won'])
    num_medals_each_olympics = [
        ftypes.Count(
            es['medals_won']['medal_id'], es['countries'],
            where=olympic_id == i).rename("num_medals_olympics_{}".format(i))
        for i in es.get_all_instances('olympic_games')
    ]
    return num_medals_each_olympics, mean_num_medals


def get_feature_importances(estimator, feature_matrix, labels, splitter,
                            n=100):
    feature_imps_by_time = {}
    for i, train_test_i in enumerate(splitter.split(None, labels.values)):
        train_i, test_i = train_test_i
        train_dates, test_dates = splitter.get_dates(i, y=labels.values)
        X = feature_matrix.values[train_i, :]
        icols_used = [i for i, c in enumerate(X.T) if not pd.isnull(c).all()]
        cols_used = feature_matrix.columns[icols_used].tolist()

        X = X[:, icols_used]
        y = labels.values[train_i]
        clf = clone(estimator)
        clf.fit(X, y)
        feature_importances = feature_importances_as_df(clf, cols_used)
        feature_imps_by_time[test_dates[-1]] = feature_importances.head(n)

    return feature_imps_by_time


def plot_confusion_matrix(cm,
                          classes=[0, 1],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    Source:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
