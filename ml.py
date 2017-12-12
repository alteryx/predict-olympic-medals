import pandas as pd
import numpy as np
from sklearn.metrics import (r2_score,
                             roc_auc_score,
                             mean_squared_error,
                             f1_score)
from sklearn.base import clone
import math
from collections import defaultdict

# In[28]:


def f1_micro(a, p):
    return f1_score(a, p, average='micro')


def roc_auc_score_threshold(a, p):
    score = roc_auc_score(a, p)
    return max(score, 1 - score)


def f1_score_threshold(a, p):
    score = f1_score(a, p)
    return max(score, 1 - score)


regression_scoring_funcs = {
    'r2': (r2_score, False),
    'mse': (mean_squared_error, False),
}

binary_scoring_funcs = {
    'roc_auc': (roc_auc_score_threshold, True),
    'f1': (f1_score_threshold, False),
}

binned_scoring_funcs = {
    'f1_micro': (f1_micro, False),
}


def score_predictions(splitter, prediction_series, y, y_binary, y_binned):
    X = prediction_series.values
    binary_X = (prediction_series.values >= 10).astype(int)
    binned_X, bins = bin_labels(prediction_series, [2, 6, 10, 50])

    scores_over_time = defaultdict(list)
    for _, test_i in splitter.split(y=y_binary):
        predicted = X[test_i]
        predicted_binary = binary_X[test_i]
        predicted_binned = binned_X[test_i]
        actual = y[test_i]
        actual_binary = y_binary[test_i]
        actual_binned = y_binned[test_i]
        for scoring_name, scoring_func in regression_scoring_funcs.items():
            sfunc = scoring_func[0]
            scores_over_time[scoring_name].append(sfunc(actual, predicted))
        for scoring_name, scoring_func in binary_scoring_funcs.items():
            sfunc = scoring_func[0]
            scores_over_time[scoring_name].append(sfunc(actual_binary, predicted_binary))
        for scoring_name, scoring_func in binned_scoring_funcs.items():
            sfunc = scoring_func[0]
            scores_over_time[scoring_name].append(sfunc(actual_binned, predicted_binned))
    return scores_over_time


def fit_and_score(X, y, splitter, estimator, _type='regression'):
    scores = []
    dates = []
    if _type == 'regression':
        scoring_funcs = regression_scoring_funcs
    elif _type == 'classification' and np.unique(y).shape[0] > 2:
        scoring_funcs = binned_scoring_funcs
    else:
        scoring_funcs = binary_scoring_funcs

    scores = defaultdict(list)
    for i, train_test in enumerate(splitter.split(X, y)):
        train, test = train_test
        train_dates, test_dates = splitter.get_dates(i, y=y)
        dates.append(test_dates[-1])
        cloned = clone(estimator)
        cloned.fit(X[train], y[train])
        actual = y[test]
        predictions = cloned.predict(X[test])
        try:
            probs = cloned.predict_proba(X[test])
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                probs = probs[:, 1]
        except:
            probs = None
        for name, scoring_func in scoring_funcs.items():
            sfunc, proba = scoring_func
            if proba:
                using = probs
            else:
                using = predictions
            scores[name].append(sfunc(actual, using))
    df = pd.DataFrame(scores)
    df['Olympics Year'] = dates
    return df


class TimeSeriesSplitByDate(object):
    def __init__(self,
                 dates,
                 n_splits=None,
                 combine_single_class_splits=True,
                 ignore_splits=None,
                 earliest_date=None):
        self.date_name = dates.name
        self.dates = dates.to_frame()
        self.combine_single_class_splits = combine_single_class_splits
        if n_splits is None:
            if earliest_date:
                dates = dates[dates >= earliest_date]
            n_splits = dates.nunique() - 1
        self.nominal_n_splits = n_splits
        self.earliest_date = earliest_date
        self.gen_splits()
        self.splits = None
        self.ignore_splits = ignore_splits

    def split(self, X=None, y=None, groups=None):
        if self.ignore_splits:
            if self.splits is None or (y != self.y).any():
                self.y = y
                self.splits = [
                    x for i, x in enumerate(self.nominal_splits)
                    if i not in self.ignore_splits
                ]
            return self.splits
        elif self.combine_single_class_splits:
            if self.splits is None or self.y is None or (y != self.y).any():
                self.y = y
                self.splits = []
                for i, train_test in enumerate(self.nominal_splits):
                    self.splits.append(train_test)
                    while len(self.splits) > 1 and self.single_class(
                            self.splits[-1], y):
                        last = self.splits.pop(-1)
                        penultimate = self.splits.pop(-1)
                        combined = []
                        for _last, _pen in zip(last, penultimate):
                            combined.append(
                                pd.concat([pd.Series(_last),
                                           pd.Series(_pen)]).drop_duplicates()
                                .sort_values())
                        self.splits.append(combined)
            return self.splits
        else:
            return self.nominal_splits

    def single_class(self, split, y):
        return pd.Series(y[split[1]]).nunique() == 1

    def get_dates(self, split, X=None, y=None, groups=None):
        if self.splits is None or (y != self.y).any():
            self.split(None, y)
        train_i, test_i = self.splits[split]
        return [
            self.split_index.iloc[ti][self.date_name].drop_duplicates()
            .tolist() for ti in (train_i, test_i)
        ]

    def get_split_by_date(self, date):
        date = pd.Timestamp(date)
        dates = self.split_index.drop_duplicates([self.date_name])
        split_index = dates[dates[self.date_name] == date]['split'].iloc[-1]
        return self.splits[split_index]

    def gen_splits(self):
        date_index = self.dates.drop_duplicates()
        if self.earliest_date:
            early_date_index = date_index[date_index[self.date_name] <
                                          self.earliest_date]
            early_date_index['split'] = 0
            date_index = date_index[date_index[self.date_name] >=
                                    self.earliest_date]
        date_index = date_index.reset_index(drop=True)
        date_index.index.name = 'split'
        date_index = date_index.reset_index(drop=False)
        div = math.ceil(len(date_index) / (self.nominal_n_splits + 1))
        date_index['split'] = (date_index['split'] / (div)).astype(int)
        if self.earliest_date:
            date_index = pd.concat([early_date_index, date_index])
        self.split_index = self.dates.merge(
            date_index, on=self.date_name, how='left')
        self.split_index.index = range(self.split_index.shape[0])
        splits = self.split_index['split']
        train_splits = [
            splits[splits < (i + 1)].index.values
            for i in range(self.nominal_n_splits)
        ]
        test_splits = [
            splits[splits == (i + 1)].index.values
            for i in range(self.nominal_n_splits)
        ]
        self.nominal_splits = list(zip(train_splits, test_splits))

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.split(None, y))


def bin_labels(labels, bin_edges):
    num_bins = len(bin_edges) + 1
    new_labels = [
        int(class_) for class_ in np.digitize(labels.values, bin_edges)
    ]
    bins_used = set()
    bins = []
    for i in range(num_bins):
        if i == 0:
            bins.append("<%.1f" % (bin_edges[0]))
        elif i + 1 == num_bins:
            bins.append(">=%.1f" % (bin_edges[-1]))
        else:
            bins.append("[%.1f,%.1f)" % (bin_edges[i - 1], bin_edges[i]))

    for i, lt in enumerate(new_labels):
        new_labels[i] = bins[int(lt)]
        bins_used.add(bins[int(lt)])
    bins = [b for b in bins if b in bins_used]
    return pd.Series(new_labels), bins
