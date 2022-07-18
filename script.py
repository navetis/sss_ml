import os
from itertools import chain, combinations
from math import ceil
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, average_precision_score, \
    brier_score_loss
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from statistics import mean, median
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
import ray

NAN_REPLACEMENT = 'knn'  # options: mean, median, knn
NAN_KNN_NEIGHBORS = 5  # should only be used, when knn for NaN replacement selected

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


SAMPLE_SELECTION = list(powerset([]))# powerset(['iqr', 'z_score'])  # options: iqr, z_score
SAMPLE_SELECTION_Q1 = 0.05  # adjust here if iqr chosen
SAMPLE_SELECTION_Q3 = 0.95
SAMPLE_SELECTION_Z = 3  # adjust if z_score chosen

# training & validation
ROUNDS = 50 #50
SAMPLE_SIZES = np.array([0.03, 0.05, 0.1, 0.25, 0.5, 1.0]) # np.linspace(0.05, 0.5, 2, endpoint=True) #1, 20
TRAIN_SIZE = np.array([0.3, 0.6, 0.8, 0.9]) # np.linspace(0.1, 0.6, 2, endpoint=False) #1, 2
VALIDATION_TYPES = np.array(['ts', 'all_nested', 'all_kfold', 'fs_nested_pt_kfold', 'fs_kfold_pt_nested']) #  ['ts', 'all_nested', 'all_kfold', 'fs_nested_pt_kfold', 'fs_kfold_pt_nested']  # pt: parameter tuning
FS_CV_SPLIT_SIZE = np.array([2, 3, 5, 8]) # np.linspace(2, 10, 2, endpoint=True).astype(int) #5
MODELS = np.array(['svm']) # 'knn', 'naive_bayes', 'random_forest', 'decision_tree', , 'logistic_regression'
PERFORMANCE_METRICS = np.array(['accuracy'])  # use sklearn scoring parameters; , 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score'
FEATURE_SELECTOR = np.array(['rfe']) # , 'sequential'
FEATURE_SELECTION_FRAC = np.array([0.25 , 0.5, 0.75, 1.0]) #np.linspace(0.1, 1, 2, endpoint=True)  # relevant for rfe and sequential, 10

# parameter ranges for models
PAR_SPLIT_SIZE = np.array([2, 3, 5, 8]) # np.linspace(2, 10, 2, endpoint=True).astype(int)
par_grid = {'svm': {'C': np.logspace(1, 7, num=7, base=2), 'gamma': np.logspace(-1, -7, num=7, base=2)},
            'logistic_regression': {'C': np.logspace(0, 4, num=10, base=10), 'penalty': ['l1', 'l2']},
            'random_forest': {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                              'max_depth': [
                                  [int(x) for x in np.linspace(10, 100, num=10, endpoint=True)].extend([None, 'sqrt'])],
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 2, 4],
                              'bootstrap': [True, False]},
            'decision_tree': {'max_depth': [[int(x) for x in np.linspace(10, 100, num=10, endpoint=True)].append(None)],
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 2, 4]},
            'knn': {'n_neighbors': np.linspace(1, 10, num=10, endpoint=True, dtype=int)},
            'naive_bayes': {'var_smoothing': np.logspace(0, -9, num=100)}}


def select_features(estimator, features, target, sel_type, frac):
    if sel_type == 'rfe':
        selector = RFE(estimator, n_features_to_select=ceil(frac * len(features.columns)), step=1)
        selector = selector.fit(features, target)
        return selector.support_
    elif sel_type == 'sequential':
        selector = SequentialFeatureSelector(estimator, n_features_to_select=ceil(frac * len(features.columns)))
        selector = selector.fit(features, target)
        return selector.support_


def tune_parameters(estimator, features, target, param_grid, split_size):
    cross_val = StratifiedKFold(n_splits=split_size, shuffle=True)
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid,
                           cv=cross_val)  # define a model with parameter tuning
    clf.fit(features, target)
    return clf.best_params_


def read_data():
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                    'ca', 'thal', 'target']
    data_path = os.path.join(os.getcwd(), "data/")

    cleveland_data = pd.read_csv(os.path.join(data_path, "processed.cleveland.data"), names=column_names, na_values='?')
    cleveland_data['location'] = 0

    hungarian_data = pd.read_csv(os.path.join(data_path, "reprocessed.hungarian.data"), na_values='-9',
                                 names=column_names, delimiter=' ')
    hungarian_data['location'] = 1

    switzerland_data = pd.read_csv(os.path.join(data_path, "processed.switzerland.data"), names=column_names, na_values='?')
    switzerland_data['location'] = 2

    va_data = pd.read_csv(os.path.join(data_path, "processed.va.data"), names=column_names, na_values='?')
    va_data['location'] = 3

    r = pd.concat([cleveland_data, hungarian_data, switzerland_data, va_data], axis=0, ignore_index=True)

    return pre_process(r)


def pre_process(data_to_process):
    r = data_to_process.astype(float, errors='raise')
    r.replace({'chol': 0, 'trestbps': 0}, value=np.NaN, inplace=True)  # chol and trestbps seem to have 0-values
    r['thal'].replace({3.0: 0.0, 6.0: 1.0, 7.0: 2.0}, inplace=True)
    r = r[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
           'ca', 'thal', 'location', 'target']]
    return r


def replace_nan(data_to_process):
    if NAN_REPLACEMENT == 'mean':
        return data_to_process.fillna(data_to_process.mean())
    elif NAN_REPLACEMENT == 'median':
        return data_to_process.fillna(data_to_process.median())
    elif NAN_REPLACEMENT == 'knn':
        imputer = KNNImputer(n_neighbors=NAN_KNN_NEIGHBORS)
        return pd.DataFrame(imputer.fit_transform(data_to_process), columns=data_to_process.columns)


def select_samples(data_to_process, sample_selection):
    selected_data = data_to_process
    for selection_type in sample_selection:
        if selection_type == 'iqr':
            q1 = selected_data.quantile(SAMPLE_SELECTION_Q1)
            q3 = selected_data.quantile(SAMPLE_SELECTION_Q3)
            iqr = q3 - q1
            selected_data = selected_data[~((selected_data < (q1 - 1.5 * iqr)) |
                                            (selected_data > (q1 + 1.5 * iqr))).any(axis=1)]
        elif selection_type == 'z_score':
            z = np.abs(stats.zscore(selected_data))
            selected_data = selected_data[np.logical_or(z < SAMPLE_SELECTION_Z, z.isna()).all(axis=1)]
    return selected_data


def select_features_estimator(name):
    if name == 'svm':
        return SVC(kernel='linear')
    elif name == 'random_forest':
        return RandomForestClassifier()
    elif name == 'decision_tree':
        return DecisionTreeClassifier()
    elif name == 'logistic_regression':
        return LogisticRegression(max_iter=10000, solver='liblinear')
    elif name == 'naive_bayes':
        return SVC(kernel='linear')
    elif name == 'knn':
        return SVC(kernel='linear')


def select_parameters_estimator(name):
    if name == 'naive_bayes':
        return GaussianNB()
    elif name == 'knn':
        return KNeighborsClassifier()
    else:
        return select_features_estimator(name)


def select_validation_estimator(name):
    if name == 'svm':
        return SVC()
    else:
        return select_parameters_estimator(name)


def measure_performance(metric, prediction, target):
    if metric == 'accuracy':
        return accuracy_score(prediction, target)
    elif metric == 'balanced_accuracy':
        return balanced_accuracy_score(prediction, target)
    elif metric == 'top_k_accuracy':
        return top_k_accuracy_score(prediction, target)
    elif metric == 'average_precision':
        return average_precision_score(prediction, target)
    elif metric == 'neg_brier_score':
        return brier_score_loss(prediction, target)


@ray.remote
def do_calc(model, sample_size, feature_selector, feature_selection_frac, performance_metric, validation_type, sample_selection, par_split_size, fs_cv_split_size, train_size):
    performance = []
    num_sample_selected = []
    for i in range(ROUNDS):
        j = 0
        tries = 0
        while j < 1:
            if tries >= 1:
                entry = pd.DataFrame({'model': model,
                          'sample_size': int(sample_size * len(data)),
                          'sample_selectors': ' '.join(sample_selection),
                          'sample_size_selected': np.NaN,
                          'feature_selector': feature_selector,
                          'num_feature_sel': ceil(feature_selection_frac * len(data.columns)),
                          'performance_metric': performance_metric,
                          'validation_type': validation_type,
                          'train_size': train_size,
                          'fs_cv_split_size': fs_cv_split_size_list,
                          'par_split_size': par_split_size,
                          'mean_performance': np.NaN,
                          'median_performance': np.NaN,
                          'max_performance': np.NaN,
                          'min_performance': np.NaN}, index=[0])
                entry.to_csv('output_entries_new', mode='a', index=False, header=False)
                return entry

            try:
                sample = data.groupby('target', group_keys=False).apply(
                    lambda x: x.sample(n=ceil(sample_size*len(x))))
                #sample_selected = select_samples(sample, sample_selection)
                sample_selected = sample

                X = sample_selected.drop('target', axis=1)
                y = sample_selected['target']

                num_sample = len(sample)
                num_sample_selected.append(len(sample_selected))

                if validation_type == 'ts':
                    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                        train_size=train_size,
                                                                        shuffle=True, stratify=y.values)

                    feature_selection_mask = select_features(
                        select_features_estimator(model),
                        X_train, y_train, feature_selector,
                        feature_selection_frac)
                    X_train_sel = X_train.loc[:, feature_selection_mask]
                    X_test_sel = X_test.loc[:, feature_selection_mask]

                    parameter = tune_parameters(select_parameters_estimator(model), X_train,
                                                y_train, par_grid[model], par_split_size)
                    model_object = select_validation_estimator(model).set_params(
                        **parameter)
                    model_object.fit(X_train_sel, y_train)
                    predictions = model_object.predict(X_test_sel)

                    performance.append(
                        measure_performance(performance_metric, predictions, y_test))

                elif validation_type == 'all_nested':
                    kf = StratifiedKFold(n_splits=fs_cv_split_size, shuffle=True)
                    inner_perf = []

                    for train_index, test_index in kf.split(X, y):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        feature_selection_mask = select_features(
                            select_features_estimator(model),
                            X_train, y_train, feature_selector,
                            feature_selection_frac)

                        X_train_sel = X_train.loc[:, feature_selection_mask]
                        X_test_sel = X_test.loc[:, feature_selection_mask]

                        parameter = tune_parameters(select_parameters_estimator(model),
                                                    X_train,
                                                    y_train, par_grid[model],
                                                    par_split_size)
                        model_object = select_validation_estimator(model).set_params(
                            **parameter)
                        model_object.fit(X_train_sel, y_train)
                        predictions = model_object.predict(X_test_sel)

                        inner_perf.append(
                            measure_performance(performance_metric, predictions, y_test))

                    performance.append(mean(inner_perf))

                elif validation_type == 'all_kfold':
                    feature_selection_mask = select_features(
                        select_features_estimator(model),
                        X, y, feature_selector,
                        feature_selection_frac)
                    X_sel = X.loc[:, feature_selection_mask]

                    parameter = tune_parameters(select_parameters_estimator(model), X_sel,
                                                y, par_grid[model], par_split_size)
                    model_object = select_validation_estimator(model).set_params(
                        **parameter)

                    cv = StratifiedKFold(n_splits=fs_cv_split_size,
                                                              shuffle=True)
                    cv_res = cross_validate(model_object, X_sel, y,
                                           cv=cv,
                                           scoring=performance_metric)['test_score'].mean()
                    performance.append(cv_res)

                elif validation_type == 'fs_nested_pt_kfold':
                    kf = StratifiedKFold(n_splits=fs_cv_split_size, shuffle=True)
                    inner_perf = []

                    for train_index, test_index in kf.split(X, y):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        feature_selection_mask = select_features(
                            select_features_estimator(model),
                            X_train, y_train, feature_selector,
                            feature_selection_frac)

                        X_train_sel = X_train.loc[:, feature_selection_mask]
                        X_test_sel = X_test.loc[:, feature_selection_mask]
                        X_sel = X.loc[:, feature_selection_mask]

                        parameter = tune_parameters(select_parameters_estimator(model),
                                                    X_sel,
                                                    y, par_grid[model], par_split_size)

                        model_object = select_validation_estimator(model).set_params(
                            **parameter)
                        model_object.fit(X_train_sel, y_train)
                        predictions = model_object.predict(X_test_sel)

                        inner_perf.append(
                            measure_performance(performance_metric, predictions, y_test))

                    performance.append(mean(inner_perf))

                elif validation_type == 'fs_kfold_pt_nested':
                    feature_selection_mask = select_features(
                        select_features_estimator(model),
                        X, y, feature_selector,
                        feature_selection_frac)

                    kf = StratifiedKFold(n_splits=fs_cv_split_size, shuffle=True)
                    inner_perf = []

                    for train_index, test_index in kf.split(X, y):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        X_train_sel = X_train.loc[:, feature_selection_mask]
                        X_test_sel = X_test.loc[:, feature_selection_mask]

                        parameter = tune_parameters(select_parameters_estimator(model),
                                                    X_train_sel,
                                                    y_train, par_grid[model],
                                                    par_split_size)

                        model_object = select_validation_estimator(model).set_params(
                            **parameter)
                        model_object.fit(X_train_sel, y_train)
                        predictions = model_object.predict(X_test_sel)

                        inner_perf.append(
                            measure_performance(performance_metric, predictions, y_test))

                    performance.append(mean(inner_perf))
                j = 1
            except:
                tries += 1
                j = 0


    mean_performance = mean(performance)
    median_performance = median(performance)
    max_performance = max(performance)
    min_performance = min(performance)

    entry = pd.DataFrame({'model': model,
                          'sample_size': num_sample,
                          'sample_selectors': ' '.join(sample_selection),
                          'sample_size_selected': mean(num_sample_selected),
                          'feature_selector': feature_selector,
                          'num_feature_sel': ceil(feature_selection_frac * len(data.columns)),
                          'performance_metric': performance_metric,
                          'validation_type': validation_type,
                          'train_size': train_size,
                          'fs_cv_split_size': fs_cv_split_size_list,
                          'par_split_size': par_split_size,
                          'mean_performance': "%.3f" % round(mean_performance, 2),
                          'median_performance': "%.3f" % round(median_performance, 2),
                          'max_performance': "%.3f" % round(max_performance, 2),
                          'min_performance': "%.3f" % round(min_performance, 2)}, index=[0])
    entry.to_csv('output_entries_new', mode='a', index=False, header=False)
    return entry


ray.init(ignore_reinit_error=True, num_cpus= 128) #ignore_reinit_error=True, num_cpus= 128

if __name__ == '__main__':
    data = read_data()

    data_without_nan = data.dropna()
    # customize here for your output
    # X_data_without_nan = data_without_nan.drop('target', axis=1)
    # y_data_without_nan = data_without_nan['target']

    data = replace_nan(data)
    data = data.astype({'age': int, 'sex': int, 'cp': int, 'trestbps': float, 'chol': float, 'fbs': float,
                        'restecg': float, 'thalach': float, 'exang': float, 'oldpeak': float, 'slope': float,
                        'ca': float, 'thal': float, 'location': int, 'target': int})

    male = data['sex'] = 1.0
    young = data[(data.age < 50)]
    middle = data[(data.age >= 50) & (data.age < 65)]
    elder = data[(data.age >= 65)]

    output = pd.DataFrame(columns=['model', 'sample_size', 'sample_selectors', 'sample_size_selected', 'feature_selector',
                                   'num_feature_sel', 'performance_metric', 'validation_type', 'train_size',
                                   'fs_cv_split_size', 'par_split_size', 'mean_performance', 'median_performance',
                                   'max_performance', 'min_performance'])

    result = []
    for model in MODELS:
        for sample_size in SAMPLE_SIZES:
            for sample_selection in SAMPLE_SELECTION:
                for feature_selector in FEATURE_SELECTOR:
                    for feature_selection_frac in FEATURE_SELECTION_FRAC:
                        for performance_metric in PERFORMANCE_METRICS:
                            for validation_type in VALIDATION_TYPES:
                                for par_split_size in PAR_SPLIT_SIZE:
                                    if validation_type == 'ts':
                                        train_size_list = TRAIN_SIZE
                                        fs_cv_split_size_list = np.array([np.NaN])
                                    else:
                                        train_size_list = np.array([np.NaN])
                                        fs_cv_split_size_list = FS_CV_SPLIT_SIZE
                                    for fs_cv_split_size in fs_cv_split_size_list:
                                        for train_size in train_size_list:
                                            # output = pd.concat([output, ray.get(do_calc.remote(model, sample_size, feature_selector, feature_selection_frac, performance_metric, validation_type, sample_selection, par_split_size, fs_cv_split_size, train_size, X_data_without_nan, y_data_without_nan))], ignore_index=True)
                                            #ray.get(do_calc.remote(model, sample_size, feature_selector, feature_selection_frac, performance_metric, validation_type, sample_selection, par_split_size, fs_cv_split_size, train_size, X_data_without_nan, y_data_without_nan))
                                            result.append(do_calc.remote(model, sample_size, feature_selector, feature_selection_frac,
                                                                   performance_metric, validation_type, sample_selection,
                                                                   par_split_size, fs_cv_split_size, train_size))
                                            #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                                            #    print(entry)
                                            #output = pd.concat([output, entry], ignore_index=True)

    ready, not_ready = ray.wait(result, num_returns=len(result), timeout=None)
    for f in ready:
        output = pd.concat([output, ray.get(f)], ignore_index=True)
    output.to_csv('output_new', index=False)
