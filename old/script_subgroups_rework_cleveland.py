import os
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score, \
    balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import ray
import warnings

NAN_REPLACEMENT = 'knn'  # options: mean, median, knn
NAN_KNN_NEIGHBORS = 5  # should only be used, when knn for NaN replacement selected


ray.init(ignore_reinit_error=True, num_cpus=128) #ignore_reinit_error=True, num_cpus= 128

# training & validation
ROUNDS = 50
SAMPLE_SIZES = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]) # np.linspace(0.05, 0.5, 2, endpoint=True) # 0.03, 0.05, 0.1, 0.25, 0.5, 1.0
TRAIN_SIZE = np.array([0.6, 0.8, 0.9]) # np.linspace(0.1, 0.6, 2, endpoint=False) # 0.6, 0.8, 0.9
MODELS = np.array(['svm', 'logistic_regression']) # 'svm', 'logistic_regression', 'naive_bayes', 'knn', 'random_forest', 'decision_tree'
VALIDATION_TYPES = np.array(['ts', 'all_nested', 'all_kfold', 'fs_nested_pt_kfold', 'fs_kfold_pt_nested'])#'ts', 'all_nested', 'all_kfold', 'fs_nested_pt_kfold', 'fs_kfold_pt_nested'
CV_SPLIT_SIZE = np.array([2, 5, 7, 9, 13]) # np.linspace(2, 10, 2, endpoint=True).astype(int) #2, 3, 5, 8, 13
MAIN_METRICS = np.array(['accuracy', 'balanced_accuracy', 'f1'])  # , 'balanced_accuracy', 'f1', 'precision', 'recall' ; use sklearn scoring parameters; , 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score'
SHOULD_BE_BINARY = True

FEATURE_SELECTOR = np.array(['rfe']) # , 'sequential'
FEATURE_SELECTION_FRAC = np.array([0.4, 0.7, 1.0]) # 0.4, 0.7, 1.0
MAX_FEATURES = 13 # this is the maximum number of features for your dataset

# parameter ranges for models
PAR_SPLIT_SIZE = np.array([2, 5, 7, 9, 13]) # now:  2, 5, 7, 9, 13
par_grid = {'svm': {'C': np.logspace(-1, 7, num=7, base=2), 'gamma': np.logspace(1, -7, num=7, base=2)},
            'logistic_regression': {'C': np.logspace(0, 4, num=10, base=10), 'penalty': ['l1', 'l2']},
            'random_forest': {'n_estimators': [int(x) for x in np.linspace(start=5, stop=500, num=5, endpoint=True)],
                              'max_depth': [[int(x) for x in np.linspace(10, 100, num=5, endpoint=True)].extend([None, 'sqrt'])],
                              'min_samples_split': [2, 5],
                              'min_samples_leaf': [1, 2],
                              'bootstrap': [True, False]},
            'decision_tree': {'max_depth': [[int(x) for x in np.linspace(10, 100, num=10, endpoint=True)].append(None)],
                              'min_samples_split': [2, 5, 10], #5,
                              'min_samples_leaf': [1, 2, 4]}, #2
            'knn': {'n_neighbors': np.linspace(1, 10, num=10, endpoint=True, dtype=int)},
            'naive_bayes': {'var_smoothing': np.logspace(0, -9, num=100)}}

SCORING_METRICS = {
    'accuracy': make_scorer(accuracy_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'f1': make_scorer(f1_score, average='weighted'),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted')
}

NUM_METRICS = len(SCORING_METRICS.keys())
PERFORMANCE_METRICS_TEST = ['test_' + i for i in SCORING_METRICS.keys()]


warnings.filterwarnings("ignore")

# ------------------------

def select_features(estimator, features, target, sel_type, frac):
    if sel_type == 'rfe':
        selector = RFE(estimator, n_features_to_select=frac, step=1)
        selector = selector.fit(features, target)
        return selector.support_
    elif sel_type == 'sequential':
        selector = SequentialFeatureSelector(estimator, n_features_to_select=frac)
        selector = selector.fit(features, target)
        return selector.support_



# def get_performance_metric(metric, target, prediction):
#     if metric == 'accuracy':
#         return metrics.accuracy_score(target, prediction)
#     elif metric == 'balanced_accuracy':
#         return metrics.balanced_accuracy_score(target, prediction)
#     elif metric == 'f1':
#         return metrics.f1_score(target, prediction, average='weighted')
#     elif metric == 'precision':
#         return metrics.precision_score(target, prediction, average='weighted')
#     elif metric == 'recall':
#         return metrics.recall_score(target, prediction, average='weighted')
#     # elif metric == 'top_k_accuracy':
#     #     return metrics.top_k_accuracy_score(target, prediction)
#     # elif metric == 'average_precision':
#     #     return metrics.average_precision_score(target, prediction)
#     # elif metric == 'neg_brier_score':
#     #     return metrics.brier_score_loss(target, prediction)


def tune_parameters(estimator, features, target, param_grid, split_size, main_metric):
    cross_val = StratifiedKFold(n_splits=split_size, shuffle=True)
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid,
                       cv=cross_val, scoring=SCORING_METRICS, refit=main_metric)  # define a model with parameter tuning
    clf.fit(features, target)
    results = clf.cv_results_
    return clf.best_params_, np.array([[results["mean_test_%s" % scorer][np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]] for scorer in SCORING_METRICS.keys()]])


def read_data():
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                    'ca', 'thal', 'target']
    data_path = os.path.join(os.getcwd(), "../data/")

    cleveland_data = pd.read_csv(os.path.join(data_path, "processed.cleveland.data"), names=column_names, na_values='?')
    return pre_process(cleveland_data)


def pre_process(data_to_process):
    r = data_to_process.astype(float, errors='raise')
    r.replace({'chol': 0, 'trestbps': 0}, value=np.NaN, inplace=True)  # chol and trestbps seem to have 0-values
    r['thal'].replace({3.0: 0.0, 6.0: 1.0, 7.0: 2.0}, inplace=True)
    r = r[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
           'ca', 'thal', 'target']]
    return r


def replace_nan(data_to_process):
    if NAN_REPLACEMENT == 'mean':
        return data_to_process.fillna(data_to_process.mean())
    elif NAN_REPLACEMENT == 'median':
        return data_to_process.fillna(data_to_process.median())
    elif NAN_REPLACEMENT == 'knn':
        imputer = KNNImputer(n_neighbors=NAN_KNN_NEIGHBORS)
        return pd.DataFrame(imputer.fit_transform(data_to_process), columns=data_to_process.columns)


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
    elif name == 'svm':
        return SVC(kernel='rbf')
    else:
        return select_features_estimator(name)

def select_validation_estimator(name):
    return select_parameters_estimator(name)


def measure_performances(model_object, X_to_predict, y_true):
    return np.array([[score(model_object, X_to_predict, y_true) for score in SCORING_METRICS.values()]])


@ray.remote
def do_calc(subgroup, model, main_metric, sample_size, feature_selector, feature_selection_frac, validation_type, par_split_size, cv_split_size, train_size):
    warnings.filterwarnings("ignore")
    performance = np.empty((ROUNDS,NUM_METRICS), float)
    subgroup_data = subgroups[subgroup]
    for i in range(ROUNDS):
        j = 0
        tries = 0
        while j < 1:
            if tries >= 1:
                entry = pd.DataFrame({'subgroup': subgroup,
                          'model': model,
                          'main_metric': main_metric,
                          'sample_size': sample_size,
                          'feature_selector': feature_selector,
                          'feature_selection_frac': feature_selection_frac,
                          'validation_type': validation_type,
                          'train_size': train_size,
                          'cv_split_size': cv_split_size,
                          'par_split_size': par_split_size,
                          'accuracy': np.NaN,
                          'balanced_accuracy': np.NaN,
                          'f1': np.NaN,
                          'precision': np.NaN,
                          'recall': np.NaN}, index=[0])
                entry.to_csv('output_entries', mode='a', index=False, header=False)
                return entry

            try:
                sample = subgroup_data.groupby('target', group_keys=False).apply(
                    lambda x: x.sample(frac=sample_size))
                X = sample.drop('target', axis=1)
                y = sample['target']

                # print(model, sample_size, feature_selector, feature_selection_frac, validation_type, par_split_size, fs_cv_split_size, train_size, i, j, tries)
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

                    parameter, _ = tune_parameters(select_parameters_estimator(model), X_train_sel,
                                                y_train, par_grid[model], par_split_size, main_metric)
                    model_object = (select_validation_estimator(model).set_params(
                        **parameter)).fit(X_train_sel, y_train)
                    # predictions = model_object.predict(X_test_sel)

                    performance[i] = measure_performances(model_object, X_test_sel, y_test)
                    #performance[i] = measure_performance(y_test, predictions)
                    #np.put(performance, i, measure_performance(y_test, predictions))
                    #performance.append(measure_performance(performance_metric, predictions, y_test))

                elif validation_type == 'all_nested':
                    kf = StratifiedKFold(n_splits=cv_split_size, shuffle=True)
                    inner_perf = np.empty((cv_split_size, NUM_METRICS), float)

                    c = 0
                    for train_index, test_index in kf.split(X, y):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        feature_selection_mask = select_features(
                            select_features_estimator(model),
                            X_train, y_train, feature_selector,
                            feature_selection_frac)

                        X_train_sel = X_train.loc[:, feature_selection_mask]
                        X_test_sel = X_test.loc[:, feature_selection_mask]

                        parameter, _ = tune_parameters(select_parameters_estimator(model),
                                                    X_train_sel,
                                                    y_train, par_grid[model],
                                                    par_split_size, main_metric)
                        model_object = (select_validation_estimator(model).set_params(
                            **parameter)).fit(X_train_sel, y_train)
                        # predictions = model_object.predict(X_test_sel)

                        inner_perf[c] = measure_performances(model_object, X_test_sel, y_test)
                            #measure_performance(y_test, predictions)
                        c+=1
                        #inner_perf = np.append(inner_perf, measure_performance(performance_metric, predictions, y_test), axis=0)
                        #inner_perf.append(measure_performance(performance_metric, predictions, y_test))

                    performance[i] = inner_perf.mean(axis=0)
                    # np.put(performance, i, inner_perf.mean(axis=0))
                    #performance = np.append(performance, inner_perf.mean(axis=0), axis=0)
                    #performance.append(mean(inner_perf))

                elif validation_type == 'all_kfold':
                    feature_selection_mask = select_features(
                        select_features_estimator(model),
                        X, y, feature_selector,
                        feature_selection_frac)
                    X_sel = X.loc[:, feature_selection_mask]

                    parameter, results = tune_parameters(select_parameters_estimator(model), X_sel,
                                                y, par_grid[model], par_split_size, main_metric)

                    # so sinvoller?
                    #model_object = select_validation_estimator(model).set_params(**parameter)
                    #cv = StratifiedKFold(n_splits=fs_cv_split_size, shuffle=True)
                    #cv_res = cross_validate(model_object, X_sel, y,cv=cv,scoring=SCORING_METRICS)
                    #performance[i] = np.array([[np.mean(cv_res.get(x)) for x in PERFORMANCE_METRICS_TESTS]])
                    performance[i] = results
                    #performance.append(cv_res)

                elif validation_type == 'fs_nested_pt_kfold':
                    kf = StratifiedKFold(n_splits=cv_split_size, shuffle=True)

                    inner_perf = np.empty((cv_split_size, NUM_METRICS), float)
                    c = 0
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

                        parameter, _ = tune_parameters(select_parameters_estimator(model),
                                                    X_sel,
                                                    y, par_grid[model], par_split_size, main_metric)

                        model_object = (select_validation_estimator(model).set_params(**parameter)).fit(X_train_sel, y_train)
                        # predictions = model_object.predict(X_test_sel)

                        inner_perf[c] = measure_performances(model_object, X_test_sel, y_test)
                        #measure_performance(y_test, predictions)
                        #np.put(inner_perf, c, measure_performance(y_test, predictions))
                        c += 1
                        #inner_perf = np.append(inner_perf, measure_performance(performance_metric, predictions, y_test),axis=0)
                        # inner_perf.append(measure_performance(performance_metric, predictions, y_test))

                    performance[i] = inner_perf.mean(axis=0)
                    #np.put(performance, i, inner_perf.mean(axis=0))
                    # performance = np.append(performance, inner_perf.mean(axis=0), axis=0)
                    #performance.append(mean(inner_perf))

                elif validation_type == 'fs_kfold_pt_nested':
                    feature_selection_mask = select_features(
                        select_features_estimator(model),
                        X, y, feature_selector,
                        feature_selection_frac)

                    kf = StratifiedKFold(n_splits=cv_split_size, shuffle=True)
                    inner_perf = np.empty((cv_split_size, NUM_METRICS), float)

                    c=0
                    for train_index, test_index in kf.split(X, y):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        X_train_sel = X_train.loc[:, feature_selection_mask]
                        X_test_sel = X_test.loc[:, feature_selection_mask]

                        parameter, _ = tune_parameters(select_parameters_estimator(model),
                                                    X_train_sel,
                                                    y_train, par_grid[model],
                                                    par_split_size, main_metric)
                        model_object = (select_validation_estimator(model).set_params(**parameter)).fit(X_train_sel, y_train)
                        #predictions = model_object.predict(X_test_sel)

                        inner_perf[c] = measure_performances(model_object, X_test_sel, y_test)
                            #measure_performance(y_test, predictions)
                        #np.put(inner_perf, c, measure_performance(y_test, predictions))
                        c += 1
                        #inner_perf.append(measure_performance(performance_metric, predictions, y_test))

                    performance[i] = inner_perf.mean(axis=0)
                    # np.put(performance, i, inner_perf.mean(axis=0))
                    #performance = np.append(performance, inner_perf.mean(axis=0), axis=0)
                j = 1
            except:
                tries += 1


    entry = pd.DataFrame({'subgroup': subgroup,
                           'model': model,
                           'main_metric': main_metric,
                           'sample_size': sample_size,
                           'feature_selector': feature_selector,
                           'feature_selection_frac': feature_selection_frac,
                           'validation_type': validation_type,
                           'train_size': train_size,
                           'cv_split_size': cv_split_size,
                           'par_split_size': par_split_size,
                           'accuracy': performance[:,0],
                           'balanced_accuracy': performance[:,1],
                           'f1': performance[:,2],
                           'precision': performance[:,3],
                           'recall': performance[:,4]})

    entry.to_csv('output_entries', mode='a', index=False, header=False)
    return entry

if __name__ == '__main__':
    data = read_data()


    # data_without_nan = data.dropna()
    # customize here for your output
    # X_data_without_nan = data_without_nan.drop('target', axis=1)
    # y_data_without_nan = data_without_nan['target']

    data = replace_nan(data)
    data = data.astype({'age': int, 'sex': int, 'cp': int, 'trestbps': float, 'chol': float, 'fbs': float,
                        'restecg': float, 'thalach': float, 'exang': float, 'oldpeak': float, 'slope': float,
                        'ca': float, 'thal': float, 'target': int})

    if SHOULD_BE_BINARY:
        data.replace({'target': [1,2,3,4]}, value=1, inplace=True) # making it binary

    male = data[data['sex'] == 1.0]
    female = data[data['sex'] == 0.0]
    young = data[(data.age < 50)]
    middle = data[(data.age >= 50) & (data.age < 65)]
    elder = data[(data.age >= 65)]

    subgroups = {'young': young,
    'middle': middle,
    'elder': elder}

    output = pd.DataFrame(columns=['subgroup', 'model', 'main_metric', 'sample_size', 'feature_selector',
                                   'feature_selection_frac', 'validation_type', 'train_size',
                                   'cv_split_size', 'par_split_size', 'accuracy', 'balanced_accuracy',
                                   'f1', 'precision', 'recall'])

    result = []

    for subgroup in subgroups.keys():
        for model in MODELS:
            for main_metric in MAIN_METRICS:
                for feature_selector in FEATURE_SELECTOR:
                    for sample_size in SAMPLE_SIZES:
                        for feature_selection_frac in FEATURE_SELECTION_FRAC:
                            for validation_type in VALIDATION_TYPES:
                                for par_split_size in PAR_SPLIT_SIZE:
                                    if validation_type == 'ts':
                                        train_size_list = TRAIN_SIZE
                                        cv_split_size_list = np.array([np.NaN])
                                    elif validation_type == 'all_kfold':
                                        train_size_list = np.array([np.NaN])
                                        cv_split_size_list = np.array([np.NaN])
                                    else:
                                        train_size_list = np.array([np.NaN])
                                        cv_split_size_list = CV_SPLIT_SIZE

                                    for cv_split_size in cv_split_size_list:
                                        for train_size in train_size_list:
                                            result.append(
                                                do_calc.remote(subgroup, model, main_metric, sample_size, feature_selector,
                                                               feature_selection_frac,
                                                               validation_type,
                                                               par_split_size, cv_split_size, train_size))
                                            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                                            #    print(entry)

    ready, not_ready = ray.wait(result, num_returns=len(result), timeout=None)
    for f in ready:
        output = pd.concat([output, ray.get(f)], ignore_index=True)
    output.to_csv('output', index=False)
