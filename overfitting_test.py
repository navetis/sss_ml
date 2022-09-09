import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score, \
    balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import ray
import warnings

NAN_REPLACEMENT = 'knn'  # options: mean, median, knn
NAN_KNN_NEIGHBORS = 5  # should only be used, when knn for NaN replacement selected


ray.init(ignore_reinit_error=True, num_cpus=128)

# training & validation
ROUNDS = 50
SAMPLE_SIZES = np.array([0.04, 0.05, 0.07, 0.1, 0.5])
TRAIN_SIZE = np.array([0.6, 0.8, 0.9])
MODELS = np.array(['svm', 'logistic_regression']) # 'svm', 'logistic_regression', 'naive_bayes', 'knn', 'random_forest', 'decision_tree'
VALIDATION_TYPES = np.array(['ts', 'all_nested', 'all_kfold'])#'ts', 'all_nested', 'all_kfold', 'fs_nested_pt_kfold', 'fs_kfold_pt_nested'
CV_SPLIT_SIZE = np.array([2, 7, 13])
MAIN_METRICS = np.array(['accuracy'])  # 'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall' ; use sklearn scoring parameters; , 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score'
SHOULD_BE_BINARY = True

FEATURE_SELECTOR = np.array(['rfe']) # , 'sequential'
FEATURE_SELECTION_FRAC = np.array([0.4, 0.7, 1.0])
MAX_FEATURES = 14 # this is the maximum number of features for your dataset

# parameter ranges for models
PAR_SPLIT_SIZE = np.array([2, 7, 13])
par_grid = {'svm': {'C': np.logspace(-1, 7, num=7, base=2), 'gamma': np.logspace(1, -7, num=7, base=2)},
            'logistic_regression': {'C': np.logspace(0, 4, num=10, base=10), 'penalty': ['l1', 'l2']},
            'random_forest': {'n_estimators': [int(x) for x in np.linspace(start=5, stop=500, num=5, endpoint=True)],
                              'max_depth': [
                                  [int(x) for x in np.linspace(10, 100, num=5, endpoint=True)].extend([None, 'sqrt'])],
                              'min_samples_split': [2, 5],
                              'min_samples_leaf': [1, 2],
                              'bootstrap': [True, False]},
            'decision_tree': {'max_depth': [[int(x) for x in np.linspace(10, 100, num=10, endpoint=True)].append(None)],
                              'min_samples_split': [2, 5, 10],
                              'min_samples_leaf': [1, 2, 4]},
            'knn': {'n_neighbors': np.linspace(1, 10, num=10, endpoint=True, dtype=int)},
            'naive_bayes': {'var_smoothing': np.logspace(0, -9, num=100)}}

SCORING_METRICS = {
    'accuracy': make_scorer(accuracy_score)
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


def tune_parameters(estimator, features, target, param_grid, split_size, main_metric, feature_selection_mask):
    cross_val = StratifiedKFold(n_splits=split_size, shuffle=True)
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid,
                       cv=cross_val, scoring=SCORING_METRICS, refit=main_metric)  # define a model with parameter tuning
    clf.fit(features, target)
    results = clf.cv_results_
    return clf.best_params_, np.array([np.append(np.array([results["mean_test_%s" % scorer][np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]] for scorer in SCORING_METRICS.keys()]), np.array([[score(clf.best_estimator_, X_reference.loc[:, feature_selection_mask], y_reference) for score in SCORING_METRICS.values()]]))])



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


def measure_performances(model_object, X_to_predict, y_true, feature_selection_mask):
    return np.array([np.append(np.array([[score(model_object, X_to_predict, y_true) for score in SCORING_METRICS.values()]]),
                     np.array([[score(model_object, X_reference.loc[:, feature_selection_mask], y_reference) for score in SCORING_METRICS.values()]]))])


@ray.remote
def do_calc(model, main_metric, sample_size, feature_selector, feature_selection_frac, validation_type, par_split_size, cv_split_size, train_size):
    warnings.filterwarnings("ignore")
    performance = np.empty((ROUNDS,2 * NUM_METRICS), float)
    for i in range(ROUNDS):
        j = 0
        tries = 0
        while j < 1:
            if tries >= 1:
                entry = pd.DataFrame({'model': model,
                          'main_metric': main_metric,
                          'sample_size': sample_size,
                          'feature_selector': feature_selector,
                          'feature_selection_frac': feature_selection_frac,
                          'validation_type': validation_type,
                          'train_size': train_size,
                          'cv_split_size': cv_split_size,
                          'par_split_size': par_split_size,
                          'accuracy': np.NaN,
                          'reference_accuracy': np.NaN}, index=[0])
                entry.to_csv('output_entries', mode='a', index=False, header=False)
                return entry

            try:
                sample = data.groupby('target', group_keys=False).apply(
                    lambda x: x.sample(frac=sample_size))

                X = sample.drop('target', axis=1)
                y = sample['target']

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
                                                y_train, par_grid[model], par_split_size, main_metric, feature_selection_mask)
                    model_object = (select_validation_estimator(model).set_params(
                        **parameter)).fit(X_train_sel, y_train)

                    performance[i] = measure_performances(model_object, X_test_sel, y_test, feature_selection_mask)

                elif validation_type == 'all_nested':
                    kf = StratifiedKFold(n_splits=cv_split_size, shuffle=True)
                    inner_perf = np.empty((cv_split_size, 2 * NUM_METRICS), float)

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
                                                    par_split_size, main_metric,
                                                       feature_selection_mask)
                        model_object = (select_validation_estimator(model).set_params(
                            **parameter)).fit(X_train_sel, y_train)

                        inner_perf[c] = measure_performances(model_object, X_test_sel, y_test, feature_selection_mask)
                        c+=1

                    performance[i] = inner_perf.mean(axis=0)

                elif validation_type == 'all_kfold':
                    feature_selection_mask = select_features(
                        select_features_estimator(model),
                        X, y, feature_selector,
                        feature_selection_frac)
                    X_sel = X.loc[:, feature_selection_mask]

                    parameter, results = tune_parameters(select_parameters_estimator(model), X_sel,
                                                y, par_grid[model], par_split_size, main_metric, feature_selection_mask)

                    performance[i] = results

                elif validation_type == 'fs_nested_pt_kfold':
                    kf = StratifiedKFold(n_splits=cv_split_size, shuffle=True)

                    inner_perf = np.empty((cv_split_size, 2 * NUM_METRICS), float)
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
                                                    y, par_grid[model], par_split_size, main_metric, feature_selection_mask)

                        model_object = (select_validation_estimator(model).set_params(**parameter)).fit(X_train_sel, y_train)

                        inner_perf[c] = measure_performances(model_object, X_test_sel, y_test, feature_selection_mask)
                        c += 1

                    performance[i] = inner_perf.mean(axis=0)

                elif validation_type == 'fs_kfold_pt_nested':
                    feature_selection_mask = select_features(
                        select_features_estimator(model),
                        X, y, feature_selector,
                        feature_selection_frac)

                    kf = StratifiedKFold(n_splits=cv_split_size, shuffle=True)
                    inner_perf = np.empty((cv_split_size, 2 * NUM_METRICS), float)

                    c=0
                    for train_index, test_index in kf.split(X, y):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        X_train_sel = X_train.loc[:, feature_selection_mask]
                        X_test_sel = X_test.loc[:, feature_selection_mask]

                        parameter, _ = tune_parameters(select_parameters_estimator(model),
                                                    X_train_sel,
                                                    y_train, par_grid[model],
                                                    par_split_size, main_metric, feature_selection_mask)
                        model_object = (select_validation_estimator(model).set_params(**parameter)).fit(X_train_sel, y_train)

                        inner_perf[c] = measure_performances(model_object, X_test_sel, y_test, feature_selection_mask)
                        c += 1

                    performance[i] = inner_perf.mean(axis=0)
                j = 1
            except:
                tries += 1

    entries = pd.DataFrame({'model': model,
                           'main_metric': main_metric,
                           'sample_size': sample_size,
                           'feature_selector': feature_selector,
                           'feature_selection_frac': feature_selection_frac,
                           'validation_type': validation_type,
                           'train_size': train_size,
                           'cv_split_size': cv_split_size,
                           'par_split_size': par_split_size,
                           'accuracy': performance[:,0],
                           'reference_accuracy': performance[:,1]})
    entries.to_csv('output_entries', mode='a', index=False, header=False)
    return entries

if __name__ == '__main__':
    data = read_data()

    data = replace_nan(data)
    data = data.astype({'age': int, 'sex': int, 'cp': int, 'trestbps': float, 'chol': float, 'fbs': float,
                        'restecg': float, 'thalach': float, 'exang': float, 'oldpeak': float, 'slope': float,
                        'ca': float, 'thal': float, 'location': int, 'target': int})

    if SHOULD_BE_BINARY:
        data.replace({'target': [1,2,3,4]}, value=1, inplace=True) # making it binary

    reference = data.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=0.2))

    data.drop(reference.index, inplace=True)

    X_reference = reference.drop('target', axis=1)
    y_reference = reference['target']

    output = pd.DataFrame(columns=['model', 'main_metric', 'sample_size', 'feature_selector',
                                   'feature_selection_frac', 'validation_type', 'train_size',
                                   'cv_split_size', 'par_split_size', 'accuracy', 'reference_accuracy'])

    result = []

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
                                        result.append(do_calc.remote(model, main_metric, sample_size, feature_selector, feature_selection_frac,
                                                                     validation_type,
                                                                     par_split_size, cv_split_size, train_size))

    ready, not_ready = ray.wait(result, num_returns=len(result), timeout=None)
    for f in ready:
        output = pd.concat([output, ray.get(f)], ignore_index=True)
    output.to_csv('output', index=False)
