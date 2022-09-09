import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import ray
import warnings

NAN_REPLACEMENT = 'knn'  # options: mean, median, knn
NAN_KNN_NEIGHBORS = 5  # should only be used, when knn for NaN replacement selected


ray.init(ignore_reinit_error=True, num_cpus=128)

# training & validation
ROUNDS = 50
SAMPLE_SIZES = np.array([0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0])
MODELS = np.array(['svm', 'logistic_regression']) # 'svm', 'logistic_regression', 'naive_bayes', 'knn', 'random_forest', 'decision_tree'

FEATURE_SELECTOR = np.array(['rfe']) # , 'sequential'
FEATURE_SELECTION_FRAC = np.array([0.4, 0.7, 1.0])
MAX_FEATURES = 14 # this is the maximum number of features for your dataset

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


@ray.remote
def do_calc_binary(model, sample_size, feature_selector, feature_selection_frac):
    warnings.filterwarnings("ignore")
    selection = np.empty((ROUNDS,MAX_FEATURES), float)
    for i in range(ROUNDS):
        j = 0
        tries = 0
        while j < 1:
            if tries >= 1:
                entry = pd.DataFrame({'model': model,
                                      'sample_size': sample_size,
                                      'is_binary': True,
                                      'feature_selector': feature_selector,
                                      'feature_selection_frac': feature_selection_frac,
                                      'selection_age': np.NaN,
                                      'selection_sex': np.NaN,
                                      'selection_cp': np.NaN,
                                      'selection_trestbps': np.NaN,
                                      'selection_chol': np.NaN,
                                      'selection_fbs': np.NaN,
                                      'selection_restecg': np.NaN,
                                      'selection_thalach': np.NaN,
                                      'selection_exang': np.NaN,
                                      'selection_oldpeak': np.NaN,
                                      'selection_slope': np.NaN,
                                      'selection_ca': np.NaN,
                                      'selection_thal': np.NaN,
                                      'selection_location': np.NaN}, index=[0])
                entry.to_csv('output_selection_entries', mode='a', index=False, header=False)
                return entry

            try:
                sample = data_binary.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=sample_size))

                X = sample.drop('target', axis=1)
                y = sample['target']

                feature_selection_mask = select_features(
                        select_features_estimator(model),
                        X, y, feature_selector,
                        feature_selection_frac)

                selection[i] = np.array(feature_selection_mask, dtype=float)
                j = 1
            except:
                tries += 1

    mean_selection = np.mean(selection, axis=0)
    entries = pd.DataFrame({'model': model,
                            'sample_size': sample_size,
                            'is_binary': True,
                            'feature_selector': feature_selector,
                            'feature_selection_frac': feature_selection_frac,
                            'selection_age': mean_selection[0],
                            'selection_sex': mean_selection[1],
                            'selection_cp': mean_selection[2],
                            'selection_trestbps': mean_selection[3],
                            'selection_chol': mean_selection[4],
                            'selection_fbs': mean_selection[5],
                            'selection_restecg': mean_selection[6],
                            'selection_thalach': mean_selection[7],
                            'selection_exang': mean_selection[8],
                            'selection_oldpeak': mean_selection[9],
                            'selection_slope': mean_selection[10],
                            'selection_ca': mean_selection[11],
                            'selection_thal': mean_selection[12],
                            'selection_location': mean_selection[13]
                            }, index=[0])

    entries.to_csv('output_selection_entries', mode='a', index=False, header=False)
    return entries

@ray.remote
def do_calc_nonbinary(model, sample_size, feature_selector, feature_selection_frac):
    warnings.filterwarnings("ignore")
    selection = np.empty((ROUNDS,MAX_FEATURES), float)
    for i in range(ROUNDS):
        j = 0
        tries = 0
        while j < 1:
            if tries >= 1:
                entry = pd.DataFrame({'model': model,
                                      'sample_size': sample_size,
                                      'is_binary': False,
                                      'feature_selector': feature_selector,
                                      'feature_selection_frac': feature_selection_frac,
                                      'selection_age': np.NaN,
                                      'selection_sex': np.NaN,
                                      'selection_cp': np.NaN,
                                      'selection_trestbps': np.NaN,
                                      'selection_chol': np.NaN,
                                      'selection_fbs': np.NaN,
                                      'selection_restecg': np.NaN,
                                      'selection_thalach': np.NaN,
                                      'selection_exang': np.NaN,
                                      'selection_oldpeak': np.NaN,
                                      'selection_slope': np.NaN,
                                      'selection_ca': np.NaN,
                                      'selection_thal': np.NaN,
                                      'selection_location': np.NaN}, index=[0])
                entry.to_csv('output_selection_entries', mode='a', index=False, header=False)
                return entry

            try:
                sample = data.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=sample_size))

                X = sample.drop('target', axis=1)
                y = sample['target']

                feature_selection_mask = select_features(
                        select_features_estimator(model),
                        X, y, feature_selector,
                        feature_selection_frac)

                selection[i] = np.array(feature_selection_mask, dtype=float)
                j = 1
            except:
                tries += 1

    mean_selection = np.mean(selection, axis=0)
    entries = pd.DataFrame({'model': model,
                            'sample_size': sample_size,
                            'is_binary': False,
                            'feature_selector': feature_selector,
                            'feature_selection_frac': feature_selection_frac,
                            'selection_age': mean_selection[0],
                            'selection_sex': mean_selection[1],
                            'selection_cp': mean_selection[2],
                            'selection_trestbps': mean_selection[3],
                            'selection_chol': mean_selection[4],
                            'selection_fbs': mean_selection[5],
                            'selection_restecg': mean_selection[6],
                            'selection_thalach': mean_selection[7],
                            'selection_exang': mean_selection[8],
                            'selection_oldpeak': mean_selection[9],
                            'selection_slope': mean_selection[10],
                            'selection_ca': mean_selection[11],
                            'selection_thal': mean_selection[12],
                            'selection_location': mean_selection[13]
                            }, index=[0])

    entries.to_csv('output_selection_entries', mode='a', index=False, header=False)
    return entries

if __name__ == '__main__':
    data = read_data()

    data = replace_nan(data)
    data = data.astype({'age': int, 'sex': int, 'cp': int, 'trestbps': float, 'chol': float, 'fbs': float,
                        'restecg': float, 'thalach': float, 'exang': float, 'oldpeak': float, 'slope': float,
                        'ca': float, 'thal': float, 'location': int, 'target': int})

    data_binary = data.replace({'target': [1,2,3,4]}, value=1) # making it binary

    output = pd.DataFrame(columns=['model', 'sample_size', 'is_binary', 'feature_selector', 'feature_selection_frac', 'selection_age', 'selection_sex', 'selection_cp', 'selection_trestbps', 'selection_chol', 'selection_fbs',
                        'selection_restecg', 'selection_thalach', 'selection_exang', 'selection_oldpeak', 'selection_slope',
                        'selection_ca', 'selection_thal', 'selection_location'])

    result = []

    for model in MODELS:
        for sample_size in SAMPLE_SIZES:
            for feature_selector in FEATURE_SELECTOR:
                for feature_selection_frac in FEATURE_SELECTION_FRAC:
                    result.append(do_calc_binary.remote(model, sample_size, feature_selector, feature_selection_frac))
                    result.append(do_calc_nonbinary.remote(model, sample_size, feature_selector, feature_selection_frac))

    ready, not_ready = ray.wait(result, num_returns=len(result), timeout=None)
    for f in ready:
        output = pd.concat([output, ray.get(f)], ignore_index=True)
    output.to_csv('output_selection', index=False)
