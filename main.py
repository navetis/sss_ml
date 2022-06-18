import os
from math import ceil

import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from statistics import mean
from scipy import stats


# meeting
# takeaway: NaN sind kein Problem, sollte man aber analysieren -> z.B. das beste Modell ohne NaN samples ausführen


# settings

# preprocessing
NAN_REPLACEMENT = 'knn'  # options: mean, median, knn
NAN_KNN_NEIGHBORS = 5  # should only be used, when knn for NaN replacement selected

SAMPLE_SELECTION = []  # options: all, iqr, z_score
SAMPLE_SELECTION_Q1 = 0.25  # adjust here if iqr chosen
SAMPLE_SELECTION_Q3 = 0.75
SAMPLE_SELECTION_Z = 3  # adjust if z_score chosen

# training & validation
ROUNDS = 10
SAMPLE_SIZES = np.linspace(0.1, 1, 50, endpoint=True)  # percentage
TRAIN_SIZE = np.linspace(0.3, 1, 5, endpoint=False)  # percentage
VALIDATION_TYPES = ['ts', 'all_nested', 'all_kfold', 'fs_nested_pt_kfold', 'fs_kfold_pt_nested']  # pt: parameter tuning
MODELS = ['svm', 'random_forest', 'decision_tree', 'logistic_regression', 'naive_bayes', 'knn']
PERFORMANCE_TYPES = ['accuracy', 'balanced_accuracy', 'top_k_accuracy']  # check sklearn scoring parameters
FEATURE_SELECTOR = ['rfe', 'sequential'] # maybe corr later or as preprocessing?
FEATURE_SELECTION_FRAC = np.linspace(0.1, 1, 10, endpoint=True) # relevant for rfe and sequential

# parameter ranges for models
par_split_size = 10
par_grid_svc = {'C':np.logspace(0.1, 10.0, num=20, base=2),'gamma': np.logspace(0.1, 10.0, num=20, base=0.5)}
par_grid_log = {'C':np.logspace(0.1, 10.0, num=20, base=2),'penalty':['l1', 'l2']}
par_grid_rf = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)].append(None),
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}
par_grid_dt = {'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)].append(None),
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4]}
par_grid_knn = {'n_neighbors': np.linspace(1, 10, num=10, endpoint=True)}
par_grid_nb =  {'var_smoothing': np.logspace(0,-9, num=100)}

def select_feat(estimator, features, target, sel_type, frac):
    selector = RFE()
    if sel_type == 'rfe':
        selector = RFE(estimator, n_features_to_select=ceil(frac*len(features.columns)), step=1)
        selector = selector.fit(features, target)
        return features.loc[selector.support_]
    elif sel_type == 'sequential':
        selector = SequentialFeatureSelector(estimator, n_features_to_select=ceil(frac * len(features.columns)), step=1)
        selector = selector.fit(features, target)
        return features.loc[selector.support_]

def tune_param(estimator, features, target, par_grid):
    cross_val = StratifiedKFold(n_splits=par_split_size, shuffle=True)
    clf = GridSearchCV(estimator=estimator, param_grid=par_grid, cv=cross_val, iid='False')  # define a model with parameter tunning
    clf.fit(features, target)  # fit a model
    return clf.best_params_  # gives optimised C and Gamma parameters


# customize so your dataset is loaded and merged correctly
def read_data():
    # only necessary, if dataset is not labelled
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                    'ca', 'thal', 'target']

    # reads and concatenates all datasets from /data sub folder
    path = os.getcwd()
    data_files = glob.glob(os.path.join(path, "data/*.data"))

    li = []
    for filename in data_files:
        df = pd.read_csv(filename, names=column_names)
        li.append(df)

    return pre_process(pd.concat(li, axis=0, ignore_index=True))


def pre_process(data_to_process):
    # preprocess data (customize, if desired)
    r = data_to_process.replace('?', np.NaN).astype(float, errors='raise')
    r.replace({'chol': 0, 'trestbps': 0}, value=np.NaN, inplace=True) # chol and trestbps seem to have 0-values
    return r


def replace_nan(data_to_process):
    if NAN_REPLACEMENT == 'mean':
        return data_to_process.fillna(data_to_process.mean())
    elif NAN_REPLACEMENT == 'median':
        return data_to_process.fillna(data_to_process.median())
    elif NAN_REPLACEMENT == 'knn':
        imputer = KNNImputer(n_neighbors=NAN_KNN_NEIGHBORS)
        return pd.DataFrame(imputer.fit_transform(data_to_process), columns=data_to_process.columns)


def select_samples(data_to_process):
    selected_data = data_to_process
    for selection_type in SAMPLE_SELECTION:
        if selection_type == 'iqr':
            q1 = selected_data.quantile(SAMPLE_SELECTION_Q1)
            q3 = selected_data.quantile(SAMPLE_SELECTION_Q3)
            iqr = q3 - q1
            selected_data = selected_data[~((selected_data < (q1 - 1.5 * iqr)) |
                                     (selected_data > (q1 + 1.5 * iqr))).any(axis=1)]
        elif selection_type == 'z_score':
            z = np.abs(stats.zscore(selected_data))
            selected_data = selected_data[(z < 3).all(axis=1)]
    return selected_data


if __name__ == '__main__':

    data = read_data()
    data_without_nan = data.dropna
    data = replace_nan(data)
    data = select_samples(data)

    # plotting section

    # print data in console
    # print(data)
    # print(data.info())
    # print(data.describe())
    # print(data.dtypes)
    # print(data.columns)
    # print(data.head())

    # fancy plots

    # heatmap correlation
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(data.corr(method='pearson'), annot=True, fmt='.1f')
    # plt.show()

    # correlation with target
    # sns.set_context('notebook', font_scale=2.3)
    # data.drop('target', axis=1).corrwith(data.target).plot(kind='bar', grid=True, figsize=(20, 10),
    #                                                        title="Correlation with the target feature")
    # plt.tight_layout()
    # plt.show()

    # pairplot
    # sns.pairplot(data)
    # plt.show()

    # hist plot
    # sns.histplot(data=data, x='age', binwidth=5, hue=data[['sex', 'target']].apply(tuple, axis=1))
    # plt.xlabel('age')
    # plt.ylabel('amount')
    # plt.show()

    # sample size histogram for target/sex group
    # g = sns.FacetGrid(data, col='target', height=4, aspect=.5, row='sex')
    # g.map(sns.histplot, "age", binwidth=5, alpha=0.5)
    # g.add_legend()
    # plt.show()

    # jointplot
    # sns.jointplot(data=data, x="age", y="target", hue="sex", alpha=0.6)
    # plt.show()

    # dis plot
    # sns.displot(data, x='age')
    # plt.xlabel('age')
    # plt.ylabel('amount')
    #plt.show()

    # count plot
    # plt.figure(figsize=(25,5))
    # sns.countplot(data=data, x='age')
    # plt.xlabel('age')
    # plt.ylabel('amount')
    # plt.show()

    # count plot (sex)
    # plt.figure(figsize=(5,5))
    # sns.countplot(data=data, x='sex')
    # plt.xlabel('sex')
    # plt.ylabel('amount')
    # plt.show()

    # violinplot male vs female
    # plt.figure(figsize=(10, 10))
    # sns.catplot(data=data, kind='violin', x='target', y='age', hue='sex', split=True)
    # plt.show()

    output = pd.DataFrame(columns=['train_size', 'sample_size', 'mean_acc', 'median_acc', 'max_acc', 'min_acc', ])

    for train_size in TRAIN_SIZE:
        for sample_size in SAMPLE_SIZES:
            acc = []
            for i in range(ROUNDS):
                sample = data.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=sample_size,
                                                                                           random_state=i))
                X = sample.drop('target', axis=1)
                y = sample['target']

                for feature_selection_frac in FEATURE_SELECTION_FRAC:

                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                                    random_state=42, stratify=y.values)
                acc.append(SVC().fit(X_train, y_train).score(X_test, y_test))
                #acc.append(LogisticRegression(random_state=i, max_iter=1000, solver='liblinear').fit(X_train, y_train).score(X_test, y_test))
            avg_acc = mean(acc)

            entry = pd.DataFrame({'train_size': "{:.2f}".format(train_size), 'sample_size': sample_size * 920, 'avg_acc': avg_acc}, index=[0])
            output = pd.concat([output, entry], ignore_index=True)

    sns.relplot(
        data=output, kind="line",
        x="sample_size", y="avg_acc",
        hue="train_size",
        facet_kws=dict(sharex=False),
        legend='full'
    )
    plt.show()

    output.to_csv('output', index=False)