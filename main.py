import os
from math import ceil
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, average_precision_score, \
    brier_score_loss
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from statistics import mean, median
from scipy import stats



# meeting
# takeaway: NaN sind kein Problem, sollte man aber analysieren -> z.B. das beste Modell ohne NaN samples ausführen


# settings

# preprocessing
from sklearn.tree import DecisionTreeClassifier

NAN_REPLACEMENT = 'knn'  # options: mean, median, knn
NAN_KNN_NEIGHBORS = 5  # should only be used, when knn for NaN replacement selected

SAMPLE_SELECTION = []  # options: iqr, z_score
SAMPLE_SELECTION_Q1 = 0.25  # adjust here if iqr chosen
SAMPLE_SELECTION_Q3 = 0.75
SAMPLE_SELECTION_Z = 3  # adjust if z_score chosen

# maybe correlation preprocessing??


# training & validation
ROUNDS = 50
SAMPLE_SIZES = np.linspace(0.05, 1, 50, endpoint=True)  # percentage
TRAIN_SIZE = np.linspace(0.5, 1, 5, endpoint=False)  # percentage
VALIDATION_TYPES = ['ts', 'all_nested', 'all_kfold', 'fs_nested_pt_kfold', 'fs_kfold_pt_nested']  # pt: parameter tuning
FOLD_SPLIT_SIZE = 5
MODELS = ['svm', 'knn', 'naive_bayes',  'random_forest', 'decision_tree', 'logistic_regression' ]
PERFORMANCE_METRICS = ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score']  # check sklearn scoring parameters
FEATURE_SELECTOR = ['rfe', 'sequential'] # maybe corr later or as preprocessing?
FEATURE_SELECTION_FRAC = np.linspace(0.1, 1, 10, endpoint=True) # relevant for rfe and sequential

# parameter ranges for models
PAR_SPLIT_SIZE = 5
par_grid = {'svm': {'C': np.logspace(0.1, 10.0, num=20, base=2),'gamma': np.logspace(0.1, 10.0, num=20, base=0.5)},
            'logistic_regression': {'C': np.logspace(0.1, 10.0, num=20, base=2),'penalty':['l1', 'l2']},
            'random_forest': {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                              'max_depth': [[int(x) for x in np.linspace(10, 100, num=10, endpoint=True)].extend([None, 'sqrt'])],
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
        selector = RFE(estimator, n_features_to_select=ceil(frac*len(features.columns)), step=1)
        selector = selector.fit(features, target)
        return selector.support_
    elif sel_type == 'sequential':
        selector = SequentialFeatureSelector(estimator, n_features_to_select=ceil(frac * len(features.columns)))
        selector = selector.fit(features, target)
        return selector.support_


def tune_parameters(estimator, features, target, par_grid):
    cross_val = StratifiedKFold(n_splits=PAR_SPLIT_SIZE, shuffle=True)
    clf = GridSearchCV(estimator=estimator, param_grid=par_grid, cv=cross_val)  # define a model with parameter tuning
    clf.fit(features, target)
    return clf.best_params_


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


if __name__ == '__main__':

    data = read_data()

    data_without_nan = data.dropna()
    # customize here for your output
    X_data_without_nan = data_without_nan.drop('target', axis=1)
    y_data_without_nan = data_without_nan['target']

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

    output = pd.DataFrame(columns=['model', 'train_size', 'sample_size', 'performance_metric', 'num_feature_sel',
                                   'feature_selector', 'mean_performance', 'median_performance',
                                   'max_performance', 'min_performance', 'validation_type', 'mean_nan_performance',
                                   'median_nan_performance', 'max_nan_performance', 'min_nan_performance'])
    for model in MODELS:
        for train_size in TRAIN_SIZE:
            for sample_size in SAMPLE_SIZES:
                sample = data.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=sample_size))
                X = sample.drop('target', axis=1)
                y = sample['target']
                num_sample_size = int(sample_size * len(data))

                for feature_selector in FEATURE_SELECTOR:
                    for feature_selection_frac in FEATURE_SELECTION_FRAC:
                        for performance_metric in PERFORMANCE_METRICS:
                            for validation_type in VALIDATION_TYPES:
                                performance = []
                                nan_performance = []
                                for i in range(ROUNDS):
                                    if validation_type == 'ts':
                                        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                                                            shuffle=True) #stratify=y.values

                                        feature_selection_mask = select_features(select_features_estimator(model),
                                                                                 X_train, y_train, feature_selector,
                                                                                 feature_selection_frac)
                                        X_train_sel = X_train.loc[:, feature_selection_mask]
                                        X_test_sel = X_test.loc[:, feature_selection_mask]

                                        parameter = tune_parameters(select_parameters_estimator(model), X_train,
                                                                    y_train, par_grid[model])
                                        model_object = select_validation_estimator(model).set_params(**parameter)
                                        model_object.fit(X_train_sel, y_train)
                                        predictions = model_object.predict(X_test_sel)

                                        performance.append(measure_performance(performance_metric, predictions, y_test))
                                        nan_performance.append(measure_performance(performance_metric,
                                                model_object.predict(X_data_without_nan.loc[:, feature_selection_mask]),
                                                y_data_without_nan))
                                    elif validation_type == 'all_nested':
                                        kf = StratifiedKFold(n_splits=FOLD_SPLIT_SIZE, shuffle=True)
                                        inner_perf = []
                                        inner_nan_perf = []
                                        for train_index, test_index in kf.split(X, y):
                                            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                                            feature_selection_mask = select_features(select_features_estimator(model),
                                                                                     X_train, y_train, feature_selector,
                                                                                     feature_selection_frac)
                                            X_train_sel = X_train.loc[:, feature_selection_mask]
                                            X_test_sel = X_test.loc[:, feature_selection_mask]

                                            parameter = tune_parameters(select_parameters_estimator(model), X_train,
                                                                        y_train, par_grid[model])
                                            model_object = select_validation_estimator(model).set_params(**parameter)
                                            model_object.fit(X_train_sel, y_train)
                                            predictions = model_object.predict(X_test_sel)

                                            inner_perf.append(
                                                measure_performance(performance_metric, predictions, y_test))
                                            inner_nan_perf.append(measure_performance(performance_metric,
                                                                                       model_object.predict(
                                                                                           X_data_without_nan.loc[:,
                                                                                           feature_selection_mask]),
                                                                                       y_data_without_nan))
                                        performance.append(mean(inner_perf))
                                        nan_performance.append(mean(inner_nan_perf))
                                    elif validation_type == 'all_kfold':
                                        feature_selection_mask = select_features(select_features_estimator(model),
                                                                                 X, y, feature_selector,
                                                                                 feature_selection_frac)
                                        X_sel = X.loc[:, feature_selection_mask]

                                        parameter = tune_parameters(select_parameters_estimator(model), X,
                                                                    y, par_grid[model])
                                        model_object = select_validation_estimator(model).set_params(**parameter)

                                        performance.append(cross_validate(model_object, X_sel, y,
                                                               StratifiedKFold(n_splits=FOLD_SPLIT_SIZE, shuffle=True),
                                                               scoring=performance_metric)['test_score'].mean())
                                        nan_performance.append(cross_validate(model_object, X_data_without_nan, y_data_without_nan,
                                                               StratifiedKFold(n_splits=FOLD_SPLIT_SIZE, shuffle=True),
                                                               scoring=performance_metric)['test_score'].mean())
                                    elif validation_type == 'fs_nested_pt_kfold':
                                        kf = StratifiedKFold(n_splits=FOLD_SPLIT_SIZE, shuffle=True)
                                        inner_perf = []
                                        inner_nan_perf = []
                                        for train_index, test_index in kf.split(X, y):
                                            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                                            feature_selection_mask = select_features(select_features_estimator(model),
                                                                                     X_train, y_train, feature_selector,
                                                                                     feature_selection_frac)
                                            X_train_sel = X_train.loc[:, feature_selection_mask]
                                            X_test_sel = X_test.loc[:, feature_selection_mask]
                                            X_sel = X.loc[:, feature_selection_mask]

                                            parameter = tune_parameters(select_parameters_estimator(model), X_sel,
                                                                        y, par_grid[model])
                                            model_object = select_validation_estimator(model).set_params(**parameter)
                                            model_object.fit(X_train_sel, y_train)
                                            predictions = model_object.predict(X_test_sel)

                                            inner_perf.append(
                                                measure_performance(performance_metric, predictions, y_test))
                                            inner_nan_perf.append(measure_performance(performance_metric,
                                                                                      model_object.predict(
                                                                                          X_data_without_nan.loc[:,
                                                                                          feature_selection_mask]),
                                                                                      y_data_without_nan))
                                        performance.append(mean(inner_perf))
                                        nan_performance.append(mean(inner_nan_perf))
                                    elif validation_type == 'fs_kfold_pt_nested':
                                        feature_selection_mask = select_features(select_features_estimator(model),
                                                                                 X, y, feature_selector,
                                                                                 feature_selection_frac)
                                        X_sel = X.loc[:, feature_selection_mask]

                                        inner_perf = []
                                        inner_nan_perf = []
                                        for train_index, test_index in kf.split(X, y):
                                            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                                            X_train_sel = X_train.loc[:, feature_selection_mask]
                                            X_test_sel = X_test.loc[:, feature_selection_mask]

                                            parameter = tune_parameters(select_parameters_estimator(model), X_train_sel,
                                                                        y_train, par_grid[model])
                                            model_object = select_validation_estimator(model).set_params(**parameter)
                                            model_object.fit(X_train_sel, y_train)
                                            predictions = model_object.predict(X_test_sel)

                                            inner_perf.append(
                                                measure_performance(performance_metric, predictions, y_test))
                                            inner_nan_perf.append(measure_performance(performance_metric,
                                                                                      model_object.predict(
                                                                                          X_data_without_nan.loc[:,
                                                                                          feature_selection_mask]),
                                                                                      y_data_without_nan))
                                        performance.append(mean(inner_perf))
                                        nan_performance.append(mean(inner_nan_perf))


                                print(performance)
                                mean_performance = mean(performance)
                                median_performance = median(performance)
                                max_performance = max(performance)
                                min_performance = min(performance)

                                mean_nan_performance = mean(nan_performance)
                                median_nan_performance = median(nan_performance)
                                max_nan_performance = max(nan_performance)
                                min_nan_performance = min(nan_performance)

                                entry = pd.DataFrame({'model': model,
                                                      'train_size': int(train_size * num_sample_size),
                                                      'sample_size': num_sample_size,
                                                      'performance_metric': performance_metric,
                                                      'num_feature_sel': int(feature_selection_frac * len(X.columns)),
                                                      'feature_selector': feature_selector,
                                                      'mean_performance': mean_performance,
                                                      'median_performance': median_performance,
                                                      'max_performance': max_performance,
                                                      'min_performance': min_performance,
                                                      'validation_type': validation_type,
                                                      'mean_nan_performance': mean_nan_performance,
                                                      'median_nan_performance': median_nan_performance,
                                                      'max_nan_performance': max_nan_performance,
                                                      'min_nan_performance': min_nan_performance}, index=[0])
                                print(entry)
                                output = pd.concat([output, entry], ignore_index=True)



    # sns.relplot(
    #     data=output, kind="line",
    #     x="sample_size", y="avg_acc",
    #     hue="train_size",
    #     facet_kws=dict(sharex=False),
    #     legend='full'
    # )
    # plt.show()

    output.to_csv('output', index=False)