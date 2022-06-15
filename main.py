import os

import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from statistics import mean


# meeting
# takeaway: NaN sind kein Problem, sollte man aber analysieren -> z.B. das beste Modell ohne NaN samples ausführen


def read_data():
    # only necessary, if dataset is not labelled
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                    'ca', 'thal', 'target']

    # reads and concatenates all datasets from /data subfolder
    path = os.getcwd()
    data_files = glob.glob(os.path.join(path, "data/*.data"))

    li = []
    for filename in data_files:
        df = pd.read_csv(filename, names=column_names)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)
    return data


def process(data_to_process):
    # preprocess data (customize, if desired)
    r = data_to_process.replace('?', np.NaN).astype(float, errors='raise')
    r.replace({'chol': 0, 'trestbps': 0}, value=np.NaN, inplace=True) # chol and trestbps seem to have 0-values
    return r

# settings
NAN_REPLACEMENT = 'knn'  # options: mean, median, knn
NAN_KNN_NEIGHBORS = 5  # should only be used, when knn for NaN replacement selected

# training
ROUNDS = 10
SAMPLE_SIZES = np.linspace(0.1, 1, 50, endpoint=True)
TRAIN_SIZE = np.linspace(0.3, 1, 5, endpoint=False)

if __name__ == '__main__':

    data = process(read_data())

    # replace NaN's
    if NAN_REPLACEMENT == 'mean':
        data.fillna(data.mean(), inplace=True)
    elif NAN_REPLACEMENT == 'median':
        data.fillna(data.median(), inplace=True)
    elif NAN_REPLACEMENT == 'knn':
        imputer = KNNImputer(n_neighbors=NAN_KNN_NEIGHBORS)
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

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

    g = sns.FacetGrid(data, col='target', height=4, aspect=.5, row='sex')
    g.map(sns.histplot, "age", binwidth=5, alpha=0.5)
    g.add_legend()
    plt.show()

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

    # output = pd.DataFrame(columns=['train_size', 'sample_size', 'avg_acc'])
    #
    # for train_size in TRAIN_SIZE:
    #     for sample_size in SAMPLE_SIZES:
    #         acc = []
    #         for i in range(ROUNDS):
    #             sample = data.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=sample_size,
    #                                                                                        random_state=i))
    #             X = sample.drop('target', axis=1)
    #             y = sample['target']
    #
    #             X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
    #                                                                 random_state=42, stratify=y.values)
    #             #acc.append(SVC().fit(X_train, y_train).score(X_test, y_test))
    #             #acc.append(LogisticRegression(random_state=i, max_iter=1000, solver='liblinear').fit(X_train, y_train).score(X_test, y_test))
    #         avg_acc = mean(acc)
    #
    #         entry = pd.DataFrame({'train_size': "{:.2f}".format(train_size), 'sample_size': sample_size * 920, 'avg_acc': avg_acc}, index=[0])
    #         output = pd.concat([output, entry], ignore_index=True)
    #
    # sns.relplot(
    #     data=output, kind="line",
    #     x="sample_size", y="avg_acc",
    #     hue="train_size",
    #     facet_kws=dict(sharex=False),
    #     legend='full'
    # )
    # plt.show()
    #
    # output.to_csv('output', index=False)