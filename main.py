import os

import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.impute import KNNImputer


def read_data():
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                    'ca', 'thal', 'target']

    path = os.getcwd()
    data_files = glob.glob(os.path.join(path, "data/*.data"))

    li = []
    for filename in data_files:
        df = pd.read_csv(filename, names=column_names)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True).replace('?', np.NaN).astype(float, errors='raise')


# settings
NAN_REPLACEMENT = 'knn'  # options: mean, median, knn
NAN_KNN_NEIGHBORS = 5  # should only be used, when knn for NaN replacement selected


if __name__ == '__main__':

    data = read_data()

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

    # heatmap
    #plt.figure(figsize=(10, 10))
    #sns.heatmap(data.corr(), annot=True, fmt='.1f')
    #plt.show()

    # pairplot
    #sns.pairplot(data)
    #plt.show()

    # hist plot
    # sns.histplot(data=data, x='age', binwidth=5)
    # plt.xlabel('age')
    # plt.ylabel('amount')
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

    plt.figure(figsize=(10, 10))
    sns.catplot(data=data, kind='violin', x='target', y='age', hue='sex', split=True)
    plt.show()