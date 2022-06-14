import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sl

if __name__ == '__main__':
    data = pd.read_csv('heart.csv', sep=',')
    print(data.describe())
