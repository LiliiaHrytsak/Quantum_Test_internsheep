import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Preprocessing_data():
    def __init__(self):
        pass

    def download_csv(self, path):
        data = pd.read_csv(path)
        return data

    def split_data(self, data, train_size, name_target_column):
        X = data.drop(columns=[name_target_column]).copy()
        y = data[name_target_column]
        X_train_valid, X_test_main, y_train_valid, y_test_main = \
            train_test_split(X, y, train_size=train_size)
        return  X_train_valid, X_test_main, y_train_valid, y_test_main

    def scale_data(self, data, not_numerical_columns, X_train, X_test):
        numerical_columns = data.columns.drop(not_numerical_columns)
        scaler = StandardScaler()
        scaler.fit_transform(X_train[numerical_columns])
        scaler.transform(X_test[numerical_columns])
        return X_train, X_test, numerical_columns


