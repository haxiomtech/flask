import numpy as np

from sklearn.impute import SimpleImputer


def simean(X):
    # Create an instance of the SimpleImputer class
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Use the fit_transform method to fill in the missing values with the mean
    X_imputed = imputer.fit_transform(X)

    return X_imputed


def simf(X):
    # Create an instance of the SimpleImputer class
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    # Use the fit_transform method to fill in the missing values with the most frequent value
    X_imputed = imputer.fit_transform(X)

    return X_imputed
