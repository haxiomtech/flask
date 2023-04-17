# Importações para balanceamento de classes
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks


def smote(X, y):
    # Create an instance of the SMOTE class with a sampling strategy of "minority"
    smote = SMOTE(sampling_strategy='minority')

    # Apply SMOTE to a feature matrix X and target vector y
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled


def adasyn(X, y):
    # Create an instance of the ADASYN class with a sampling strategy of "minority"
    adasyn = ADASYN(sampling_strategy='minority')

    # Apply ADASYN to a feature matrix X and target vector y
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    return X_resampled, y_resampled


def ros(X, y):
    # Create an instance of the RandomOverSampler class with a sampling strategy of "minority"
    ros = RandomOverSampler(sampling_strategy='minority')

    # Apply RandomOverSampler to a feature matrix X and target vector y
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled


def rus(X, y):
    # Create an instance of the RandomUnderSampler class with a sampling strategy of "majority"
    rus = RandomUnderSampler(sampling_strategy='majority')

    # Apply RandomUnderSampler to a feature matrix X and target vector y
    X_resampled, y_resampled = rus.fit_resample(X, y)

    return X_resampled, y_resampled


def cc(X, y):
    # Create an instance of the ClusterCentroids class with a sampling strategy of "majority"
    cc = ClusterCentroids(sampling_strategy='majority')

    # Apply ClusterCentroids to a feature matrix X and target vector y
    X_resampled, y_resampled = cc.fit_resample(X, y)

    return X_resampled, y_resampled


def tmkl(X, y):
    tl = TomekLinks()
    X_resampled, y_resampled = tl.fit_resample(X, y)

    return X_resampled, y_resampled