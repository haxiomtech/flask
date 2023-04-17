from sklearn.model_selection import train_test_split


def dsplit(df, var_pred):
    # Split the data into X (features) and y (target variable)
    X = df.drop(var_pred, axis=1)
    y = df[var_pred]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Return the training and testing data as a tuple
    return X_train, X_test, y_train, y_test


def dsplitb(df, var_pred):
    # Split the data into X (features) and y (target variable)
    X = df.drop(var_pred, axis=1)
    y = df[var_pred]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Return the training and testing data as a tuple
    return X_train, X_test, y_train, y_test


def lastcolsplit(df):
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=.3, random_state=42)
    return X_train, X_test, y_train, y_test
