from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize


def nminmax(df):

    # criar scaler
    scaler = MinMaxScaler()

    # ajustar scaler aos dados
    scaler.fit(df)

    # transformar dados com o scaler
    x_norm = scaler.transform(df)

    return x_norm


def nzscore(df):

    # criar scaler
    scaler = StandardScaler()

    # ajustar scaler aos dados
    scaler.fit(df)

    # transformar dados com o scaler
    x_norm = scaler.transform(df)

    return x_norm


def nnorm(df):

    # normalizar dados
    x_norm = normalize(df, axis=0)

    return x_norm
