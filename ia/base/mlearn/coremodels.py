import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_squared_error, mean_absolute_error, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle

warnings.filterwarnings('ignore')


class Model:
    def __init__(self, models, x_train, y_train, x_test, y_test, problem_type):
        self.models = models
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.problem_type = problem_type
        self.scores = []
        self.predicted = []
        self.metrics_df = None

    def fit(self):
        for name, model in self.models.items():
            if self.problem_type == 'classification':
                model.fit(self.x_train, self.y_train)
            elif self.problem_type == 'regression':
                model.fit(self.x_train, self.y_train)
            elif self.problem_type == 'dimensionality_reduction':
                model.fit(self.x_train)
            elif self.problem_type == 'clustering':
                model.fit(self.x_train)
            else:
                raise ValueError(f'Invalid problem_type: {self.problem_type}')

            y_pred = model.predict(self.x_test)
            self.predicted.append(y_pred)

            if self.problem_type == 'classification':
                score = int(accuracy_score(y_pred, self.y_test) * 100)
            elif self.problem_type == 'regression':
                score = r2_score(self.y_test, y_pred)
            elif self.problem_type == 'dimensionality_reduction':
                score = model.explained_variance_ratio_.sum()
            elif self.problem_type == 'clustering':
                score = silhouette_score(self.x_train, model.labels_)
            else:
                raise ValueError(f'Invalid problem_type: {self.problem_type}')

            self.scores.append(score)

    def evaluate(self):
        for score, name in zip(self.scores, self.models.keys()):
            print(f'{name} score: {score}')
        plt.figure(figsize=(25, 8))
        ax = sns.barplot(x=list(self.models.keys()), y=self.scores)
        for i in ax.patches:
            width, height = i.get_width(), i.get_height()
            x, y = i.get_xy()
            ax.annotate(f'{round(height, 2)}', (x + width / 2, y + height * 1.02), ha='center')
        best_model_name = list(self.models.keys())[self.scores.index(max(self.scores))]
        print(f'{best_model_name}, {max(self.scores)}')

    def save_metrics(self, path):
        if self.problem_type == 'classification':
            metrics = []
            for name, predicted in zip(self.models.keys(), self.predicted):
                report = classification_report(predicted, self.y_test, output_dict=True)
                accuracy = round(report['accuracy'], 2)
                precision = round(report['weighted avg']['precision'], 2)
                recall = round(report['weighted avg']['recall'], 2)
                f1_score = round(report['weighted avg']['f1-score'], 2)
                metrics.append({'Model': name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
                                'F1 Score': f1_score})
            self.metrics_df = pd.DataFrame(metrics)
            self.metrics_df.to_csv(path, index=False)
        elif self.problem_type == 'regression':
            metrics = []
            for name, predicted in zip(self.models.keys(), self.predicted):
                r2 = round(r2_score(predicted, self.y_test), 2)
                mse = round(mean_squared_error(predicted, self.y_test), 2)
                mae = round(mean_absolute_error(predicted, self.y_test), 2)
                metrics.append({'Model': name, 'R2': r2, 'MSE': mse, 'MAE': mae})
            self.metrics_df = pd.DataFrame(metrics)
            self.metrics_df.to_csv(path, index=False)
        elif self.problem_type == 'dimensionality_reduction':
            metrics = []
            for name, model in self.models.items():
                variance = round(model.explained_variance_ratio_.sum(), 2)
                metrics.append({'Model': name, 'Explained Variance': variance})
            self.metrics_df = pd.DataFrame(metrics)
            self.metrics_df.to_csv(path, index=False)
        elif self.problem_type == 'clustering':
            metrics = []
            for name, model in self.models.items():
                silhouette = round(silhouette_score(self.x_train, model.labels_), 2)
                metrics.append({'Model': name, 'Silhouette Score': silhouette})
            self.metrics_df = pd.DataFrame(metrics)
            self.metrics_df.to_csv(path, index=False)
        else:
            raise ValueError(f'Invalid problem_type: {self.problem_type}')


"""

=========
Regressão
=========

models= {
    "Logistic": LogisticRegression(),
    "rfClassifier": RandomForestClassifier(),
    "tree": DecisionTreeClassifier(max_depth=5, criterion="gini"),
    "knClassifier": KNeighborsClassifier(n_neighbors=5),
    "gBoost": GradientBoostingClassifier(),
    "Ada Boost": AdaBoostClassifier(n_estimators=150),
    "Bagging": BaggingClassifier(n_estimators=150),
    "xgBoost": XGBClassifier(),
    "catBoost": CatBoostClassifier(logging_level="Silent"),
    "lightGBM": LGBMClassifier(),
    "svm": SVC(),
}


=============
Classificação
=============

models = {
    "Logistic": LogisticRegression(),
    "rfClassifier": RandomForestClassifier(),
    "tree": DecisionTreeClassifier(max_depth=5, criterion="gini"),
    "knClassifier": KNeighborsClassifier(n_neighbors=5),
    "gBoost": GradientBoostingClassifier(),
    "Ada Boost": AdaBoostClassifier(n_estimators=150),
    "Bagging": BaggingClassifier(n_estimators=150),
    "xgBoost": XGBClassifier(),
    "catBoost": CatBoostClassifier(logging_level="Silent"),
    "lightGBM": LGBMClassifier(),
    "svm": SVC(),
}

=============
Clusterização
=============

models = {
    "Logistic": LogisticRegression(),
    "rfClassifier": RandomForestClassifier(),
    "tree": DecisionTreeClassifier(max_depth=5, criterion="gini"),
    "knClassifier": KNeighborsClassifier(n_neighbors=5),
    "gBoost": GradientBoostingClassifier(),
    "Ada Boost": AdaBoostClassifier(n_estimators=150),
    "Bagging": BaggingClassifier(n_estimators=150),
    "xgBoost": XGBClassifier(),
    "catBoost": CatBoostClassifier(logging_level="Silent"),
    "lightGBM": LGBMClassifier(),
    "svm": SVC(),
}

"""




"""

Esse código é uma classe chamada Model, que é usada para comparar o desempenho de diferentes modelos de aprendizado de máquina em relação a um determinado conjunto de dados.

Você pode usá-lo da seguinte maneira:

Instancie um objeto Model, passando como argumentos:
models: um dicionário que contém os modelos que você deseja comparar, onde as chaves são os nomes dos modelos e os valores são as instâncias dos modelos. Por exemplo: {'Modelo A': modelo_a, 'Modelo B': modelo_b}.
x_train, y_train, x_test e y_test: os dados de treinamento e teste que você deseja usar para avaliar os modelos.
problem_type: o tipo de problema que você está resolvendo, que pode ser 'classification' (classificação), 'regression' (regressão), 'dimensionality_reduction' (redução de dimensionalidade) ou 'clustering' (clusterização).
Chame o método fit() do objeto Model para treinar e avaliar os modelos. Esse método ajustará cada modelo aos dados de treinamento e avaliará seu desempenho nos dados de teste. Ele também armazenará os resultados em uma lista de pontuações e uma lista de previsões.

Chame o método evaluate() para imprimir as pontuações dos modelos em uma tabela e exibir um gráfico de barras que compara as pontuações. Ele também retornará o nome do modelo com a pontuação mais alta.

Opcionalmente, chame o método save_metrics(path) para salvar as métricas dos modelos em um arquivo CSV no caminho especificado.

=== Exemplo ===

# Carregar os dados
df = pd.read_csv('dados.csv')
X = df.drop('target', axis=1)
y = df['target']

# Separar em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelos para treinar
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'PCA': PCA(n_components=2),
    'K-Means': KMeans(n_clusters=2)
}

# Criar instância da classe Model
model = Model(models, X_train, y_train, X_test, y_test, 'classification')

# Treinar modelos e fazer previsões nos dados de teste
model.fit()

# Avaliar desempenho dos modelos
model.evaluate()

# Salvar métricas de desempenho em arquivo CSV
model.save_metrics('metrics.csv')

"""