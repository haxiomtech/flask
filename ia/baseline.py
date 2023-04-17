from ia.base.mlearn.coremodels import *

# Carregar os dados tratados
df = pd.read_csv('base/data/dados.csv')
X = df.drop('target', axis=1)
y = df['target']

# Separar em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tipo de problema
problem_type = ''  # classification, regression, dimensionality_reduction, clustering

# Modelos
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

# Crie um objeto do tipo "Model"
model_pk = Model(models, X_train, X_test, y_train, y_test, problem_type)

# Ajuste os modelos aos dados
model_pk.fit()

# Salve o modelo treinado usando o pickle
with open('modelo_treinado.pkl', 'wb') as file:
    pickle.dump(model_pk, file)

