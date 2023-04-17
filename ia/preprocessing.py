from ia.base.pproc.encoder import *
from ia.base.pproc.norm import *
from ia.base.pproc.traintest import *
from ia.base.pproc.classbalance import *

import pandas as pd

# Carregar dados
db = pd.read_csv('ia/base/data/dados.csv')

hd_db = db.head()

# Calcula valores nulos
tabela_valores_nulos = pd.DataFrame({
    'Nulos': db.isna().sum(),
    '% Nulos': (db.isna().sum() / db.shape[0] * 100).round(2).astype(str) + '%',
    'Tipos': db.dtypes,
    'Val. Un.': db.nunique()
})

# Calcula estatísticas descritivas
descr = db.describe().transpose()

# Remove a coluna "count"
descr_r = descr.drop("count", axis=1)

# Arredonda os valores para duas casas decimais
descr_a = descr_r.round(2)


# Cria tabela de resumo
df_resumo = pd.merge(tabela_valores_nulos, descr_a, how='outer', right_index=True, left_index=True)

# Valores NA ficam em branco
df_resumo = df_resumo.fillna('-')



