import numpy as np

'''
# Eliminação de Outliers

outcol = list(['Price', 'Transmission'])

for i in outcol:
    Q1 = np.percentile(db[i], 25)
    Q3 = np.percentile(db[i], 75)

    IQR = Q3 - Q1

    Qupper = np.where(db[i] >= (Q3 + 1.5 * IQR))
    Qlower = np.where(db[i] <= (Q1 - 1.5 * IQR))

    # Na fórmula acima , de acordo com as estatísticas , o aumento de 0 ,5 de IQR (new_IQR = IQR + 0 ,5 * IQR) é 
    considerado , para considerar todos os data entre 2 ,7 desvios padrão na Distribuição Gaussiana.

    db.drop(Qupper[0], inplace=True)
    db.drop(Qlower[0], inplace=True)
'''
