'''
    Wine database: 
        https://archive.ics.uci.edu/ml/machine-learning-databases/wine/
'''

import pandas as pd
df = pd.read_csv('data/winequality-red.csv', sep=';')
print( df.describe())