import pandas as pd
import numpy as np
from numpy.random import randn
df = pd.DataFrame(randn(5,4),['a','b','c','d','e'],['w','x','y','z'])

# print(pd.DataFrame(np.random.randn(5,5),[1,2,3,4,5],[1,2,3,4,5]))
# dataf = pd.DataFrame(np.random.randn(5,5),['a','b','c','d','e'],['6','7','8','9','0'])

# print(dataf['a']['6'])
# print (df[['w','x'],['a','b']])
# print (df['w','a'])
# print(df.loc['a']['w']
