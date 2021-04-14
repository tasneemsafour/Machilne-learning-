import numpy
import pandas as pd
'''arrays=[[3, 3, 7, 8, 5], [9, 3, 3, 3, 3], [9, 10, 11, 3, 23, 3, 3], [20, 3, 3, 3, 3, 3, 3, 3], [20, 3, 3, 3, 3, 3, 3]]


np_arrays = []

for array in arrays:
 for x in array:
    np_arrays.append(x)
print(np_arrays)'''
df_train = pd.DataFrame({'names': ['Jack', 'Helen', 'Nick', 'Helen']})
print(df_train['names'])