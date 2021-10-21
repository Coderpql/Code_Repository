import numpy as np
import pandas as pd

data = pd.read_csv('data/ml-1m/train_test/test_80_triple.csv')

print(data.loc[[1, 4, 6], :])
print(data.loc[data['user_id'].isin([1, 4, 6])])