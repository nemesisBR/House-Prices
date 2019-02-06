# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data = raw_data.copy(deep = True)

data = [train_data, test_data]

print('Train Data Null: \n',train_data.isnull().sum())
#print("*" * 10)
print('Test Data Null: \n',test_data.isnull().sum())

#raw_data.describe( include = 'all')

for dataset in data:
    drop_column = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
    dataset.drop(drop_column, axis = 1, inplace = True)
    
# Features with large missing data :  LotFrontage, FireplaceQu
