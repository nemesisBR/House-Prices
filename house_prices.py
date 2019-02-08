# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def modify_data(dataset):
    drop_column = ['Id','Alley', 'PoolQC', 'Fence', 'MiscFeature']
    dataset.drop(drop_column, axis = 1, inplace = True)
    dataset = dataset.dropna(axis = 0 , how='all')
    
    null = dataset.isnull().sum()                                    #Remove Columns 'Alley', 'PoolQC', 'Fence', 'MiscFeature'
    nullColumns = null[null > 0].index.values.tolist()
    describe = dataset[nullColumns].describe()                       # Only 3 columns to be filed with int values inplace of na
    intNullColumns = describe.columns.values.tolist()
    
    for col in intNullColumns:
        dataset[col].fillna(dataset[col].mean(),inplace = True)         #Integer null columns with mean
        nullColumns.remove(col)
    for col in nullColumns:
        dataset[col] = dataset[col].str.encode('utf-8')
        dataset[col].fillna(dataset[col].mode(),inplace = True)         #String null columns with mode
    
    return dataset

raw_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data = raw_data.copy(deep = True)

train_data = modify_data(train_data)
test_data = modify_data(test_data)
    #dataset['LotFrontage','MasVnrArea','GarageYrBlt'] = 
# Features with large missing data :  LotFrontage, FireplaceQu

drop_column = ['Id','Alley', 'PoolQC', 'Fence', 'MiscFeature']
train_data.drop(drop_column, axis = 1, inplace = True)
train_data = train_data.dropna(axis = 0 , how='all')
    
null = train_data.isnull().sum()                                    #Remove Columns 'Alley', 'PoolQC', 'Fence', 'MiscFeature'
nullColumns = null[null > 0].index.values.tolist()
describe = train_data[nullColumns].describe()                       # Only 3 columns to be filed with int values inplace of na
intNullColumns = describe.columns.values.tolist()
    
for col in intNullColumns:
    dataset[col].fillna(dataset[col].mean(),inplace = True)         #Integer null columns with mean
    nullColumns.remove(col)
for col in nullColumns:
    dataset[col].fillna(dataset[col].mode(),inplace = True) 