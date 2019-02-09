# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def modify_data(dataset):
    # Highly Blank Values & Correlation < Zero
    drop_column = ['Id','Alley', 'PoolQC', 'Fence', 'MiscFeature','MSSubClass','OverallCond','BsmtFinSF2','LowQualFinSF','BsmtHalfBath','KitchenAbvGr','EnclosedPorch','MiscVal','YrSold']
    dataset.drop(drop_column, axis = 1, inplace = True)
     
    #Filling BLank Values, integer with mean & Rest with mode
    null = dataset.isnull().sum()                                    
    nullColumns = null[null > 0].index.values.tolist()
    describe = dataset[nullColumns].describe()                       
    intNullColumns = describe.columns.values.tolist()
    
    for col in intNullColumns:
        dataset[col].fillna(dataset[col].mean(),inplace = True)  #Integer null columns with mean
        nullColumns.remove(col)
    for col in nullColumns:
        dataset[col].fillna(dataset[col].dropna().mode()[0],inplace = True) #String null columns with mode
    
    dataset.dropna(axis = 0 , how='all', inplace = True)
    
def remLeastUniqueColumns(dataset,columns):
    dataset.drop(columns, axis = 1, inplace = True)
    
        
    
raw_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data = raw_data.copy(deep = True)

data = [train_data,test_data]
for dataset in data:
    modify_data(dataset)

#correlation = raw_data.corr()
#leastCorrelation = correlation[correlation['SalePrice'] < 0].index.values.tolist()
#leastCorrelation
nunique = train_data.nunique()
nuniqueCol = nunique[nunique < 100].index.values.tolist()

for dataset in data:
    remLeastUniqueColumns(dataset,nuniqueCol)
