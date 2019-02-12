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
nuniqueCol = nunique[nunique < 11].index.values.tolist()

for dataset in data:
    remLeastUniqueColumns(dataset,nuniqueCol)
    dataset.drop(['Exterior1st'], axis = 1, inplace = True)
    dataset.drop(['Exterior2nd'], axis = 1, inplace = True)

x = train_data.iloc[:,:-1].values
y = train_data.iloc[:,-1:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:,2] = labelencoder.fit_transform(x[:,2] )
onehotencoder = OneHotEncoder(categorical_features = [2])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
x = sc_X.fit_transform(x)
y = sc_Y.fit_transform(y)

temp = train_data['Neighborhood'].nunique()



'''labelencoder2 = LabelEncoder()
x[:,5] = labelencoder2.fit_transform(x[:,5] )
onehotencoder2 = OneHotEncoder(categorical_features = [5])
x = onehotencoder2.fit_transform(x).toarray()
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

'''from sklearn.linear_model import LinearRegression
model = LinearRegression()
'''
from sklearn.svm import SVR
model = SVR(kernel='rbf')

model.fit(x_train,y_train)
y_pred = sc_Y.inverse_transform(model.predict(x_test))


temp = sc_Y.inverse_transform(y_test)

test = test_data.iloc[:,:].values
labelencoder = LabelEncoder()
test[:,2] = labelencoder.fit_transform(test[:,2] )
onehotencoder = OneHotEncoder(categorical_features = [2])
test = onehotencoder.fit_transform(test).toarray()
test = test[:, 1:]
test =sc_X.fit_transform(test)

final = sc_Y.inverse_transform(model.predict(test))
final = pd.Series(final)

Id = pd.Series([x for x in range(1461,2920)])

result = pd.concat([Id, final], axis =1, names = ['ImageId', 'Label'])

result.to_csv('submission.csv', index=False )