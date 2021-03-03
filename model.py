# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:14:53 2021

@author: SNARAHA1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
pip install scikit-learn
import sklearn

dataset=pd.read_csv('hiring.csv')
dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)

X=dataset.iloc[:,:-1]
X

def convert_to_int(word):
    word_dict={'one':1, 'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,
               'ten':10,'eleven':11,'tweleve':12,'zero':0,0:0}
    return word_dict[word]

X['experience']=X['experience'].apply(lambda x : convert_to_int(x))
X

y=dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()

regressor.fit(X,y)

#saving model to disk
pickle.dump(regressor,open('model.pkl','wb'))

#loading model to compare results

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))
