import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import keras
import sklearn
df = pd.read_csv("housing2.data", sep=',',header=None, names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
'RM', 'AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])

from sklearn.cross_validation import train_test_split

df_train,df_test= train_test_split(df,test_size=0.25, random_state=0)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df)
x_train_scaled = pd.DataFrame(scaler.transform(df_train),columns=df_train.columns)
y_train_scaled = df_train.pop('MEDV')

x_test_scaled = pd.DataFrame(scaler.transform(df_test),columns=df_test.columns)
y_test_scaled = df_test.pop('MEDV')



from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD




from sklearn import cross_validation


xm = x_train_scaled.as_matrix()
ym = y_train_scaled.as_matrix()
kfold = cross_validation.KFold(len(xm), 10)
cvscores = []
for i, (train, val) in enumerate(kfold):
# create model
	model = Sequential()
	model.add(Dense(200, input_dim=xm.shape[1], init='uniform'))
	model.add(Activation('relu'))
	model.add(Dense(1, init='uniform'))
	model.add(Activation('linear'))
	# Compile model
	sgd = SGD(lr=0.01)
	model.compile(optimizer=sgd,loss='mean_squared_error',verbose="0")
	# Fit the model
	model.fit(xm[train], ym[train], nb_epoch=300,verbose="0")
	# evaluate the model
	scores = model.evaluate(xm[val], ym[val])
	cvscores.append(scores)
print("Error:",np.mean(cvscores))
