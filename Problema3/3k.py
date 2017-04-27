import pandas as pd
import numpy as np

#url = 'http://mldata.org/repository/data/download/csv/regression-datasets-housing/'
url = 'data/housing.csv'
df = pd.read_csv(url, sep=',',header=None, names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
'RM', 'AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
from sklearn.cross_validation import train_test_split
df_train,df_test= train_test_split(df,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df_train)
X_train_scaled = pd.DataFrame(scaler.transform(df_train),columns=df_train.columns)
y_train = df_train.pop('MEDV')

X_test_scaled = pd.DataFrame(scaler.transform(df_test),columns=df_test.columns)
y_test = df_test.pop('MEDV')


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
import matplotlib.pyplot as plt
import numpy as np
from keras.regularizers import l2,l1

n_w_re = 10
w_re = np.linspace(0.000001,1,n_w_re)


i=0
j=0
for m in w_re:
	model = Sequential()
	#la regularization se debe incorporar a cada capa separadamente
	idim=X_train_scaled.shape[1]
	model.add(Dense(200,input_dim=idim,init='uniform',W_regularizer=l1(m)))
	model.add(Activation('relu'))
	model.add(Dense(2000,init='uniform',W_regularizer=l1(m)))
	model.add(Activation('relu'))
	model.add(Dense(1, init='uniform',W_regularizer=l1(m)))
	model.add(Activation('linear'))


   	sgd = SGD(lr=0.001)

	model.compile(optimizer=sgd,loss='mean_squared_error')

	hist = model.fit(X_train_scaled.as_matrix(), y_train.as_matrix(), nb_epoch=300,verbose=1, validation_data=(X_test_scaled.as_matrix(), y_test.as_matrix()))


	# summarize history for loss
	plt.plot(hist.history['val_loss'])
	plt.title('RELU (With hidden-L) model loss with w_regularizer L1: %f (lr=%f)' % (m,0.001, ))
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['test'], loc='upper left')
	plt.savefig('images/k/w_regularizer_%d.jpg' % (i, ))
	i=i+1


