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
from keras.optimizers import SGD
import matplotlib.pyplot as plt
n_decay = 10
lear_decay = np.logspace(-6,0,n_decay)

i=0
for x in lear_decay:
	model = Sequential()
	model.add(Dense(200, input_dim=X_train_scaled.shape[1], init='uniform', activation='sigmoid'))
	model.add(Dense(1, init='uniform', activation='linear'))

	sgd = SGD(lr=0.001, decay=x) #lear_decay!
	model.compile(optimizer=sgd,loss='mean_squared_error')

	hist = model.fit(X_train_scaled.as_matrix(), y_train.as_matrix(), nb_epoch=300,
	verbose=1, validation_data=(X_test_scaled.as_matrix(), y_test.as_matrix()))


	print(hist.history.keys())

	# summarize history for loss
	plt.plot(hist.history['val_loss'])
	plt.title('SIGMOID model loss with lear_decay: %f' % (x, ))
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['test'], loc='upper left')
	plt.savefig('images/g/lear_decay_%d.jpg' % (i, ))
	i=i+1
