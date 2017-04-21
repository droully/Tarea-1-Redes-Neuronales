from __future__ import print_function
import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D,Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',\
'frog', 'horse', 'ship', 'truck']


def load_CIFAR_one(filename):
	with open(filename, 'rb') as f:
		datadict = pickle.load(f)
		X = datadict['data']
		Y = datadict['labels']
		Y = np.array(Y)
		return X, Y

def load_CIFAR10(PATH):
	xs = []
	ys = []
	for b in range(1,6):
		f = os.path.join(PATH, 'data_batch_%d' % (b, ))
		X, Y = load_CIFAR_one(f)
		xs.append(X)
		ys.append(Y)
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	del X, Y
	Xte, Yte = load_CIFAR_one(os.path.join(PATH, 'test_batch'))
	return Xtr, Ytr, Xte, Yte

Xtr,Ytr,Xte,Yte = load_CIFAR10('data/')

Xva = Xtr[0:10000].copy()
Yva = Ytr[0:10000].copy()



batch_size = 32
num_classes = 10
epochs = 400


# Convert class vectors to binary class matrices.
#Ytr = keras.utils.to_categorical(Ytr, num_classes)
#Yte = keras.utils.to_categorical(Yte, num_classes)

Xtr = Xtr.astype('float32')
Xte= Xte.astype('float32')

from sklearn.preprocessing import StandardScaler
from keras.regularizers import l1
from keras.layers import Dropout

scaler = StandardScaler().fit(Xtr)
Xtrs = pd.DataFrame(scaler.transform(Xtr))
#Xtes = pd.DataFrame(scaler.transform(Xte))

scalerVa = StandardScaler().fit(Xva)
Xvas = pd.DataFrame(scalerVa.transform(Xva))
Xtes = pd.DataFrame(scaler.transform(Xva))
 
# Convert class vectors to binary class matrices.
Ytr = keras.utils.to_categorical(Ytr, num_classes)
Yte = keras.utils.to_categorical(Yte, num_classes)
Yva = keras.utils.to_categorical(Yva, num_classes)

regu = 0.00001

model = Sequential()

model.add(Dense(2000, activation="relu", kernel_initializer="uniform", input_dim=Xtr.shape[1],W_regularizer=l1(regu)))
model.add(Dropout(0.4))
model.add(Dense(1200, activation="relu", kernel_initializer="uniform",W_regularizer=l1(regu)))
model.add(Dropout(0.25))
model.add(Dense(1200, activation="relu", kernel_initializer="uniform",W_regularizer=l1(regu)))
model.add(Dropout(0.25))
model.add(Dense(1200, activation="relu", kernel_initializer="uniform",W_regularizer=l1(regu)))
model.add(Dropout(0.25))
model.add(Dense(1200, activation="relu", kernel_initializer="uniform",W_regularizer=l1(regu)))
model.add(Dropout(0.25))
model.add(Dense(1200, activation="relu", kernel_initializer="uniform",W_regularizer=l1(regu)))
model.add(Dropout(0.25))
model.add(Dense(1200, activation="relu", kernel_initializer="uniform",W_regularizer=l1(regu)))
model.add(Dropout(0.25))
model.add(Dense(1200, activation="relu", kernel_initializer="uniform",W_regularizer=l1(regu)))
model.add(Dropout(0.25))
model.add(Dense(1200, activation="relu", kernel_initializer="uniform",W_regularizer=l1(regu)))
model.add(Dropout(0.25))
model.add(Dense(512, activation="relu", kernel_initializer="uniform",W_regularizer=l1(regu)))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation="softmax"))


sgd = SGD(lr=0.001, decay=1e-6, momentum = 0.8)
model.compile(optimizer=sgd ,loss='categorical_crossentropy',metrics=['accuracy'])

hist = model.fit(Xtrs.as_matrix(), Ytr, epochs=epochs,batch_size=batch_size, validation_data=(Xtes.as_matrix(), Yte))

print(hist.history.keys())

# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('problema4.jpg')
