from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from keras.optimizers import SGD

X, y = make_moons(500, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
Xtr = X[0:400].copy()
ytr = y[0:400].copy()
Xte = X[400:500].copy()
yte = y[400:500].copy()

# create model
model = Sequential()
model.add(Dense(30, input_dim=2, activation='sigmoid'))
model.add(Dense(12, activation='relu'))
#model.add(Dense(12, activation='relu'))
#model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
# Compile model
sgd = SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Fit the model
model.fit(Xtr, ytr, epochs=300, batch_size=10, validation_data=(Xte, yte))
# Scores
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)


#plt.show()
