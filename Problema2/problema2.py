import pandas as pd
import matplotlib.pyplot as plt
import funciones as fun
from sklearn.preprocessing import StandardScaler
import random 
df=pd.read_csv("seeds.txt", sep='\t', index_col=7)
data= df.values.tolist()


a=data[15]



data=random.sample(data,100)
indices=df.index.tolist()
scaler=StandardScaler().fit(data)
data=scaler.transform(data)
n_inputs = len(data[0])
n_hidden=3
n_outputs = 3

network = fun.red(n_inputs, n_hidden, n_outputs)
lr=0.5
decay=0.001
n_epochs=10
err=[]
fun.traindecay(network, data,indices, lr,decay, n_epochs, n_outputs,err)

plt.plot(err)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
