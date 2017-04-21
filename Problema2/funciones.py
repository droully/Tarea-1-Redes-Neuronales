import numpy as np


#Inicilizar la red con todos los pesos nulos
def red(n_inputs, n_hidden, n_outputs):
	red = []
	hidden_layer = [{'weights':[0 for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	red.append(hidden_layer)
	output_layer = [{'weights':[0 for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	red.append(output_layer)
	return red

# forwardpass
def forward(red, inputs):
	lista = inputs
	for capa in red:
		new_inputs = []
		for neuron in capa:
			
			act=np.dot(neuron["weights"][0:-1],lista[0:len(neuron["weights"])])+neuron["weights"][-1]#activa
			neuron['output'] = 1.0/(1.0+np.exp(-act)) #sigmoide
			new_inputs.append(neuron['output'])
			
		lista = new_inputs
	return lista


# Backward pass
def backward(network, expected):
	for i in reversed(range(len(network))):
		capa = network[i]
		errores = []
		if i != len(network)-1:#si la capa no es la ultima

			for j in range(len(capa)):
				error = 0.0
				for neuron in network[i + 1]:
					
					error += (neuron['weights'][j] * neuron['epsilon'])
				errores.append(error)
		else:
			
			for j in range(len(capa)):
				errores.append(expected[j] - capa[j]['output'])
		
		for j in range(len(capa)):
			derivada=capa[j]['output']*(1.0-capa[j]['output'])#derivada de la sigmoide
			capa[j]['epsilon'] = errores[j] * derivada #se guarda el error en la neurona



# Entrenamiento
def train(network, train_data, indices, l_rate, n_epoch, n_outputs,listaerror):
	for epoch in range(n_epoch):
		suma_error = 0
		for row in train_data:
			h=0
			outputs = forward(network, row)
			expected=[0 for i in range(n_outputs)]
			expected[indices[h]-1]=1
			suma_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward(network, expected)
			for i in range(len(network)):
				inputs = row[:-1]
				if i != 0:
					inputs = [neuron['output'] for neuron in network[i - 1]]
				for neuron in network[i]:
					for j in range(len(inputs)):
						neuron['weights'][j] += l_rate * neuron['epsilon'] * inputs[j]
					neuron['weights'][-1] += l_rate * neuron['epsilon']
			h=h+1
		listaerror.append(suma_error)
		
		print('>epoch=%d,error=%.3f' % (epoch,suma_error))		
		
def traindecay(network, train_data, indices, l_rate,w_decay, n_epoch, n_outputs,listaerror):
	for epoch in range(n_epoch):
		suma_error = 0
		for row in train_data:
			h=0
			outputs = forward(network, row)
			expected=[0 for i in range(n_outputs)]
			expected[indices[h]-1]=1
			suma_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward(network, expected)
			for i in range(len(network)):
				inputs = row[:-1]
				if i != 0:
					inputs = [neuron['output'] for neuron in network[i - 1]]
				for neuron in network[i]:
					for j in range(len(inputs)):
						neuron['weights'][j] += l_rate * neuron['epsilon'] * inputs[j] + w_decay*l_rate*neuron['weights'][j]
					neuron['weights'][-1] += l_rate * neuron['epsilon']+w_decay*l_rate*neuron['weights'][-1]
			h=h+1
		listaerror.append(suma_error)

		print('>epoch=%d,error=%.3f' % (epoch,suma_error))	
		
