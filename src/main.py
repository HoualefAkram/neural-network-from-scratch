# Neural Network From Scratch
from models.model import Model
from models.layer import Dense
from models.activation import SoftPlus, Linear

x_train = [0, 0.5, 1]
y_train = [0, 1, 0]

model = Model()

model.add_layer(layer=Dense(1, activation=Linear()))  # input layer
model.add_layer(layer=Dense(2, activation=SoftPlus()))  # hidden layer 1
model.add_layer(layer=Dense(1, activation=Linear()))  # output layer

model.fit(x_train=x_train, y_train=y_train)


neuron1 = model.layers[0].neurons[0]

neuron2 = model.layers[1].neurons[0]
neuron3 = model.layers[1].neurons[1]

neuron4 = model.layers[2].neurons[0]

print(neuron2.input_links)
print(neuron2.bias)
