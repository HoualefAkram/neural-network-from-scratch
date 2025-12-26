# Neural Network From Scratch
from models.model import Model
from models.layer import Dense
from models.activation import SoftPlus, Linear

x_train = [0, 0.5, 1]
y_train = [0, 1, 0]

model = Model()

model.add_layer(layer=Dense(number_neurons=1, activation=Linear()))  # input layer
model.add_layer(layer=Dense(number_neurons=2, activation=SoftPlus()))  # hidden layer 1
model.add_layer(layer=Dense(number_neurons=1, activation=Linear()))  # output layer

model.fit(x_train=x_train, y_train=y_train)
