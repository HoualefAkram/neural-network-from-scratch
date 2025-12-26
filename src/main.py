# Neural Network From Scratch
from models.model import Model
from models.layer import Dense
from models.activation import SoftPlus

model = Model()
model.add_layer(layer=Dense(number_neurons=2, activation=SoftPlus()))
