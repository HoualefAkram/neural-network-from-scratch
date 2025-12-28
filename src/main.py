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

w1 = neuron2.input_links[0].weight

w2 = neuron3.input_links[0].weight

w3 = neuron4.input_links[0].weight
w4 = neuron4.input_links[1].weight
print(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}")

prediction = model.predict(x_test=[2])
print(prediction)
