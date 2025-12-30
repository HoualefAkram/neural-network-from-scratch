# Neural Network From Scratch
from models.model import Model
from models.layer import Input, Dense
from models.activation import SoftPlus, Linear

x_train = [[0], [0.5], [1]]
y_train = [[0], [1], [0]]
SHOW_GRAPH = True

model = Model()

model.add_layer(layer=Input(number_features=1))  # input layer
model.add_layer(layer=Dense(2, activation=SoftPlus()))  # hidden layer 1
model.add_layer(layer=Dense(1, activation=Linear()))  # output layer


model.fit(x_train=x_train, y_train=y_train, iterations=1000, learning_rate=1e-2)

prediction = model.predict(x_test=[0.5])
print(prediction)


if SHOW_GRAPH:
    from matplotlib import pyplot as plt

    x_axis = [i / 10 for i in range(10)]
    y_axis = [model.predict([x]) for x in x_axis]
    plt.plot(x_axis, y_axis)
    plt.show()
