# Neural Network From Scratch
from models.model import Model
from models.layer import Input, Dense
from models.activation import SoftPlus, Linear

x_train = [[0], [0], [1], [2], [4], [7]]
y_train = [[0], [1], [0], [-1], [-2], [-2]]
SHOW_GRAPH = True

model = Model()

model.add_layer(layer=Input(number_features=1))  # input layer
model.add_layer(layer=Dense(4, activation=SoftPlus()))  # hidden layer 1
model.add_layer(layer=Dense(4, activation=SoftPlus()))  # hidden layer 2
model.add_layer(layer=Dense(1, activation=Linear()))  # output layer


model.fit(x_train=x_train, y_train=y_train, iterations=10000, learning_rate=0.1)


if SHOW_GRAPH:
    from matplotlib import pyplot as plt

    x_train_max = max([x[0] for x in x_train])
    x_train_min = min([x[0] for x in x_train])
    points_number = 10
    step = (x_train_max - x_train_min) / (points_number - 1)
    x_axis = []
    current_val = x_train_min
    for i in range(points_number):
        x_axis.append(current_val)
        current_val += step
    y_axis = [model.predict([x])[0] for x in x_axis]

    plt.plot(x_axis, y_axis)
    plt.scatter(
        [x[0] for x in x_train],
        [y[0] for y in y_train],
        color="red",
        label="Training Data",
    )
    plt.title("Neural Network Regression")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.show()
