import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf


# dataset
def data_set(n, test_size):
    x = np.array([[random() / 2 for _ in range(2)] for _ in range(n)])
    y = np.array([[i[0] + i[1]] for i in x])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = data_set(5000, 0.3)

    # build a model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation="sigmoid")])

    # compile
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='MSE', )
    # train
    model.fit(x_train, y_train, epochs=100)

    # evaluate
    model.evaluate(x_test, y_test, verbose=1)
    # predict

    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    print(model.predict(data))

