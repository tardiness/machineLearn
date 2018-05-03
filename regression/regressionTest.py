import numpy as np
import matplotlib.pyplot as plt
from SimpleLinearRegression import SimpleLinearRegression


def simple_linear_regression():
    sig = SimpleLinearRegression()
    x_train = np.array([1., 2., 3., 4., 5.])
    y_train = np.array([1., 3., 2., 3., 5.])

    x_predict = 6

    sig.fit(x_train, y_train)
    y_predict = sig.predict(np.array([x_predict]))
    print(y_predict)
    y_hat = sig.predict(x_train)
    plt.scatter(x_train, y_train)
    plt.plot(x_train, y_hat, color="r")
    plt.axis([0, 6, 0, 6])
    plt.show()


if __name__ == "__main__":
    simple_linear_regression()
