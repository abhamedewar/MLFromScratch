import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, lr, epoch):
        self.lr = lr
        self.epoch = epoch
        self.weight = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for i in range(self.epoch):
            #hypothesis calculation
            y_pred = np.dot(X, self.weight) + self.bias
            #gradient descent, derivative of cost function w.r.t weights
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            #derivative of cost function w.r.t bias
            db = (1/n_samples) * np.sum(y_pred - y)
            #update the weights and bias
            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred

if __name__=="__main__":

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    regressor = LinearRegression(0.01, 400)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = (np.mean(y_test - predictions) ** 2)
    print("MSE: ", mse)

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()

