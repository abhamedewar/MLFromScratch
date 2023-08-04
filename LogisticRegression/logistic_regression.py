import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LogisticRegression:

    def __init__(self, lr, epoch):
        self.lr = lr
        self.epoch = epoch
        self.weight = None
        self.bias = None
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features) 
        self.bias = 0
        for _ in range(self.epoch):
            linear_model = np.dot(X, self.weight) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weight -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_final = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_final


if __name__=="__main__":
    breast_cancer = datasets.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    lr = LogisticRegression(0.001, 600)
    lr.fit(X_train, Y_train)
    predictions = lr.predict(X_test)
    acc = np.sum(Y_test == predictions) / len(Y_test)
    print("Accuracy: ", acc)
