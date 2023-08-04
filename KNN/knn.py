from collections import Counter
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

class KNN:

    def __init__(self, k):
        #initialize the k value, k is the number of nearest neighbours that you want to find.
        self.k = k
    
    def fit(self, X, y):
        #assign train data and labels
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        
        #for every value of X(test points) find the euclidean distance between the test point and all the training data points
        y_pred = []
        for x in X:
            distances = []
            #running through all the training samples
            for x_train in self.X_train:
                #calculate euclidean distance
                dist = np.sqrt(np.sum((x - x_train)**2))
                distances.append(dist)

            #find the top k closest
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            #find the class of the most common label among the k labels
            most_common = Counter(k_labels).most_common(1)
            y_pred.append(most_common[0][0])
        
        return y_pred


if __name__=="__main__":

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    k = 3
    knn = KNN(k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = np.sum(y_test == predictions) / len(y_test)
    print("Accuracy: ", accuracy)
