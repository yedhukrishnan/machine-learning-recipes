from scipy.spatial import distance
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# A very basic Nearest Neigbour implementation which predicts
# using shortest euclidean distance between test data and training
# data
class NearestNeighbor():
    def predict(self, test_data):
        predictions = []
        for row in test_data:
            label = self.closest_neighbor(row)
            predictions.append(label)
        return predictions

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def closest_neighbor(self, position):
        closest_neighbor_distance = distance.euclidean(position, y_train[0])
        closest_neighbor_index = 0
        for i in range(1, len(X_train)):
            neighbor_distance = distance.euclidean(position, X_train[i])
            if closest_neighbor_distance > neighbor_distance:
                closest_neighbor_distance = neighbor_distance
                closest_neighbor_index = i
        return y_train[closest_neighbor_index]


# Import and extract out training and testing data
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

classifier = NearestNeighbor()
classifier.fit(X_train, y_train)

# Predictions
predictions = classifier.predict(X_test)
# print predictions

# Accuracy
print "Accuracy score for Basic Nearest Neighbor:"
print accuracy_score(y_test, predictions)
