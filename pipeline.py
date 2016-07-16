from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print "Accuracy score for Decision Tree Classifier:"
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print "Accuracy score for Decision Tree Classifier:"
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
