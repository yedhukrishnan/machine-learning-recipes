from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image
import numpy as np
from sklearn.externals.six import StringIO
import pydotplus

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

test_data = np.array([4.8, 2.1, 1.5, 0.25])
test_data = test_data.reshape(1, -1)
print clf.predict(test_data)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
        feature_names = iris.feature_names,
        class_names = iris.target_names,
        filled = True, rounded = True,
        special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
# Image(graph.create_png())
