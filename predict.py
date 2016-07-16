from sklearn import tree

input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_data = [0, 1, 1, 0]

clf = tree.DecisionTreeClassifier()
clf.fit(input_data, output_data)

print clf.predict([1, 1])
