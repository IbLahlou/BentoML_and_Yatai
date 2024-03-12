from sklearn import datasets
from sklearn import svm  # Import svm module

import bentoml

iris = datasets.load_iris()
X, y = iris.data , iris.target 

clf = svm.SVC(gamma='scale')
clf.fit(X, y)

saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model saved: {saved_model}")
