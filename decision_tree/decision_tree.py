from sklearn.tree import DecisionTreeClassifier
from dataset import X_train, y_train, X_test, y_test


def decision_tree():
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    accuracy = decision_tree_classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy} %")
