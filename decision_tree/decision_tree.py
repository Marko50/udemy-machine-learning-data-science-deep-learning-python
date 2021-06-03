from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dataset import X, y


def decision_tree():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    accuracy = decision_tree_classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy} %")
