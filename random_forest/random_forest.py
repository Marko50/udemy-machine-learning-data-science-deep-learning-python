from sklearn.ensemble import RandomForestClassifier
from dataset import X_train, y_train, X_test, y_test


def random_forest():
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X_train, y_train)
    accuracy = random_forest_classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy} %")
