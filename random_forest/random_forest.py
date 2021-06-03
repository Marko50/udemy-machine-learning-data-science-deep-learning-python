from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dataset import X, y


def random_forest():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X_train, y_train)
    accuracy = random_forest_classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy} %")
