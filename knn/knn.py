from sklearn.neighbors import KNeighborsClassifier
from dataset import X_train, y_train, X_test, y_test


def knn():
    neighborhood_classifier = KNeighborsClassifier(n_neighbors=3)
    neighborhood_classifier.fit(X_train, y_train)
    accuracy = neighborhood_classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy} %")
