from sklearn.naive_bayes import GaussianNB
from dataset import X_train, y_train, X_test, y_test


def naive_bayes():
    naive_bayes_class = GaussianNB()
    naive_bayes_class.fit(X_train, y_train)
    accuracy = naive_bayes_class.score(X_test, y_test)
    print(f"Accuracy: {accuracy} %")
