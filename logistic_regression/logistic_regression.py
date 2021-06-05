from sklearn.linear_model import LogisticRegression
from dataset import X_train, y_train, X_test, y_test


def logistic_regression():
    naive_bayes_class = LogisticRegression()
    naive_bayes_class.fit(X_train, y_train)
    accuracy = naive_bayes_class.score(X_test, y_test)
    print(f"Accuracy: {accuracy} %")
