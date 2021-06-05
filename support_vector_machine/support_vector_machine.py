from dataset import X_train, y_train, X_test, y_test
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def support_vector_machine():
    svm_classifier = make_pipeline(StandardScaler(), SVC())
    svm_classifier.fit(X_train, y_train)
    accuracy = svm_classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy} %")
