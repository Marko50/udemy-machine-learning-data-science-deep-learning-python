from sys import argv
from logging import error

from decision_tree import decision_tree
from random_forest import random_forest
from knn import knn
from naive_bayes import naive_bayes
from logistic_regression import logistic_regression
from support_vector_machine import support_vector_machine
from neural_network import neural_network

funcs = {
    'decision_tree': decision_tree,
    'random_forest': random_forest,
    'knn': knn,
    'naive_bayes': naive_bayes,
    'logistic_regression': logistic_regression,
    'support_vector_machine': support_vector_machine,
    'neural_network': neural_network
}

if __name__ == '__main__':
    try:
        funcs[argv[1]]()
    except KeyError:
        error(f"No function named {argv[1]}")
