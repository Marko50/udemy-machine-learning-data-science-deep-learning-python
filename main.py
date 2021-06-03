from sys import argv
from logging import error

from decision_tree import decision_tree
from random_forest import random_forest

funcs = {
    'decision_tree': decision_tree,
    'random_forest': random_forest
}

if __name__ == '__main__':
    try:
        funcs[argv[1]]()
    except KeyError:
        error(f"No function named {argv[1]}")
