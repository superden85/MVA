"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    # your code here #
    ##################
    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros(n_train)
    for i in range(n_train):
        card = np.random.randint(1, max_train_card+1)
        X_train[i, -card:] = np.random.randint(1, 11, size = card)
        y_train[i] = np.sum(X_train[i, -card:])

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    # your code here #
    ##################
    n_train = 10000
    X_test = list()
    y_test = list()
    for card in range(5, 101, 5):
        Xi = list()
        yi = list()
        for i in range(n_train):
            X = np.random.randint(1, 11, card)
            y = np.sum(X)
            
            Xi.append(X)
            yi.append(y)
        X_test.append(np.array(Xi))
        y_test.append(yi)
    return X_test, y_test