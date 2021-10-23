import pandas as pd
import numpy as np
import pprint
import math
import random
import numpy.linalg as la

"""
Helper method which computes the gradient of the cost function J(w).
"""


def compute_grad_vec(x, y, w):
    grad_vec = np.zeros(len(w))
    for j in range(len(w)):
        grad_vec[j] = -sum([(y[i] - w.dot(x[i].T)) * x[i][j] for i in range(len(x))])
    return grad_vec


"""
Helper method which computes the cost function of a weight vector. 
Also called LMS.
"""


def cost_function(x, y, w):
    return 0.5 * sum([((y[i] - w.T.dot(x[i])) ** 2) for i in range(len(x))])


"""
Helper method which splits train and test datasets into array objects.
"""


def split_data(train, test):
    x = np.array(train.drop('label', axis=1))
    x_test = np.array(test.drop('label', axis=1))
    y = np.array(train['label'])
    y_test = np.array(test['label'])
    return x, x_test, y, y_test


"""
Algorithm for Batch Gradient Descent. 
"""


def batch_gradient_descent(train_df, test_df, w_0, max_iterations, r, threshold, converged=False):
    X, X_test, y, y_test = split_data(train_df, test_df)
    # 1. Initialize W[0]
    W = [w_0]
    gradient_J = []
    J = []
    t = 0
    # 2. Iterate until convergence (total error < threshold)
    while not converged:
        w = W[t]
        # i. Compute gradient of J(w)
        gradient_J.append(compute_grad_vec(X, y, w))
        J.append(cost_function(X, y, w))
        # ii. Update W: W[t+1] = W[t] - r*grad_J(w)
        W.append(w - r * gradient_J[t])

        t += 1  # increment step

        # Check for convergence
        temp_check = la.norm(W[t] - W[t - 1])
        if temp_check < threshold:
            converged = True
            final_w = W[t]
            cost_value_test = cost_function(X_test, y_test, final_w)
        if t > max_iterations:
            raise ValueError('The max number of iterations has been reached. Try lowering the learning rate r.')

    return final_w, cost_value_test, J


"""
Implementation of Stochastic Gradient Descent.
"""


def SGD(train_df, test_df, w_0, max_iterations, r, threshold, converged=False):
    # 1. Initialize W[0]
    W = [w_0]
    J = []
    t = 0
    # 2. Iterate until convergence.
    while not converged:
        # i. Take random sample of 1 example from training data.
        sample = train_df.sample(1, replace=True).reset_index(drop=True)
        X = np.array(sample.drop('label', axis=1))
        y = np.array(sample['label'])

        w = W[t]
        # ii. update W[t+1]
        new_w = np.zeros(len(w))
        for j in range(len(w)):
            new_w[j] = w[j] + r * (y - w.dot(X[0])) * X[0][j]

        W.append(new_w)
        t += 1  # Increment step
        # iii. Calculate total error for current weight vector.
        J.append(
            cost_function(np.array(train_df.drop('label', axis=1)), np.array(train_df['label']), w))
        temp_check = la.norm(W[t] - W[t - 1])
        # 3. Check for convergence
        if temp_check < threshold:
            converged = True
            final_w = W[t]
            cost_value_test = cost_function(np.array(test_df.drop('label', axis=1)), np.array(test_df['label']),
                                            final_w)
        if t > max_iterations:
            raise ValueError('The max number of iterations has been reached. Try lowering the learning rate r.')

    return final_w, cost_value_test, J


"""
Implementation of the analytical solution to LMS regression tasks.
"""


def analyze_LMS(train_df, test_df):
    X, X_test, Y, Y_test = split_data(train_df, test_df)
    X = X.T
    Y = Y.reshape(len(Y), 1)
    optimal_w_vec = la.inv(X.dot(X.T)).dot(X.dot(Y))
    test_cost_value = cost_function(X_test, Y_test, optimal_w_vec)
    return optimal_w_vec, test_cost_value


if __name__ == '__main__':
    concrete_train_df = pd.read_csv('data/train.csv',
                                    names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr',
                                           'label'])
    concrete_test_df = pd.read_csv('data/test.csv',
                                   names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr',
                                          'label'])

    # UNCOMMENT BELOW FOR BATCH GRADIENT DESCENT
    #learned_weight_vec, cost_func_val_test, cost_func_vals_train = batch_gradient_descent(concrete_train_df,concrete_test_df, np.zeros(len(concrete_train_df.columns) - 1), 20000, .005, 10 ** -6)

    # UNCOMMENT BELOW FOR STOCHASTIC GRADIENT DESCENT
    #learned_weight_vec, cost_func_val_test, cost_func_vals_train = SGD(concrete_train_df,concrete_test_df, np.zeros(len(concrete_train_df.columns) - 1), 20000, .005, 10 ** -6)

    # UNCOMMENT BELOW FOR ANALYTICAL SOLUTION OF LMS
    # learned_weight_vec, cost_func_val_test = analyze_LMS(concrete_train_df, concrete_test_df)