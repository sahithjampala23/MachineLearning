import pandas as pd
import numpy as np

"""
Implementation of standard perceptron algorithm. 
:param DataFrame train_df: Training dataset
:param DataFrame test_df: Training dataset
:param int T: Hyperparameter for how many epochs to iterate through.
:return vector w: The final learned weight vector
:return float avg_pred_err: The average prediction error on test dataset. 
"""


def standard_perceptron(train_df, test_df, T):
    # 1. Initialize weight vector.
    w = np.zeros(len(train_df.columns) - 1)
    test_errors = dict.fromkeys(i for i in range(1, T))
    testX = np.array(test_df.drop(['label'], axis=1))
    # 2. For epoch = 1...T
    for epoch in range(1, T + 1):
        # A. Shuffle data
        shuffled_df = train_df.sample(frac=1).reset_index(drop=True)
        X = np.array(shuffled_df.drop('label', axis=1))
        y = np.array(shuffled_df['label'])
        # B. For each training example...
        for i in range(len(shuffled_df)):
            # i. Find the error
            error = y[i] * (w.T.dot(X[i]))
            if error <= 0:
                # ii. Update weight when error <= 0.
                w = w + y[i] * X[i]

        predictions = []
        # C. Calculating prediction errors on test data for each epoch
        for i in range(0, len(testX)):
            predictions.append(np.sign(w.T.dot(testX[i])))
        test_df['prediction'] = predictions

        test_errors[epoch] = len(test_df[test_df['prediction'] != test_df['label']]) / len(test_df)

    avg_pred_err = sum(test_errors.values()) / T
    return w, avg_pred_err


"""
Implementation of voted perceptron algorithm. 
:param DataFrame train_df: Training dataset
:param DataFrame test_df: Training dataset
:param int T: Hyperparameter for how many epochs to iterate through.
:return dict W: The distinct weight vectors found.
:return dict C: The count of correctly predicting examples for each weight vector.
:return float (test_errors / T): The average prediction error on test dataset. 
"""


def voted_perceptron(train_df, test_df, T):
    # 1. Initialize W and m. (& C, training arrays, test array)
    w = np.zeros(len(train_df.columns) - 1)
    W = [w]
    m = 0
    C = dict.fromkeys(i for i in range(0, T))
    test_errors = 0
    testX = np.array(test_df.drop(['label'], axis=1))
    # 2. For epoch in 1...T:
    for epoch in range(1, T + 1):
        # A. Shuffle data
        shuffled_df = train_df.sample(frac=1).reset_index(drop=True)
        X = np.array(shuffled_df.drop('label', axis=1))
        y = np.array(shuffled_df['label'])
        # B. For each training example...
        for i in range(len(X)):
            # C. Update W if error <= 0. Also update count of correct predictions.
            error = y[i] * (W[m].T.dot(X[i]))
            if error <= 0:
                W.append(W[m] + y[i] * X[i])
                m = m + 1
                C[m] = 1
            else:
                C[m] = C[m] + 1
        # 3. Calculate predictions and errors for each epoch.
        final_predictions = []
        for j in range(len(testX)):
            predictions = []
            for i in range(1, len(W)):
                predictions.append(C[i] * np.sign(W[i].T.dot(testX[j])))
            final_predictions.append(np.sign(sum(predictions)))
        test_df['prediction'] = final_predictions
        test_errors += len(test_df[test_df['prediction'] != test_df['label']]) / len(test_df)

    return W, C, (test_errors / T)


"""
Implementation of standard perceptron algorithm. 
:param DataFrame train_df: Training dataset
:param DataFrame test_df: Training dataset
:param int T: Hyperparameter for how many epochs to iterate through.
:return vector a: The averaged learned weight vector.
:return float (test_errors / T): The average prediction error on test dataset. 
"""


def averaged_perceptron(train_df, test_df, T):
    w = np.zeros(len(train_df.columns) - 1)
    a = np.zeros(len(train_df.columns) - 1)
    test_errors = 0
    testX = np.array(test_df.drop(['label'], axis=1))
    for epoch in range(1, T + 1):
        shuffled_df = train_df.sample(frac=1).reset_index(drop=True)
        X = np.array(shuffled_df.drop('label', axis=1))
        y = np.array(shuffled_df['label'])
        for i in range(len(X)):
            error = y[i] * (w.T.dot(X[i]))
            if error <= 0:
                w = w + y[i] * X[i]

            a = a + w

        final_predictions = []
        for j in range(len(testX)):
            final_predictions.append(np.sign(a.T.dot(testX[j])))
        test_df['prediction'] = final_predictions
        test_errors += len(test_df[test_df['prediction'] != test_df['label']]) / len(test_df)
    return a, (test_errors / T)


if __name__ == '__main__':
    train_df = pd.read_csv('data/train.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
    test_df = pd.read_csv('data/test.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])

    train_df.loc[train_df.label == 0, "label"] = -1
    test_df.loc[test_df.label == 0, "label"] = -1

    print("Standard Perceptron: \n", "Learned Weight Vector & Error: ", standard_perceptron(train_df.copy(), test_df.copy(), 10))
    print("Voted Perceptron: \n", "Learned Weight Vectors, their Counts & Error: ", voted_perceptron(train_df.copy(), test_df.copy(), 10))
    print("Averaged Perceptron: \n", "Learned Weight Vectors, their Counts & Error: ", averaged_perceptron(train_df.copy(), test_df.copy(), 10))
