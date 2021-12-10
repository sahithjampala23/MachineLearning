import pandas as pd
import numpy as np
from scipy import optimize


def SGD_SVM(train_df, test_df, T, steps, schedule=1):
    train_error = []
    test_error = []
    W = []
    for C in steps:

        # learning rate
        r_0 = .1
        # alpha
        a = .001
        w_0 = np.zeros(4)
        w = np.zeros(5)
        for t in range(1, T):
            shuffled_df = train_df.sample(frac=1)

            X = np.array(shuffled_df.drop('label', axis=1))
            y = np.array(shuffled_df['label'])
            if schedule == 1:
                r = r_0 / (1 + (r_0 / a) * t)
            else:
                r = r_0 / (1 + t)
            for i in range(len(X)):
                if y[i] * (w.T.dot(X[i])) <= 1:
                    w = w - r * (np.insert(w_0, len(w_0), 0)) + r * C * len(train_df) * y[i] * X[i]
                else:
                    w_0 = (1 - r) * w_0

        train_predictions = []

        X_train = np.array(train_df[['1', 'variance', 'skewness', 'curtosis', 'entropy']])
        X_test = np.array(test_df[['1', 'variance', 'skewness', 'curtosis', 'entropy']])
        for i in range(len(train_df)):
            train_predictions.append(np.sign(w.T.dot(X_train[i])))

        train_pred = pd.Series(train_predictions)
        train_error.append(len(train_df[train_df['label'] != train_pred]) / len(train_df))

        test_predictions = []
        for i in range(len(test_df)):
            test_predictions.append(np.sign(w.T.dot(X_test[i])))

        test_pred = pd.Series(test_predictions)
        test_error.append(len(test_df[test_df['label'] != test_pred]) / len(test_df))

        W.append(w)
    return train_error, test_error,W


def linear_kernel(X):
    return np.dot(X, X.T)


def gaussian_kernel(X1, X2, lr):
    k = np.broadcast(X1[:, np.newaxis], X2[np.newaxis, :])
    K = np.empty((k.shape))
    K.flat = [u - v for (u, v) in k]
    GK = -np.linalg.norm(K, axis=2)
    return np.exp(GK / lr)


def objective_function(alphas, y, X):
    return (0.5 * np.sum((np.outer(y, y)) * (np.outer(alphas, alphas)) * linear_kernel(X))) - np.sum(alphas)


def get_langrangian_multipliers(alpha_0, y, X, C):
    bnds = [(0, C)] * len(y)
    res = optimize.minimize(objective_function, alpha_0, args=(y, X), method='SLSQP', bounds=bnds)
    alphas = res.x
    return alphas


def ga_objective_function(alphas, y, K):
    return (0.5 * np.sum((np.outer(y, y)) * (np.outer(alphas, alphas)) * K)) - np.sum(alphas)


def get_langrangian_multipliers_nonlinear(alpha_0, y, X, C, lr):
    bnds = [(0, C)] * len(y)
    K = gaussian_kernel(X, X, lr)
    res = optimize.minimize(ga_objective_function, alpha_0, args=(y, K,), method='SLSQP', bounds=bnds)
    alphas = res.x
    return alphas


def learn_dual_SVM(X, y, alpha_0, C):
    alphas = get_langrangian_multipliers(alpha_0, y, X, C)

    indices = np.where(alphas > 0)

    w = np.zeros(X.shape[1])
    for j in indices[0]:
        w += alphas[j] * y[j] * X[j]

    B = []
    for j in indices[0]:
        B.append(y[j] - np.dot(w.T, X[j]))
    b = np.mean(B)
    return {'w': w, 'b': b}


def learn_dual_SVM(X, y, alpha_0, C):
    alphas = get_langrangian_multipliers(alpha_0, y, X, C)

    indices = np.where(alphas > 0)

    w = np.zeros(X.shape[1])
    for j in indices[0]:
        w += alphas[j] * y[j] * X[j]

    B = []
    for j in indices[0]:
        B.append(y[j] - np.dot(w.T, X[j]))
    b = np.mean(B)
    return {'w': w, 'b': b}


def predict(df, w, b):
    X = np.array(df.drop('label', axis=1))
    predictions = []
    for i in range(len(X)):
        predictions.append(np.sign(np.dot(w.T, X[i]) + b))
    pred = pd.Series(predictions)
    df_err = len(df[df['label'] != pred]) / len(df)
    return df_err


def run_dual_SVM(train_df, test_df, settings):
    X_train = np.array(train_df.drop('label', axis=1))
    y_train = np.array(train_df['label'])
    alpha_0 = np.zeros(len(y_train))

    feature_weights = []
    bias = []
    train_df_error = []
    test_df_error = []

    for C in settings:
        params = learn_dual_SVM(X_train, y_train, alpha_0, C)
        feature_weights.append(params['w'])
        bias.append(params['b'])
        train_df_error.append(predict(train_df, params['w'], params['b']))
        test_df_error.append(predict(test_df, params['w'], params['b']))
    return train_df_error, test_df_error


if __name__ == '__main__':
    sub_train_df = pd.read_csv('data/train.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
    sub_test_df = pd.read_csv('data/test.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])

    sub_train_df.loc[sub_train_df.label == 0, "label"] = -1
    sub_train_df['1'] = 1
    col = sub_train_df.pop('1')
    sub_train_df.insert(0, '1', col)

    sub_test_df.loc[sub_test_df.label == 0, "label"] = -1
    sub_test_df['1'] = 1
    col = sub_test_df.pop('1')
    sub_test_df.insert(0, '1', col)

    settings = [100 / 873, 500 / 873, 700 / 873]

    train_df = pd.read_csv('data/train.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
    test_df = pd.read_csv('data/test.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])

    train_df.loc[train_df.label == 0, "label"] = -1
    test_df.loc[test_df.label == 0, "label"] = -1

    print(SGD_SVM(sub_train_df,sub_test_df,100,settings))
    print(run_dual_SVM(train_df,test_df,settings))