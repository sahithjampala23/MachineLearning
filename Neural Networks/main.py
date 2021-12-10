import pandas as pd
import numpy as np


def sigmoid_func(s, derivative=False):
    if derivative == True:
        return s * (1 - s)

    return 1 / (1 + np.exp(-s))


class NeuralNetwork:
    def __init__(self, X, y, hidden_units):
        self.input = np.atleast_2d(X)

        self.w1 = np.random.normal(loc=0, size=(self.input.shape[1], hidden_units))
        # self.w1 = np.zeros((self.input.shape[1], hidden_units))
        self.w2 = np.random.normal(loc=0, size=(self.input.shape[1], hidden_units))
        # self.w2 = np.zeros((self.input.shape[1], hidden_units))
        self.w3 = np.random.normal(loc=0, size=(hidden_units, 1))
        # self.w3 = np.zeros((hidden_units,1))
        self.y = np.atleast_2d(y)
        self.output = np.zeros(y.shape)

    def forward_pass(self):
        self.z1 = np.atleast_2d(sigmoid_func(self.input.dot(self.w1)))
        self.z2 = np.atleast_2d(sigmoid_func(self.z1.dot(self.w2.T)))
        self.output = sigmoid_func(self.z2.dot(self.w3))
        return self.output

    def back_prop(self):
        output_error = self.y - self.output
        d_output = output_error * sigmoid_func(self.output, derivative=True)

        z2_error = d_output.dot(self.w3.T)
        d_z2 = z2_error * sigmoid_func(self.z2, derivative=True)

        z1_error = d_z2.dot(self.w1)
        d_z1 = z1_error * sigmoid_func(self.z1, derivative=True)

        self.w3 = self.w3 + self.z2.T.dot(d_output)
        self.w2 = self.w2.T + self.z1.T.dot(d_z2)
        self.w1 = self.w1 + self.input.T.dot(d_z1)

    def train(self, X, y, lr):
        self.output = self.forward_pass()
        self.back_prop()
        self.error = (np.square(y - self.output) * sigmoid_func(self.output, derivative=True))
        return self.error

def sgd(train_df, epochs, learning_rate, d, hidden_units):
    X_train = np.array(train_df.drop('label', axis=1))
    y_train = np.array(train_df['label'])

    lr = learning_rate
    errors = dict.fromkeys([e for e in range(1, epochs)])
    for epoch in range(1, epochs):
        shuffled_df = train_df.sample(frac=1)
        X = np.array(shuffled_df.drop('label', axis=1))
        y = np.array(shuffled_df['label'])
        errors[epoch] = []
        NN = NeuralNetwork(X[1], y[1], hidden_units)
        for i in range(len(X)):
            errors[epoch].append(NN.train(X[i], y[i], lr))

        lr = learning_rate / (1 + (learning_rate / d) * epoch)

    return errors

def find_error(df, epochs, lr, d, hidden_units):
  err = sgd(df, epochs, lr, d, hidden_units)
  avg_err = []
  for i in err.keys():
    avg_err.append(np.sum(err[i])/len(df))
  fin_error = np.mean(avg_err)
  return round(fin_error,5)

if __name__ == '__main__':
    train_df = pd.read_csv('data/train.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])
    test_df = pd.read_csv('data/test.csv', names=['variance', 'skewness', 'curtosis', 'entropy', 'label'])

    train_df['1'] = 1
    col = train_df.pop('1')
    train_df.insert(0, '1', col)

    test_df['1'] = 1
    col = test_df.pop('1')
    test_df.insert(0, '1', col)

    hidden_nodes = [5,10,25,50,100]
    train_errors = dict.fromkeys([h for h in hidden_nodes])
    test_errors = dict.fromkeys([h for h in hidden_nodes])
    for h in hidden_nodes:
        train_errors[h] = find_error(train_df, h, .1, .005, 5)
        test_errors[h] = find_error(test_df, h, .1, .005, 5)

    print("Training set errors: ",train_errors)
    print("Test set errors: ", test_errors)