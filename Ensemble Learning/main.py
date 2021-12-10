
import pandas as pd
import numpy as np
import math


def dict_path(my_dict, path=None):
    if path is None:
        path = []
    for k, v in my_dict.items():
        newpath = path + [k]
        if isinstance(v, dict):
            for u in dict_path(v, newpath):
                yield u
        else:
            yield newpath, v


def format_tree(dict_path, tree):
    paths = list(dict_path(tree))
    label_paths_dict = {}
    for path in paths:
        tuple_path = [x for x in zip(*[iter(path[0])] * 2)]
        label_paths_dict.setdefault(path[1], []).append(dict(tuple_path))
    return label_paths_dict


def get_error_of_tree(label_paths_dict, train_df):
    total_error = 0.0
    error_counter = 0
    iteration_counter = 0
    for label, paths in label_paths_dict.items():
        for path in paths:
            sliced_df = train_df
            for attribute, value in path.items():
                sliced_df = sliced_df[train_df[attribute] == path[attribute]]
            error_counter = len(sliced_df[sliced_df['label'] != label])
            # correct_counter = len(sliced_df[sliced_df['label'] == label])
            instance_counter = len(sliced_df)
            if instance_counter > 0:
                prob_error = error_counter / instance_counter
                total_error += prob_error
            else:
                total_error += 0
            iteration_counter += 1
    return (total_error / iteration_counter)


def convert_numerical_to_binary(df):
    num_df = df.select_dtypes(include='number')
    ret_df = df
    for col in num_df.columns.values:
        ret_df[col].loc[ret_df[col] <= ret_df[col].median()] = 0
        ret_df[col].loc[ret_df[col] > ret_df[col].median()] = 1

    return ret_df


"""# HW2"""

"""
Implementation of ID3 algorithm that creates Decision Stumps. 
"""


def ID3_stump(S, Attributes, Label, max_depth):

import math
import random
from DecisionTree import main as dt
import pandas as pd
import numpy as np

"""
Implementation of ID3 algorithm to create Decision Stumps. 
:param DataFrame train_df: Training dataset
:param list Attributes: List of features/attributes to choose from for split.
:param Series Label: Label (or Y) of each example in training dataset.
:return Stump: A Decision Tree with only one feature and predictions for each value (Weak Learner).
"""


def ID3_stump(S, Attributes, Label):

    # 1. Base Cases
    if S['label'].nunique() == 1:
        return S['label'][0]
    if Attributes is None:
        return S['label'].mode()[0]

    # 2. Otherwise
    else:
        ig_dict = {attr: 0 for attr in Attributes}  # Initialize dictionary to store information gain of all attributes
        # A. Pick best attribute to split S
        for A in Attributes:
            ig_dict[A] = ig_w(S, A)
        # B. Create a Root Node for Tree
        best_attribute = max(ig_dict, key=ig_dict.get)
<<<<<<< HEAD
        max_depth -= 1
=======

>>>>>>> 90bce8c4474a02a42d0877a7b86722d3a1c86c3a
        tree = {}
        tree[best_attribute] = {}
        # Must assign a label to each value before returning the tree.
        # i. Slice dataset
        for value in S[best_attribute].unique():
            # i. Slice dataset
            S_v = S[S[best_attribute] == value]
            # ii. Get label counts for each different label
<<<<<<< HEAD

            pos_sum = sum(S_v[S_v.label == 1]['weight'])
            neg_sum = sum(S_v[S_v.label == -1]['weight'])
            # iii. Set leaf node
            if pos_sum >= neg_sum:
=======
            # l_val, count = np.unique(S_v['label'],return_counts=True)
            pos_sum = sum(S_v[S_v.label == 1]['weight'])
            neg_sum = sum(S_v[S_v.label == -1]['weight'])
            # iii. Set leaf node
            if pos_sum > neg_sum:
>>>>>>> 90bce8c4474a02a42d0877a7b86722d3a1c86c3a
                tree[best_attribute][value] = 1
            else:
                tree[best_attribute][value] = -1
        return tree
<<<<<<< HEAD

=======
        # C. For each possible value of A (Attribute)

    return tree


"""
Implementation of recursive Random Forest Decision Tree Algorithm.
:param DataFrame train_df: Training dataset
:param Series labels: Label (or Y) of each example in training dataset.
:param list A: List of features/attributes to sample from.
:param int n_features: The number of random samples from total features to be taken into consideration for each split.
:return tree: Returns a tree of all possible paths to classify the dataset.
"""


def RandTreeLearn(train_df, labels, A, n_features):
    # 1. Base Cases
    if train_df['label'].nunique() == 1:
        return train_df['label'].iloc[0]
    if len(A) == 0:
        return train_df['label'].mode()[0]

    # 2. Otherwise
    else:
        tree = {}
        if len(A) >= n_features:
            G = random.sample(A, n_features)
        else:
            G = A
        ig_dict = {attr: 0 for attr in G}  # Initialize dictionary to store information gain of all attributes

        # A. Pick best attribute to split S
        for a in G:
            ig_dict[a] = dt.ig(train_df, a)

        # B. Create a Root Node for Tree
        best_attribute = max(ig_dict, key=ig_dict.get)

        tree[best_attribute] = {}

        # C. For each possible value of A (Attribute)
        for value in train_df[best_attribute].unique():
            # i. Slice dataset
            S_v = train_df[train_df[best_attribute] == value]
            # iii. Set leaf node if pure
            if S_v.empty:
                # ii. Get label counts for each different label
                l_val, count = np.unique(train_df['label'], return_counts=True)
                tree[best_attribute][value] = l_val[0]
            # iv. Recurse for next best attribute to split until all leaf nodes are found
            else:
                tree[best_attribute][value] = RandTreeLearn(S_v[S_v.columns[~S_v.columns.isin([best_attribute])]],
                                                            S_v.columns.isin(['label']), list(
                        S_v.columns[~S_v.columns.isin([best_attribute, 'label'])]), n_features)
        return tree

    return tree


"""
<<<<<<< HEAD
Information Gain by using of Entropy.
=======
Information Gain by using Entropy calculated by weighted examples.
:param DataFrame S: Training dataset.
:param str A: The attribute being evaluated for its information gain on the dataset.
:return float ig: The information gain of attribute A on the dataset. 
>>>>>>> 90bce8c4474a02a42d0877a7b86722d3a1c86c3a
"""


def ig_w(S, A):
    # Calculate general entropy first
<<<<<<< HEAD
=======
    # prob_label_values = dict(S.label.value_counts(normalize=True))
    gen_entropy = 0
>>>>>>> 90bce8c4474a02a42d0877a7b86722d3a1c86c3a
    pos_sum = sum(S[S.label == 1]['weight'])
    neg_sum = sum(S[S.label == -1]['weight'])

    gen_entropy = -1 * (pos_sum * np.log(pos_sum)) - (neg_sum * np.log(neg_sum))
    # Store entropy and probability for each value an attribute can take.
    S_v = S[[A, 'label', 'weight']]  # Slice dataset
    prob_dict = dict.fromkeys([val for val in S_v[A].unique()])  # initialize dictionary for probabilities
    entropy_dict = {val: 0 for val in S_v[A].unique()}  # initialize dictionary for entropy
    for value in S_v[A].unique():
        temp_df = S_v[S_v[A] == value]
        prob_dict[value] = {'pos_sum': sum(temp_df[temp_df.label == 1]['weight']) / sum(temp_df['weight']),
                            'neg_sum': sum(temp_df[temp_df.label == -1]['weight']) / sum(temp_df['weight'])}
    for key, val in prob_dict.items():
        for v, prob in val.items():
            entropy_dict[key] += -1 * (prob * np.log(prob))

    subset_entropy = {val: 0 for val in S_v[A].unique()}
    for value in S_v[A].unique():
        prob_of_value = len(S_v[S_v[A] == value]) / len(S_v)
        subset_entropy[value] = prob_of_value * entropy_dict[value]
    ig = gen_entropy - sum(subset_entropy.values())

    return ig


<<<<<<< HEAD
def adaboost(train_df, T):
    m = len(train_df)
    D = dict.fromkeys([i for i in range(1, T)])
    alpha = dict.fromkeys([i for i in range(1, T)])
=======
"""
Implementation for adaboost. 
:param DataFrame train_df: Training dataset used to learn Decision Stumps (Weak Classifiers).
:param DataFrame test_df: Testing dataset used to find error of weak classifiers, as well as the final hypothesis.
:param int T: The number of iterations, or number of weak classifiers being made to create a final, strong classifier.
:return dict finalh_train_error: A dictionary, size T, of the training errors at each iteration.
:return dict finalh_test_error: A dictionary, size T, of the testing errors at each iteration.
"""


def adaboost(train_df, test_df, T):
    m = len(train_df)
    D = dict.fromkeys([i for i in range(1, T)])
    alpha = dict.fromkeys([i for i in range(1, T)])
    # 1. Initialize D_1 (and all data structures for value storage)
>>>>>>> 90bce8c4474a02a42d0877a7b86722d3a1c86c3a
    D[1] = [1 / m for i in range(0, m)]

    error = dict.fromkeys([i for i in range(1, T)])
    trees = dict.fromkeys([i for i in range(1, T)])
    formatted_trees = dict.fromkeys([i for i in range(1, T)])
<<<<<<< HEAD
    predictions = dict.fromkeys([i for i in range(1, T)])

    for t in range(1, T):
        # Weak learner
        bank_train_df['weight'] = D[t]
        trees[t] = ID3_stump(bank_train_df, list(
            bank_train_df.columns[~bank_train_df.columns.isin(['label', 'weight', 'prediction'])]),
                             bank_train_df.columns.isin(['label']), 1)
        formatted_trees[t] = format_tree(dict_path, trees[t])

        bank_train_df['prediction'] = 0
        for label, paths in formatted_trees[t].items():
            for path in paths:
                for attr, val in path.items():
                    bank_train_df.loc[bank_train_df[attr] == val, "prediction"] = label

        predictions[t] = [val for val in bank_train_df['prediction']]
        error[t] = bank_train_df.loc[bank_train_df['label'] != bank_train_df['prediction'], "weight"].sum()

        # Vote of learner
        alpha[t] = 0.5 * np.log((1 - error[t]) / error[t])
        D[t + 1] = []

        for i in range(0, len(bank_train_df)):
            D[t + 1].append((D[t][i] / sum(D[t])) * math.exp(-alpha[t] * bank_train_df['label'].loc[i] * bank_train_df['prediction'].loc[i]))

    bank_train_df['prediction'] = 0

    for i in range(0, len(bank_train_df)):
        for t in range(1, T):
            bank_train_df['prediction'].loc[i] += alpha[t] * predictions[t][i]

    # bank_train_df['prediction'] = np.sign(bank_train_df['prediction'])

    return train_df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bank_train_df = pd.read_csv('data/bank/train.csv',
                                names=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                       'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
                                       'poutcome', 'label'])
    bank_test_df = pd.read_csv('data/bank/test.csv',
=======
    train_error = dict.fromkeys([i for i in range(1, T)])
    train_predictions = dict.fromkeys([i for i in range(1, T)])
    test_error = dict.fromkeys([i for i in range(1, T)])
    test_predictions = dict.fromkeys([i for i in range(1, T)])
    finalh_train_error = dict.fromkeys([i for i in range(1, T)])
    finalh_test_error = dict.fromkeys([i for i in range(1, T)])

    # 2. Start iterations 1->T
    for t in range(1, T):
        # A. Find weak learner (Decision Stump) using ID3.
        train_df['weight'] = D[t]

        trees[t] = ID3_stump(train_df, list(
            train_df.columns[~train_df.columns.isin(['label', 'weight', 'prediction'])]),
                             train_df.columns.isin(['label']), 1)
        formatted_trees[t] = dt.format_tree(dt.dict_path, trees[t])
        # i. Set predictions for both training and testing datasets
        train_df['prediction'] = 0
        test_df['prediction'] = 0
        for label, paths in formatted_trees[t].items():
            for path in paths:
                for attr, val in path.items():
                    train_df.loc[train_df[attr] == val, "prediction"] = label
                    test_df.loc[test_df[attr] == val, "prediction"] = label
        # ii. Used for analysis and evaluation at each iteration.
        train_predictions[t] = [val for val in train_df['prediction']]
        error[t] = train_df.loc[train_df['label'] != train_df['prediction'], "weight"].sum()
        train_error[t] = len(train_df[train_df['label'] != train_df['prediction']]) / len(train_df)
        test_predictions[t] = [val for val in test_df['prediction']]
        test_error[t] = len(test_df[test_df['label'] != test_df['prediction']]) / len(test_df)

        # B. Compute vote (alpha) of learner
        alpha[t] = 0.5 * np.log((1 - error[t]) / error[t])
        D[t + 1] = []

        final_train_predictions = []
        final_test_predictions = []
        for i in range(0, len(train_df)):
            # C. Update values of weights for training examples
            D[t + 1].append((D[t][i] / sum(D[t])) * math.exp(
                -alpha[t] * train_df['label'].loc[i] * train_df['prediction'].loc[i]))
            # 3. Computing the final hypothesis
            hypothesis_train_i = []
            hypothesis_test_i = []
            # i. Creating inner nested loop for final hypothesis at each step. (Figure 1).
            for j in range(1, t + 1):
                hypothesis_train_i.append(alpha[j] * train_predictions[j][i])
                hypothesis_test_i.append(alpha[j] * test_predictions[j][i])
            final_train_predictions.append(np.sign(sum(hypothesis_train_i)))
            final_test_predictions.append(np.sign(sum(hypothesis_test_i)))
        train_df['prediction'] = final_train_predictions
        test_df['prediction'] = final_test_predictions
        finalh_train_error[t] = len(train_df[train_df['label'] != train_df['prediction']]) / len(
            train_df)
        finalh_test_error[t] = len(test_df[test_df['label'] != test_df['prediction']]) / len(
            test_df)

    return finalh_train_error, finalh_test_error


"""
Implementation of bagged trees algorithm.
:param DataFrame train_df: Training dataset.
:param DataFrame test_df: Testing dataset.
:param int T: The number of iterations, or number of trees being bagged to make a final, strong set of predictions.
:return dict avg_train_error: A dictionary, size T, of the training errors at each iteration.
:return dict avg_test_error: A dictionary, size T, of the testing errors at each iteration.
"""


def bagged_trees(train_df, test_df, T, algo_name=None, n_features=2):
    train_predictions = dict.fromkeys([i for i in range(1, T)])
    test_predictions = dict.fromkeys([i for i in range(1, T)])
    avg_test_error = dict.fromkeys([i for i in range(1, T)])
    avg_train_error = dict.fromkeys([i for i in range(1, T)])
    formatted_C = dict.fromkeys([i for i in range(1, T)])
    C = dict.fromkeys([i for i in range(1, T)])
    for t in range(1, T):
        sampled_train_df = train_df.sample(len(train_df), replace=True)
        C[t] = get_learner(sampled_train_df, algo_name, n_features)
        formatted_C[t] = dt.format_tree(dt.dict_path, C[t])

        temp_train_df = train_df.copy()
        temp_test_df = test_df.copy()
        temp_train_df['prediction'] = 0
        temp_test_df['prediction'] = 0
        for label, paths in formatted_C[t].items():
            for path in paths:
                temp_train_df.loc[eval(" & ".join(["(temp_train_df['{0}'] == {1})".format(attr, repr(value))
                                                   for attr, value in path.items()])), "prediction"] = label
                temp_test_df.loc[eval(" & ".join(["(temp_test_df['{0}'] == {1})".format(attr, repr(value))
                                                  for attr, value in path.items()])), "prediction"] = label

        train_predictions[t] = [val for val in temp_train_df['prediction']]
        test_predictions[t] = [val for val in temp_test_df['prediction']]
        final_train_prediction = []
        final_test_prediction = []
        for i in range(0, len(train_df)):
            train_pred_i = []
            test_pred_i = []
            for j in range(1, t + 1):
                train_pred_i.append(train_predictions[j][i])
                test_pred_i.append(test_predictions[j][i])
            final_train_prediction.append(np.sign(sum(train_pred_i) / t))
            final_test_prediction.append(np.sign(sum(test_pred_i) / t))
        temp_train_df['prediction'] = final_train_prediction
        temp_test_df['prediction'] = final_test_prediction
        avg_train_error[t] = len(temp_train_df[temp_train_df['prediction'] != temp_train_df['label']]) / len(
            temp_train_df)
        avg_test_error[t] = len(temp_test_df[temp_test_df['prediction'] != temp_test_df['label']]) / len(temp_test_df)
    return avg_train_error, avg_test_error


"""
Helper method which uses the specified type of learner to get the trees used in 
bagging algorithm. 
:param DataFrame df: Training dataset
:param str algo_name: The type of trees to be created. Default is 'ID3', but will compute RandomForest if 'forest' is passed.
:return tree: Classifier for the training dataset passed.
"""


def get_learner(df, algo_name, n_features=2):
    if algo_name == 'forest':
        return RandTreeLearn(df, df.columns.isin(['label']), list(df.columns[~df.columns.isin(['label'])]), n_features)
    else:
        return dt.ID3(df, list(df.columns[~df.columns.isin(['label'])]), df.columns.isin(['label']), setting, len(df.columns))


if __name__ == '__main__':
    # Datasets for Bank
    bank_train_df = pd.read_csv('data/train.csv',
                                names=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                       'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
                                       'poutcome', 'label'])
    bank_test_df = pd.read_csv('data/test.csv',
>>>>>>> 90bce8c4474a02a42d0877a7b86722d3a1c86c3a
                               names=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                      'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
                                      'poutcome', 'label'])

<<<<<<< HEAD
    bank_train_df = convert_numerical_to_binary(bank_train_df)
    bank_test_df = convert_numerical_to_binary(bank_test_df)
=======
    # When treating unknown as regular value
    bank_train_df = dt.convert_numerical_to_binary(bank_train_df)
    bank_test_df = dt.convert_numerical_to_binary(bank_test_df)
>>>>>>> 90bce8c4474a02a42d0877a7b86722d3a1c86c3a

    bank_train_df.loc[bank_train_df.label == 'no', "label"] = -1
    bank_train_df.loc[bank_train_df.label == 'yes', "label"] = 1

<<<<<<< HEAD
    T = 50
    print(adaboost(bank_train_df, T))
    # bank_train_df['prediction'] = np.sign(bank_train_df['prediction'])
=======
    bank_test_df.loc[bank_test_df.label == 'no', "label"] = -1
    bank_test_df.loc[bank_test_df.label == 'yes', "label"] = 1

    #print(adaboost(bank_train_df, bank_test_df, 500))
    #print(bagged_trees(bank_train_df, bank_test_df, 500, 'forest', 4))
>>>>>>> 90bce8c4474a02a42d0877a7b86722d3a1c86c3a
