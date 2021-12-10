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
        max_depth -= 1
        tree = {}
        tree[best_attribute] = {}
        # Must assign a label to each value before returning the tree.
        # i. Slice dataset
        for value in S[best_attribute].unique():
            # i. Slice dataset
            S_v = S[S[best_attribute] == value]
            # ii. Get label counts for each different label

            pos_sum = sum(S_v[S_v.label == 1]['weight'])
            neg_sum = sum(S_v[S_v.label == -1]['weight'])
            # iii. Set leaf node
            if pos_sum >= neg_sum:
                tree[best_attribute][value] = 1
            else:
                tree[best_attribute][value] = -1
        return tree

    return tree


"""
Information Gain by using of Entropy.
"""


def ig_w(S, A):
    # Calculate general entropy first
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


def adaboost(train_df, T):
    m = len(train_df)
    D = dict.fromkeys([i for i in range(1, T)])
    alpha = dict.fromkeys([i for i in range(1, T)])
    D[1] = [1 / m for i in range(0, m)]

    error = dict.fromkeys([i for i in range(1, T)])
    trees = dict.fromkeys([i for i in range(1, T)])
    formatted_trees = dict.fromkeys([i for i in range(1, T)])
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
                               names=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                      'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
                                      'poutcome', 'label'])

    bank_train_df = convert_numerical_to_binary(bank_train_df)
    bank_test_df = convert_numerical_to_binary(bank_test_df)

    bank_train_df.loc[bank_train_df.label == 'no', "label"] = -1
    bank_train_df.loc[bank_train_df.label == 'yes', "label"] = 1

    T = 50
    print(adaboost(bank_train_df, T))
    # bank_train_df['prediction'] = np.sign(bank_train_df['prediction'])
