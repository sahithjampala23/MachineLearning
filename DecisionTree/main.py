import pandas as pd
import numpy as np

"""
Implementation of ID3 algorithm. 
"""


def ID3(S, Attributes, Label, setting, max_depth):
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
            ig_dict[A] = find_best_attribute(S, A, setting)
        # B. Create a Root Node for Tree
        best_attribute = max(ig_dict, key=ig_dict.get)
        tree = {}
        tree[best_attribute] = {}
        if max_depth == 0:
            # Must assign a label to each value before returning the tree.
            for value in S[best_attribute].unique():
                # i. Slice dataset
                S_v = S[S[best_attribute] == value]
                # ii. Set label to be the most common occurence
                tree[best_attribute][value] = S_v['label'].mode()[0]

            return tree
        # C. For each possible value of A (Attribute)
        for value in S[best_attribute].unique():
            # i. Slice dataset
            S_v = S[S[best_attribute] == value]
            # ii. Get label counts for each different label
            l_val, count = np.unique(S_v['label'], return_counts=True)
            # iii. Set leaf node if pure
            if len(count) == 1:
                tree[best_attribute][value] = l_val[0]
            # iv. Recurse for next best attribute to split until all leaf nodes are found
            else:
                tree[best_attribute][value] = (
                    ID3(S_v, list(S_v.columns[~S_v.columns.isin([best_attribute, 'label'])]), S_v.columns[-1], setting,
                        max_depth - 1))

    return tree


"""
Helper method which finds the best attribute based on which setting is specified.
"""


def find_best_attribute(S, A, setting):
    if setting == 1:
        return ig(S, A)
    elif setting == 2:
        return majority_error(S, A)
    else:
        return gini_index(S, A)


"""
Information Gain by using of Entropy.
"""


def ig(S, A):
    # Calculate general entropy first
    prob_label_values = dict(S.label.value_counts(normalize=True))
    gen_entropy = 0
    for key, value in prob_label_values.items():
        gen_entropy += -1 * (value * np.log2(value))
    # Store entropy and probability for each value an attribute can take.
    S_v = S[[A, 'label']]  # Slice dataset
    prob_dict = dict.fromkeys([val for val in S_v[A].unique()])  # initialize dictionary for probabilities
    entropy_dict = {val: 0 for val in S_v[A].unique()}  # initialize dictionary for entropy
    for value in S_v[A].unique():
        temp_df = S_v[S_v[A] == value]
        prob_dict[value] = dict(temp_df['label'].value_counts(normalize=True))

    for key, val in prob_dict.items():
        for v, prob in val.items():
            entropy_dict[key] += -1 * (prob * np.log2(prob))

    subset_entropy = {val: 0 for val in S_v[A].unique()}
    for value in S_v[A].unique():
        prob_of_value = len(S_v[S_v[A] == value]) / len(S)
        subset_entropy[value] = prob_of_value * entropy_dict[value]

    ig = gen_entropy - sum(subset_entropy.values())
    return ig


"""
Information Gain by using Majority Error.
"""


def majority_error(S, A):
    # Calculate general ME first
    prob_label_values = dict(S.label.value_counts(normalize=True))
    general_ME = min(prob_label_values.values())
    # Calculate sum of probability * ME for each v of A

    S_v = S[[A, 'label']]  # Slice dataset
    prob_dict = dict.fromkeys([val for val in S_v[A].unique()])  # initialize dictionary for probabilities
    ME_dict = {val: 0 for val in S_v[A].unique()}  # initialize dictionary for entropy
    for value in S_v[A].unique():
        temp_df = S_v[S_v[A] == value]
        prob_dict[value] = dict(temp_df['label'].value_counts(normalize=True).reindex(S_v.label.unique(), fill_value=0))

    for key, val in prob_dict.items():
        ME_dict[key] = min(val.values())
    subset_ME = {val: 0 for val in S_v[A].unique()}
    for value in S_v[A].unique():
        prob_of_value = len(S_v[S_v[A] == value]) / len(S)
        subset_ME[value] = prob_of_value * ME_dict[value]

    ig = general_ME - sum(subset_ME.values())
    return ig


"""
Information Gain by using Gini Index.
"""


def gini_index(S, A):
    # Calculate general Gini Index first
    prob_label_values = dict(S.label.value_counts(normalize=True))
    gen_GI = 0
    for key, value in prob_label_values.items():
        gen_GI += (np.square(value))
    gen_GI = 1 - gen_GI
    # Store gini index and probability for each value an attribute can take.
    S_v = S[[A, 'label']]  # Slice dataset
    prob_dict = dict.fromkeys([val for val in S_v[A].unique()])  # initialize dictionary for probabilities
    gini_dict = {val: 0 for val in S_v[A].unique()}  # initialize dictionary for gini index
    for value in S_v[A].unique():
        temp_df = S_v[S_v[A] == value]
        prob_dict[value] = dict(temp_df['label'].value_counts(normalize=True))

    for key, val in prob_dict.items():
        for v, prob in val.items():
            gini_dict[key] += (np.square(prob))
        gini_dict[key] = 1 - gini_dict[key]
    subset_gini = {val: 0 for val in S_v[A].unique()}
    for value in S_v[A].unique():
        prob_of_value = len(S_v[S_v[A] == value]) / len(S)
        subset_gini[value] = prob_of_value * gini_dict[value]

    ig = gen_GI - sum(subset_gini.values())
    return ig


"""
Helper method which turns tree into a dictonary where:
Key: Label
Value: List of Paths to each Label
"""


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


"""
Helper method which takes the dictionary of list paths and turns them into dictionary of dictionaries
"""


def format_tree(dict_path, tree):
    paths = list(dict_path(tree))
    label_paths_dict = {}
    for path in paths:
        tuple_path = [x for x in zip(*[iter(path[0])] * 2)]
        label_paths_dict.setdefault(path[1], []).append(dict(tuple_path))
    return label_paths_dict


"""
Method which calculates the prediction error of a given tree.
"""


def get_error_of_tree(label_paths_dict, train_df):
    total_error = 0.0
    iteration_counter = 0
    for label, paths in label_paths_dict.items():
        for path in paths:
            sliced_df = train_df
            for attribute, value in path.items():
                sliced_df = sliced_df[train_df[attribute] == path[attribute]]
            error_counter = len(sliced_df[sliced_df['label'] != label])
            instance_counter = len(sliced_df)
            if instance_counter > 0:
                prob_error = error_counter / instance_counter
                total_error += prob_error
            else:
                total_error += 0
            iteration_counter += 1
    return total_error / iteration_counter


"""
Helper method which converts numerical attributes into binary 
by using the median value of that attribute as a threshold.
"""


def convert_numerical_to_binary(df):
    num_df = df.select_dtypes(include='number')
    ret_df = df
    for col in num_df.columns.values:
        ret_df[col].loc[ret_df[col] <= ret_df[col].median()] = 0
        ret_df[col].loc[ret_df[col] > ret_df[col].median()] = 1

    return ret_df


"""
Helper method which adjusts missing values to be the majority of the other values for same attribute.
"""


def convert_missing_attributes(df):
    for attr in df.columns:
        # condition is specifically for the attribute "poutcome" since unknown will be majority element.
        if df[attr].value_counts().keys()[0] == "unknown":
            df.loc[(df[attr] == "unknown"), attr] = df[attr].value_counts().keys()[1]
        else:
            df.loc[(df[attr] == "unknown"), attr] = df[attr].value_counts().keys()[0]
    return df


"""
Main method which creates tree with training dataset provided, and runs a series of tests to get average prediction error.
Tests are ran on a range of max depths provided for each tree, on each setting (information gain, majority error, gini index).
"""


def run_model(train_df, test_df):
    variant_error_dict_train = dict.fromkeys([1, 2, 3])
    variant_error_dict_test = dict.fromkeys([1, 2, 3])

    for i in range(1, 4):
        train_error_of_trees = []
        test_error_of_trees = []
        for j in range(1, len(train_df.columns)):
            tree = ID3(train_df, list(train_df.columns[~train_df.columns.isin(['label'])]),
                       train_df.columns[-1], i, j)
            label_paths_dict = format_tree(dict_path, tree)
            train_error_of_trees.append(get_error_of_tree(label_paths_dict, train_df))
            test_error_of_trees.append(get_error_of_tree(label_paths_dict, test_df))
        variant_error_dict_train[i] = np.mean(train_error_of_trees)
        variant_error_dict_test[i] = np.mean(test_error_of_trees)

    return variant_error_dict_train, variant_error_dict_test


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Datasets for Cars
    car_train_df = pd.read_csv('data/car/train.csv',
                               names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])
    car_test_df = pd.read_csv('data/car/test.csv',
                              names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'])

    # Datasets for Bank
    bank_train_df = pd.read_csv('data/bank/train.csv',
                                names=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                       'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
                                       'poutcome', 'label'])
    bank_test_df = pd.read_csv('data/bank/test.csv',
                               names=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                      'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
                                      'poutcome', 'label'])
    # When treating unknown as regular value
    bank_train_df = convert_numerical_to_binary(bank_train_df)
    bank_test_df = convert_numerical_to_binary(bank_test_df)

    # When treating unknown as missing value
    adjusted_bank_train_df = convert_missing_attributes(bank_train_df)
    adjusted_bank_test_df = convert_missing_attributes(bank_test_df)

    print(run_model(adjusted_bank_train_df, adjusted_bank_test_df))
