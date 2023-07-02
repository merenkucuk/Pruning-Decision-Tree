from collections import Counter
from math import e, log
import numpy as np
import pandas as pd
from numpy import array
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# We do not use any ready made libraries to implement ID3. We implement own.
# Attention: we make the Attrition column place to last place.
# So the printed tree is printing according to this situation.
# We make the explanation above to avoid conflict. Pruning implementation is successful.

# read the csv file with pandas
# we use pandas only here
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
# get columns to change Attrition column place to last place
cols = df.columns.tolist()
cols = cols[0:1] + cols[2:] + cols[1:2]
df = df[cols]
# get columns names
column_names = list(df)
# get column ids (0,34)
column_ids = np.array(range(0, len(column_names)))
# make dataframe to numpy array
df = df.to_numpy()
# shuffle dataframe
np.random.shuffle(df)


# For continuous features, you can simply extract minimum and maximum value of your colon and then create certain number of intervals
# between your range of minimum and maximum values for the discretization process. You can choose any number of intervals suitable for you.
# we take the split number 5
# The function makes continuous features to discrete
def cont2disc(data, col_idx):
    max_value = data[:, col_idx].max()
    min_value = data[:, col_idx].min()
    min_interval = data[:, col_idx].min()
    column_range = []
    while min_interval <= max_value:
        column_range.append(int(min_interval))
        min_interval += (max_value - min_value) / 5
    for k in range(len(data[:, col_idx])):
        for i in range(len(column_range) - 1):
            if column_range[i + 1] >= data[:, col_idx][k] >= column_range[i]:
                data[:, col_idx][k] = i + 1
    return data


# makes continuous features to discrete with their indexes
# We firstly make Age column which has index 0 to discrete.
df = cont2disc(df, 0)
df = cont2disc(df, 2)
df = cont2disc(df, 4)
df = cont2disc(df, 5)
df = cont2disc(df, 8)
df = cont2disc(df, 9)
df = cont2disc(df, 11)
df = cont2disc(df, 12)
df = cont2disc(df, 13)
df = cont2disc(df, 15)
df = cont2disc(df, 17)
df = cont2disc(df, 18)
df = cont2disc(df, 19)
df = cont2disc(df, 22)
df = cont2disc(df, 23)
df = cont2disc(df, 24)
df = cont2disc(df, 26)
df = cont2disc(df, 27)
df = cont2disc(df, 28)
df = cont2disc(df, 29)
df = cont2disc(df, 30)
df = cont2disc(df, 31)
df = cont2disc(df, 32)
df = cont2disc(df, 33)

# Making the train set from the 60 percentage of data
train_data, x_remain = train_test_split(df, test_size=0.4)
# Making the validate and test set from the 40 percentage of data (20,20)
val_data, test_data = train_test_split(x_remain, test_size=0.5)


# calculate entropy to use when we find best gain for ID3
# when the entropy is high. we can say that distribution is polarized
def entropy(data, base=None):
    attrition_column = data[:, -1]
    n_attrition = len(attrition_column)
    if n_attrition <= 1:
        return 0
    value, counts = np.unique(attrition_column, return_counts=True)
    probs = counts / n_attrition
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent


# calculate information gain to use for ID3
# the function take dataset and col_id as parameter
# return best_gain_key as index for example 0 index 0 is Age
def calc_information_gain(data, col_ids):
    gains = {}
    for id in col_ids:
        values = [item for item, count in Counter(data[:, id]).items()]
        counts = [count for item, count in Counter(data[:, id]).items()]
        all_entropy = entropy(data)
        for i in range(len(values)):
            value_data = data[data[:, id] == values[i]]
            value_entropy = entropy(value_data)
            all_entropy -= (counts[i] / len(data[:, id])) * value_entropy
        gain = all_entropy
        gains[id] = gain
    best_gain_key = max(gains, key=gains.get)
    return gains[best_gain_key], best_gain_key


# A normal Node class
# which has children, values, rule, attribute_name and id features.
# there is also prev_node function to use while pruning
class Node:
    def __init__(self, attr_id=None, attr_name=None, values=None, rule=None):
        self.children = []
        self.values = values
        self.rule = rule
        self.max_gain = None
        self.is_leaf = False
        self.attr_name = attr_name
        self.attr_id = attr_id
        self.selectLeaf = None

    def prev_node(self, node):
        new_node = Node(node.attr_id, node.attr_name, node.values, node.rule)
        self.children = node.children
        self.values = node.values
        self.rule = node.rule
        self.max_gain = node.max_gain
        self.attr_name = node.attr_name
        self.attr_id = node.attr_id
        self.selectLeaf = node.selectLeaf
        return new_node


# An attribution dict
# The dict contains dataframe values and headers
# we make the attribution dict to apply to the model
attr_dict = {'Age': array([1, 2, 3, 4, 5, 60], dtype=object), 'BusinessTravel': array(['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
      dtype=object), 'DailyRate': array([1, 2, 3, 4, 5], dtype=object), 'Department': array(['Human Resources', 'Research & Development', 'Sales'],
      dtype=object), 'DistanceFromHome': array([1, 2, 3, 4, 5], dtype=object), 'Education': array([5],
      dtype=object), 'EducationField': array(['Human Resources', 'Life Sciences', 'Marketing', 'Medical','Other', 'Technical Degree'],
      dtype=object), 'EmployeeCount': array([1], dtype=object), 'EmployeeNumber': array([1, 2, 3, 4, 5],
      dtype=object), 'EnvironmentSatisfaction': array([5], dtype=object), 'Gender': array(['Female', 'Male'], dtype=object), 'HourlyRate': array([1, 2, 3, 4, 5],
      dtype=object), 'JobInvolvement': array([5], dtype=object), 'JobLevel': array([5],
      dtype=object), 'JobRole': array(['Healthcare Representative', 'Human Resources','Laboratory Technician', 'Manager', 'Manufacturing Director','Research Director', 'Research Scientist', 'Sales Executive','Sales Representative'],
      dtype=object), 'JobSatisfaction': array([5], dtype=object), 'MaritalStatus': array(['Divorced', 'Married', 'Single'],
      dtype=object), 'MonthlyIncome': array([1, 2, 3, 4, 5], dtype=object), 'MonthlyRate': array([1, 2, 3, 4, 5], dtype=object), 'NumCompaniesWorked': array([2, 3, 4, 5],
      dtype=object), 'Over18': array(['Y'], dtype=object), 'OverTime': array(['No', 'Yes'], dtype=object), 'PercentSalaryHike': array([1, 2, 3, 4, 23, 24, 25],
      dtype=object), 'PerformanceRating': array([1, 4], dtype=object), 'RelationshipSatisfaction': array([5], dtype=object), 'StandardHours': array([80],
      dtype=object), 'StockOptionLevel': array([4, 5], dtype=object), 'TotalWorkingYears': array([1, 2, 3, 4, 5], dtype=object), 'TrainingTimesLastYear': array([5],
      dtype=object), 'WorkLifeBalance': array([5], dtype=object), 'YearsAtCompany': array([1, 2, 3, 4, 5], dtype=object), 'YearsInCurrentRole': array([1, 2, 3, 4, 5],
      dtype=object), 'YearsSinceLastPromotion': array([1, 2, 3, 4, 5], dtype=object), 'YearsWithCurrManager': array([1, 2, 3, 4, 5], dtype=object), 'Attrition': array(['No', 'Yes'],
      dtype=object)}


def ID3(data, attr_ids, rule, nodes):
    # All of the attributes in the attribute ids list have their GAIN values computed,
    # and the calculated ones are taken out of the list.
    # The variable with the greatest GAIN value is designated as the new LEAF when the list is entirely empty.
    if len(attr_ids[:-1]) == 0:
        targets = [item for item, count in Counter(df[:, -1]).items()]
        counts = [count for item, count in Counter(df[:, -1]).items()]
        index, value = 0, counts[0]
        for i, v in enumerate(counts):
            if v > value:
                index, value = i, v
        rules = str(rule) + targets[index]
        print(rules)
        return Node(-1, column_names[-1], targets[index], rules)

    #  returns leaf node yes, no. when everything is ok.
    if len(set(data[:, -1])) == 1:
        b = Counter(data[:, -1])
        common_check = b.most_common(1)[0][0]
        rules = str(rule) + common_check
        print(rules)
        return Node(-1, column_names[-1], common_check, rules)

    # best information gain values and best information gain value index
    best_val, best_val_idx = calc_information_gain(data, attr_ids[:-1])
    # the best attribute's potential values
    possible_vals = attr_dict[column_names[best_val_idx]]

    child_node = Node(best_val_idx, column_names[best_val_idx], possible_vals, rule)
    child_node.max_gain = best_val
    child_node.rule += column_names[best_val_idx] + " ==> "

    # print all combination of node and child nodes
    for values in possible_vals:
        val_data = data[np.where(data[:, best_val_idx] == values)[0]]
        if val_data.shape[0] != 0:
            attr_idx_list = []
            for i in attr_ids:
                if i == best_val_idx:
                    continue
                else:
                    attr_idx_list.append(i)
            rules = str(child_node.rule) + str(values) + " ^ "
            child_node.children.append(ID3(val_data, attr_idx_list, rules, nodes))
        else:
            targets = [item for item, count in Counter(df[:, -1]).items()]
            counts = [count for item, count in Counter(df[:, -1]).items()]
            index, value = 0, counts[0]
            for i, v in enumerate(counts):
                if v > value:
                    index, value = i, v
            rules = str(child_node.rule) + str(values) + " ^ " + targets[index]
            print(rules)
            child_node.children.append(Node(-1, column_names[-1], targets[index], rules))

    # Finding which one is the most used Yes or No as leaf value
    targets = [item for item, count in Counter(data[:, -1]).items()]
    counts = [count for item, count in Counter(data[:, -1]).items()]
    index, value = 0, counts[0]
    for i, v in enumerate(counts):
        if v > value:
            index, value = i, v
    chooseLeafValue = targets[index]
    child_node.selectLeaf = chooseLeafValue
    nodes.append(child_node)

    return child_node


# This function evaluates the row in with the decision tree and returns values of node
def predict(node, row):
    node.attr_name = ""
    while node.attr_name != "Attrition":
        vals_list = []
        for i in range(len(node.values)):
            if node.values[i] == row[node.attr_id]:
                vals_list.append(i)
        next_node = node.children[vals_list[0]]
        node = next_node
        if node.attr_name == "Attrition":
            return node.values


print("Pre-Pruning Decision Tree Rules:")
nodes = []
des_tree = ID3(train_data, column_ids, "", nodes)
predicts = []
for i in val_data:
    x = predict(des_tree, i)
    predicts.append(x)
tn, fp, fn, tp = confusion_matrix(val_data[:, -1], predicts, labels=["Yes", "No"]).ravel()
prev_acc = (tp + tn) / (tn + fp + fn + tp)  # accuracy
predicts = []
for i in test_data:
    x = predict(des_tree, i)
    predicts.append(x)
tn, fp, fn, tp = confusion_matrix(test_data[:, -1], predicts, labels=["Yes", "No"]).ravel()
acc = (tp + tn) / (tn + fp + fn + tp)  # accuracy
print("Pre-Pruning Decision Tree Accuracy is ", acc)
# The twigs are the nodes whose children are all leaves.
# it means that a node which has children but its children do not have children

# Catalog all twigs in the tree
def catalog_twings(nodeList):
    twigs = []
    for node in nodeList:
        if node.children > []:
            is_twig = True
            for child in node.children:
                if child.children > []:
                    is_twig = False
            if is_twig:
                twigs.append(node)
    return twigs


while True:
    twigList = catalog_twings(nodes)
    # find the twig with the less gain
    twig_least_node, least_ig = Node(), 1
    for twig in twigList:
        if twig.max_gain <= least_ig:
            twig_least_node = twig
            least_ig = twig.max_gain

    prev_node = twig_least_node.prev_node(twig_least_node)
    word = twig_least_node.attr_name
    string = twig_least_node.rule
    words = string.split(' ')
    word_index = words.index(word)
    index = sum(len(x) + 1 for i, x in enumerate(words) if i < word_index)
    len_word = len(twig_least_node.attr_name)
    twig_least_node.rule = twig_least_node.rule[0:index] + twig_least_node.selectLeaf + twig_least_node.rule[index + len_word:]
    twig_least_node.attr_name = column_names[-1]
    twig_least_node.values = twig_least_node.selectLeaf
    twig_least_node.children = []

    # calculate accuracy and compare with last accuracy
    predicts = []
    for i in val_data:
        x = predict(des_tree, i)
        predicts.append(x)
    tn, fp, fn, tp = confusion_matrix(val_data[:, -1], predicts, labels=["Yes", "No"]).ravel()
    curr_acc = (tp + tn) / (tn + fp + fn + tp)  # accuracy
    # after pruning, if the current accuracy is less, fix the last twig
    if prev_acc > curr_acc:
        twig_least_node = prev_node.prev_node(prev_node)
        break
    prev_acc = curr_acc
    print()
    print("************************ BEFORE PRUNING ************************")
    print(prev_node.rule)
    print("************************ AFTER PRUNING ************************")
    print(twig_least_node.rule[:-4])

predicts = []
for i in test_data:
    x = predict(des_tree, i)
    predicts.append(x)
tn, fp, fn, tp = confusion_matrix(test_data[:, -1], predicts, labels=["Yes", "No"]).ravel()
curr_acc = (tp + tn) / (tn + fp + fn + tp)  # accuracy
print()
print("Post-Pruning Decision Tree Accuracy is ", curr_acc)
