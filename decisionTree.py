import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cc_file = pd.read_csv("./creditcard.csv")
print(cc_file['Class'].value_counts()) # unabalanced dataset

print(cc_file.isna()==True) # no null values
print(cc_file.shape)

# k fold stratification
#decision tree

X = cc_file.iloc[:, :-1]
y = cc_file.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)

# num_fold = 10
# skf = StratifiedKFold(n_splits = num_fold, shuffle = True, random_state = 42)

# depth_list = np.arange(31,32)
# max_accuracy = 0
# best_depth = 0
# for depth in depth_list:
#     print("depth",depth)
#     accuracies = []
#     for train_index, test_index in skf.split(X_train, y_train):
        
#         X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[test_index]
#         y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[test_index]

#         dt = DecisionTreeClassifier(max_depth = depth, random_state = 42)
#         model = dt.fit(X_train_fold, y_train_fold)
#         y_pred = model.predict(X_valid_fold)
#         accuracy = accuracy_score(y_valid_fold, y_pred)
#         accuracies.append(accuracy)
#         print(accuracy)

#     if max_accuracy < np.mean(accuracies):
#         max_accuracy = np.mean(accuracies)
#         best_depth = depth


dt = DecisionTreeClassifier(random_state = 42, criterion = 'entropy')
model = dt.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
# matrix = confusion_matrix(y_test, y_pred)
# print(matrix)
# sns.heatmap(matrix, annot=True)
# plt.show()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
precision = precision_score(y_test, y_pred)
print(precision)
recall = recall_score(y_test, y_pred)
print(recall)
fscore = f1_score(y_test, y_pred)
print(fscore)
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
sns.heatmap(matrix, annot = True, cmap = "crest")
plt.show()