import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

cc_file = pd.read_csv("./creditcard.csv")
print(cc_file['Class'].value_counts()) # unabalanced dataset

print(cc_file.isna()==True) # no null values
print(cc_file.shape)

# k fold stratification
#decision tree

X = cc_file.iloc[:, :-1]
y = cc_file.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)

dt = DecisionTreeClassifier(random_state = 42)
model = dt.fit(X_train, y_train)

y_pred = model.predict(X_test)
total_count = len(y_test)
print(total_count)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

truep_1 = matrix[0][0]
truen_1 = matrix[1][1]
falsen_1 = matrix[1][0]
falsep_1 = matrix[0][1]

incorrect_indices = np.where(y_pred != y_test)[0]

X_test_incorrect = X_test.iloc[incorrect_indices]
y_test_incorrect = y_test.iloc[incorrect_indices]

# CNN code begins
#normalize the features
scaler1 = StandardScaler()
# scaler2 = StandardScaler()
X_standardized_train = scaler1.fit_transform(X_train)
X_standardized_test = scaler1.transform(X_test_incorrect)

#split into training and testing (20% testing data size)
# X_train_CNN,X_test_CNN,y_train_CNN,y_test_CNN= train_test_split(X_standardized,y_test_incorrect,test_size=0.2,stratify=y_test_incorrect,random_state=42)

#reshape data for CNN
X_train_CNN = X_standardized_train.reshape(X_standardized_train.shape[0],X_standardized_train.shape[1],1)
X_test_CNN = X_standardized_test.reshape(X_standardized_test.shape[0],X_standardized_test.shape[1],1)
y_train_CNN = y_train
y_test_CNN = y_test_incorrect

model = Sequential([
   Conv1D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = X_train_CNN[0].shape),
   MaxPooling1D(pool_size = 1),
   Conv1D(filters = 64, kernel_size = 3, activation = 'relu'),
   MaxPooling1D((2)),
   Conv1D(filters = 64, kernel_size = 3, activation = 'relu'),
   Flatten(),
   Dense(512,activation = 'relu'),
   Dense(1,activation = 'sigmoid')
])

print(model.summary())

model.compile(optimizer = 'rmsprop',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'])

# Train the model
history = model.fit(X_train_CNN, y_train_CNN, epochs = 5, validation_data = (X_test_CNN, y_test_CNN))

# loss, accuracy = model.evaluate(X_test_CNN, y_test_CNN)
# print(f'Test Loss: {loss}')
# print(f'Test Accuracy: {accuracy}')
y_pred_proba = model.predict(X_test_CNN)

# Convert probabilities to binary predictions
y_pred_CNN = (y_pred_proba > 0.5).astype('int32')

# Calculate precision, recall, and F1-score

matrix = confusion_matrix(y_test_CNN, y_pred_CNN)
# sns.heatmap(matrix,annot = True)
# plt.show()

truep_2 = matrix[0][0]
truen_2 = matrix[1][1]
falsen_2 = matrix[1][0]
falsep_2 = matrix[0][1]

accuracy = (truep_1 + truep_2 + truen_1 + truen_2) / total_count
precision = (truep_1 + truep_2) / (truep_1 + truep_2 + falsen_1 + falsen_2)
recall = (truep_1 + truep_2) / (truep_1 + truep_2 + falsep_2)
f1 = (2 * precision * recall) / (precision + recall)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

matrix = [[truep_1 + truep_2, falsep_2], [falsen_2, truen_1 + truen_2]]
sns.heatmap(matrix, annot = True, cmap = "crest")
plt.show()
#Time for CNN : 115
#Time for DT: 