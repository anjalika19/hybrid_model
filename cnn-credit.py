import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import seaborn as sns
import matplotlib.pyplot as plt

data_path = './creditcard.csv'
df = pd.read_csv(data_path)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#normalize the features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# print(X_standardized)

#split into training and testing (20% testing data size)
X_train,X_test,y_train,y_test = train_test_split(X_standardized, y, test_size = 0.3, stratify = y, random_state = 42)


#reshape data for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


model = Sequential([
   Conv1D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = X_train[0].shape),
   MaxPooling1D(pool_size = 1),
   Conv1D(filters = 64, kernel_size = 3, activation = 'relu'),
   MaxPooling1D((2)),
   Conv1D(filters = 64, kernel_size = 3, activation = 'relu'),
   Flatten(),
   Dense(512, activation = 'relu'),
   Dense(1, activation = 'sigmoid')
])

print(model.summary())

model.compile(optimizer = 'rmsprop',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs = 5, validation_data = (X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
y_pred_proba = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred = (y_pred_proba > 0.5).astype('int32')

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
sns.heatmap(matrix, annot = True, cmap = "crest")
plt.show()