# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(r"C:\Users\meetu\Downloads\Programming\DeepLearning_projects\Grad_Admission_regression_DL\Grad_Admission_dataset\Admission_Predict_Ver1.1.csv")
df.head()

# %%
df.shape

# %%
df.drop(columns = ['Serial No.'], inplace = True)
df.head()

# %%
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# %%
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense # type: ignore

model = Sequential()

model.add(Dense(7, activation = 'relu', input_dim = 7))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

# %%
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs = 100, validation_split=0.2)

# %%
y_pred = model.predict(x_test)

# %%
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("r2 score: ", r2_score(y_test, y_pred))
print("mean absolute error: ", mean_absolute_error(y_test, y_pred))
print("mean squared error: ", mean_squared_error(y_test, y_pred))

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


