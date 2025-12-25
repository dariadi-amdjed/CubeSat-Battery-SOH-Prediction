"This Code You Can Use"
"By Amdjed Dariadi"

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


mat = scipy.io.loadmat('data/B0005.mat')
raw_data = mat['B0005'][0, 0]['cycle'][0]

capacities = []
for i in range(len(raw_data)):
    if raw_data[i]['type'][0] == 'discharge':
        cap = raw_data[i]['data'][0, 0]['Capacity'][0][0]
        capacities.append(cap)


data = np.array(capacities).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        X.append(dataset[i:(i+look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(scaled_data)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(1, 10)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

print("Start Training The Model .... Waiting")
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

print("Success Train The Data")

predictions = model.predict(X_train)
predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(10,6))
plt.plot(capacities[11:], label='Actual Capacity (NASA)')
plt.plot(predictions, label='LSTM Predicted Capacity', linestyle='--')
plt.title('Comparison: Actual vs Predicted Capacity')
plt.legend()
plt.show()

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(capacities[11:], predictions)
rmse = np.sqrt(mean_squared_error(capacities[11:], predictions))

print(f"Value OF R-squared  {r2:.4f}")
print(f"Value OF RMSE (Error Rate) is: {rmse:.4f}")
