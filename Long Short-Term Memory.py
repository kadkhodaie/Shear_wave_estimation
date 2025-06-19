import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path, sheet_name='Sheet1')
    data = data.replace(' ', np.nan).dropna()  
    X = data[['RHOB (g/cc)', 'DTCO (us/ft)', 'GR (API)', 'NPHI (v/v)']].values.astype(float)
    y = data['Vs (Km/s)'].values.astype(float)
    return X, y

X_train, y_train = load_and_preprocess_data('Well A.xlsx')
X_test, y_test = load_and_preprocess_data('Well B.xlsx')

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

split_index = int(0.8 * len(X_train))  
X_val = X_train[split_index:]
y_val = y_train[split_index:]
X_train = X_train[:split_index]
y_train = y_train[:split_index]

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  
model.add(LSTM(50, activation='relu', return_sequences=False))  
model.add(Dropout(0.2))  
model.add(Dense(1))  
model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

predicted_data = pd.DataFrame(X_test.reshape(X_test.shape[0], -1), columns=['RHOB (g/cc)', 'DTCO (us/ft)', 'GR (API)', 'NPHI (v/v)'])
predicted_data['Vs (Km/s)'] = y_test
predicted_data['Vs Predicted (Km/s)'] = y_test_pred
predicted_data.to_excel('Predicted_Well_B_LSTM.xlsx', index=False)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"Training R2 score: {r2_train:.4f}")
print(f"Validation R2 score: {r2_val:.4f}")
print(f"Test R2 score: {r2_test:.4f}")

train_error = y_train - y_train_pred.flatten()
test_error = y_test - y_test_pred.flatten()

plt.figure(figsize=(14, 8))
plt.subplot(2,  1, 1)
plt.plot(y_train, label='Actual Vs (Km/s) - Train', color='blue', marker='o')
plt.plot(y_train_pred, label='Predicted Vs (Km/s) - Train', color='red', linestyle='--', marker='x')
plt.title(f'Training Set (Well A) - LSTM - Actual vs Predicted Vs (Km/s) - R2: {r2_train:.4f}')
plt.xlabel('Sample Index')
plt.ylabel('Vs (Km/s)')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(train_error, label='Error (Actual - Predicted) - Train', color='green', marker='s')
plt.title('Training Set (Well A) - Prediction Error')
plt.xlabel('Sample Index')
plt.ylabel('Error (Km/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(y_test, label='Actual Vs (Km/s)', color='blue', marker='o')
plt.plot(y_test_pred, label='Predicted Vs (Km/s)', color='red', linestyle='--', marker='x')
plt.title(f'LSTM - Actual vs Predicted Vs (Km/s) - R2: {r2_test:.4f}')
plt.xlabel('Number of Data')
plt.ylabel('Vs (Km/s)')
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(test_error, label='Error (Actual - Predicted)', color='green', marker='s')
plt.title('Prediction Error')
plt.xlabel('Number of Data')
plt.ylabel('Error (Km/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()