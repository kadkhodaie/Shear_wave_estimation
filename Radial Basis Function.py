import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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

def rbf_predict(X_train, y_train, X_test, sigma):
    predictions = []
    for x in X_test:
        
        distances = np.linalg.norm(X_train - x, axis=1)
        
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        
        prediction = np.sum(weights * y_train) / np.sum(weights)
        predictions.append(prediction)
    return np.array(predictions)

sigma = 0.2  

y_train_pred = rbf_predict(X_train, y_train, X_train, sigma)
y_test_pred = rbf_predict(X_train, y_train, X_test, sigma)

predicted_data = pd.DataFrame(X_test, columns=['RHOB (g/cc)', 'DTCO (us/ft)', 'GR (API)', 'NPHI (v/v)'])
predicted_data['Vs (Km/s)'] = y_test  
predicted_data['Vs Predicted (Km/s)'] = y_test_pred  
predicted_data.to_excel('Predicted_Well_B_RBF.xlsx', index=False)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"Training R2 score: {r2_train:.4f}")
print(f"Test R2 score: {r2_test:.4f}")

train_error = y_train - y_train_pred
test_error = y_test - y_test_pred

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(y_train, label='Actual Vs (Km/s) - Train', color='blue', marker='o')
plt.plot(y_train_pred, label='Predicted Vs (Km/s) - Train', color='red', linestyle='--', marker='x')
plt.title(f'Training Set (Well A) - Actual vs Predicted Vs (Km/s) - R2: {r2_train:.4f}')
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
plt.title(f'RBF - Actual vs Predicted Vs (Km/s) - R2: {r2_test:.4f}')
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