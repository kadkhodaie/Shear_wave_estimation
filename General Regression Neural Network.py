import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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

def grnn_predict(X_train, y_train, X_test, sigma):
    predictions = []
    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        prediction = np.sum(weights * y_train) / np.sum(weights)
        predictions.append(prediction)
    return np.array(predictions)

def cross_validate_sigma(X, y, sigma_values, n_splits=5):
    best_sigma = None
    best_r2 = -np.inf
    kf = KFold(n_splits=n_splits)

    for sigma in sigma_values:
        r2_scores = []
        for train_index, val_index in kf.split(X):
            X_train_cv, X_val_cv = X[train_index], X[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            y_val_pred = grnn_predict(X_train_cv, y_train_cv, X_val_cv, sigma)
            r2 = r2_score(y_val_cv, y_val_pred)
            r2_scores.append(r2)
        
        mean_r2 = np.mean(r2_scores)
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_sigma = sigma
            
    return best_sigma

sigma_values = np.linspace(0.03, 0.8, 20)  
best_sigma = cross_validate_sigma(X_train, y_train, sigma_values)
print(f"Best sigma from cross-validation: {best_sigma:.4f}")

y_train_pred = grnn_predict(X_train, y_train, X_train, best_sigma)
y_test_pred = grnn_predict(X_train, y_train, X_test, best_sigma)

predicted_data = pd.DataFrame(X_test, columns=['RHOB (g/cc)', 'DTCO (us/ft)', 'GR (API)', 'NPHI (v/v)'])
predicted_data['Vs (Km/s)'] = y_test  
predicted_data['Vs Predicted (Km/s)'] = y_test_pred  
predicted_data.to_excel('Predicted_Well_B_GRNN.xlsx', index=False)

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
plt.title(f'GRNN - Actual vs Predicted Vs (Km/s) - R2: {r2_test:.4f}')
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