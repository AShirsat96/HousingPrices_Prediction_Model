# -*- coding: utf-8 -*-
"""
@author: Aniket
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import numpy as np

### Loading and exploring the data
data = pd.read_csv('dataset.csv')
data.head(10)
data.info()

### Data Preprocessing

# Separate features (X) and target variable (y)
X = data.drop('SalePrice', axis=1)  # Features
y = data['SalePrice']  # Target variable

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Imputing missing values for numerical columns with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(X)
X_imputed[numerical_cols] = imputer.fit_transform(X_imputed[numerical_cols])

# One-hot encoding categorical variables
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X_imputed[categorical_cols])
X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate numerical and encoded categorical features
X_final = pd.concat([X_imputed.drop(categorical_cols, axis=1), X_encoded], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=100)

### Predicting the SalePrice using Linear Regression
# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")
# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

### Predicting the SalePrice using Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
### model evaluating metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Mean Squared Error: {mse_rf}")
# Calculate the Root Mean Squared Error (RMSE)
rmse_rf = np.sqrt(mse_rf)
print(f"Root Mean Squared Error (RMSE): {rmse_rf}")
# Calculate the R-squared value
r2_rf = r2_score(y_test, y_pred_rf)
print(f"R-squared: {r2_rf}")

### Predicting the SalePrice using Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor(random_state=100)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
print(f"Gradient Boosting Mean Squared Error: {mse_gb}")
rmse_gb = np.sqrt(mse_gb)
print(f"Root Mean Squared Error (RMSE): {rmse_gb}")
r2_gb = r2_score(y_test, y_pred_gb)
print(f"R-squared: {r2_gb}")

### Predicting the SalePrice using Support Vector Machine (SVM) Regressor
from sklearn.svm import SVR
svm_model = SVR()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
mse_svm = mean_squared_error(y_test, y_pred_svm)
print(f"SVM Mean Squared Error: {mse_svm}")
rmse_svm = np.sqrt(mse_svm)
print(f"Root Mean Squared Error (RMSE): {rmse_svm}")
r2_svm = r2_score(y_test, y_pred_svm)
print(f"R-squared: {r2_svm}")

# Create a list to store model names and their evaluation metrics
model_names = ['Linear Regression', 'Random Forest Regressor', 'Gradient Boosting Regressor', 'SVM Regressor']
rmse_scores = []
r2_scores = []

rmse_scores.append(rmse)
r2_scores.append(r2)

rmse_scores.append(rmse_rf)
r2_scores.append(r2_rf)

rmse_scores.append(rmse_gb)
r2_scores.append(r2_gb)

rmse_scores.append(rmse_svm)
r2_scores.append(r2_svm)

# Create a DataFrame to display the results
results_df = pd.DataFrame({
    'Model': model_names,
    'RMSE': rmse_scores,
    'R-squared': r2_scores
})

results_df





