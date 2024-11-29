# House Price Prediction Models

A Python implementation that compares different regression models for predicting house prices. The implementation includes Linear Regression, Random Forest, Gradient Boosting, and SVM models with comprehensive data preprocessing and evaluation metrics.

## Prerequisites

- Python 3.x
- pandas
- scikit-learn
- numpy

## Installation

Install required packages:
```bash
pip install pandas scikit-learn numpy
```

## Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
```

## Data Processing Pipeline

### 1. Data Loading
```python
data = pd.read_csv('dataset.csv')
```

### 2. Feature Preparation
```python
# Separate features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Handle categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
```

### 3. Data Preprocessing
- Missing value imputation
- One-hot encoding for categorical variables
- Feature concatenation
- Train-test split

## Models Implemented

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. Support Vector Machine (SVM) Regressor

## Usage

```python
# Data preprocessing
X_final = preprocess_data(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_final, 
    y, 
    test_size=0.2, 
    random_state=100
)

# Train and evaluate models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=100),
    'SVM': SVR()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(name, y_test, y_pred)
```

## Evaluation Metrics

For each model, the following metrics are calculated:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) Score

## Model Performance Comparison

The implementation creates a comparison DataFrame with:
- Model names
- RMSE scores
- R-squared scores

Example output format:
```
              Model         RMSE    R-squared
0  Linear Regression  34567.890     0.8234
1    Random Forest    23456.789     0.9123
2  Gradient Boosting  21345.678     0.9234
3     SVM Regressor   45678.901     0.7345
```

## Data Preprocessing Steps

1. Missing Value Handling:
   ```python
   imputer = SimpleImputer(strategy='mean')
   X_imputed[numerical_cols] = imputer.fit_transform(X_imputed[numerical_cols])
   ```

2. Categorical Encoding:
   ```python
   encoder = OneHotEncoder(drop='first', sparse_output=False)
   X_encoded = encoder.fit_transform(X_imputed[categorical_cols])
   ```

## Best Practices

1. Data Preparation:
   - Handle missing values appropriately
   - Encode categorical variables
   - Split data consistently

2. Model Training:
   - Use consistent random state
   - Implement cross-validation
   - Monitor for overfitting

3. Evaluation:
   - Use multiple metrics
   - Compare model performances
   - Document results

## Future Improvements

Consider implementing:
- Feature selection
- Hyperparameter tuning
- Cross-validation
- Model persistence
- Feature importance analysis
- Advanced preprocessing techniques
