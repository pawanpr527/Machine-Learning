# import pandas as pd
# import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_log_error
# from xgboost import XGBRegressor

# # Load data
# train_data = pd.read_csv('HousePrice/train.csv')
# test_data = pd.read_csv('HousePrice/test.csv')

# # Drop outliers (known based on EDA)
# train_data = train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)].index)

# # Target (log transform)
# y = np.log1p(train_data['SalePrice'])

# # Drop ID and target from train, keep test ID for submission
# train_data.drop(['Id', 'SalePrice'], axis=1, inplace=True)
# test_ids = test_data['Id']
# test_data.drop(['Id'], axis=1, inplace=True)

# # Combine for consistent encoding
# all_data = pd.concat([train_data, test_data], axis=0)

# # Separate columns
# numerical_cols = all_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
# categorical_cols = all_data.select_dtypes(include=['object']).columns.tolist()

# # Remove "Utilities" (useless column)
# if 'Utilities' in categorical_cols:
#     categorical_cols.remove('Utilities')
#     all_data.drop('Utilities', axis=1, inplace=True)

# # Preprocessing for numeric and categorical
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Combine transformers
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numerical_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])

# # Full pipeline
# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', XGBRegressor(
#         n_estimators=2500,
#         learning_rate=0.01,
#         max_depth=3,
#         subsample=0.7,
#         colsample_bytree=0.7,
#         reg_alpha=0.1,
#         reg_lambda=1,
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# # Split original train set for validation
# X = all_data.iloc[:len(y), :]
# X_test_final = all_data.iloc[len(y):, :]

# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train
# model.fit(X_train, y_train)

# # Validate
# val_preds = model.predict(X_valid)
# val_score = np.sqrt(mean_squared_log_error(y_valid, val_preds))
# print(f"Validation RMSLE: {val_score:.5f}")

# # Predict on test set and inverse log transform
# final_preds = np.expm1(model.predict(X_test_final))

# # Create submission
# submission = pd.DataFrame({
#     'Id': test_ids,
#     'SalePrice': final_preds
# })

# submission.to_csv('submission.csv', index=False)
# print("Submission saved as submission.csv")


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Load data
train = pd.read_csv('HousePrice/train.csv')
test = pd.read_csv('HousePrice/test.csv')

# Drop outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# Log transform target
y = np.log1p(train['SalePrice'])
test_ids = test['Id']
train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# Feature engineering
def add_features(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasGarage'] = df['GarageType'].notnull().astype(int)
    df['GrLivArea2'] = df['GrLivArea'] ** 2
    df['OverallQual2'] = df['OverallQual'] ** 2
    df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    return df

train = add_features(train)
test = add_features(test)

# Combine for processing
all_data = pd.concat([train, test], axis=0)

# Fix dtypes
categorical_cols = all_data.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    all_data[col] = all_data[col].astype('category')

numerical_cols = all_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove low variance features later
if 'Utilities' in all_data.columns:
    all_data.drop('Utilities', axis=1, inplace=True)
    if 'Utilities' in categorical_cols:
        categorical_cols.remove('Utilities')

# Pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(include_bias=False, degree=2)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Fit-transform
X_full = all_data.copy()
X_processed = preprocessor.fit_transform(X_full)
X_processed = VarianceThreshold(threshold=0.0).fit_transform(X_processed)

X = X_processed[:len(y)]
X_test = X_processed[len(y):]

# Base models
xgb = XGBRegressor(n_estimators=2500, learning_rate=0.01, max_depth=3,
                  subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1, random_state=42, n_jobs=-1)

lgb = LGBMRegressor(n_estimators=2500, learning_rate=0.01, max_depth=6,
                    subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1,
                    reg_lambda=1, random_state=42, n_jobs=-1)

lasso = LassoCV(alphas=np.logspace(-4, -0.5, 30), cv=5, max_iter=10000)
ridge = RidgeCV(alphas=np.logspace(-4, 4, 100))

# Stacking model
stack = StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('lgb', lgb),
        ('lasso', lasso),
        ('ridge', ridge)
    ],
    final_estimator=RidgeCV(),
    cv=5,
    n_jobs=-1
)

# Train/test split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
stack.fit(X_train, y_train)

# Validation
val_preds = stack.predict(X_val)
val_score = np.sqrt(mean_squared_log_error(y_val, val_preds))
print(f"Validation RMSLE: {val_score:.5f}")

# Final prediction
final_preds = np.expm1(stack.predict(X_test))

# Submission
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_preds
})

submission.to_csv('submission.csv', index=False)
print("Submission saved as submission.csv")
