# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVR
# from sklearn import metrics
# import matplotlib.pyplot as plt

# # Load the COVID-19 dataset
# df = pd.read_csv('covid_19_data.csv')

# # Display basic information about the dataset
# print("Dataset shape:", df.shape)
# print(df.head())

# # Ensure the 'Confirmed' column is numeric
# df['Confirmed'] = pd.to_numeric(df['Confirmed'], errors='coerce').fillna(0)

# # Sort the data by 'ObservationDate' to maintain temporal order
# df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
# df.sort_values(by='ObservationDate', inplace=True)

# # Extract the target variable (Confirmed cases)
# y = df['Confirmed'].values.astype(float)

# # Compute daily new cases as the difference between consecutive days
# y = np.diff(y)
# y = y.reshape(-1, 1)  # Ensure y is 2D

# # Create a corresponding feature variable (days since the start)
# X = np.arange(len(y)).reshape(-1, 1)

# # Split into train and test sets (80% train, 20% test)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# # Scale the data using standardization
# scx = StandardScaler()
# scy = StandardScaler()

# x_train = scx.fit_transform(x_train)
# x_test = scx.transform(x_test)
# y_train = scy.fit_transform(y_train)
# y_test = scy.transform(y_test)

# # Fit the SVR model
# regressor = SVR(kernel='rbf', epsilon=0.1)
# regressor.fit(x_train, y_train.ravel())  # Flatten y_train to 1D as required by SVR

# # Predict the test set
# y_pred = regressor.predict(x_test)

# # Calculate performance metrics
# mse = metrics.mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# print("Mean Squared Error:", mse)
# print("Root Mean Squared Error:", rmse)
# print("Model Score (R^2):", regressor.score(x_test, y_test))

# # Inverse transform the predictions and data
# y_pred = scy.inverse_transform(y_pred.reshape(-1, 1))  # Reshape to 2D
# x_test = scx.inverse_transform(x_test)

# # Plot the regression fit with original data
# plt.scatter(X, y, color='magenta', label='Original Data')
# plt.scatter(x_test, y_pred, color='green', label='Test Data')
# plt.title('COVID-19 Daily New Cases (SVR Model)')
# plt.xlabel('Days')
# plt.ylabel('Daily New Cases')
# plt.legend()
# plt.show()

# # Predict daily new cases for future days
# future_days = np.array([100, 110, 120]).reshape(-1, 1)  # Example future days
# future_days_scaled = scx.transform(future_days)
# future_predictions = scy.inverse_transform(regressor.predict(future_days_scaled))

# print("Predicted daily new cases for days 100, 110, 120:", future_predictions)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the COVID-19 dataset
df = pd.read_csv('demo.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print(df.head())

# Ensure the 'Confirmed' column is numeric
df['Confirmed'] = pd.to_numeric(df['Confirmed'], errors='coerce').fillna(0)

# Sort the data by 'ObservationDate' to maintain temporal order
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df.sort_values(by='ObservationDate', inplace=True)

# Extract the target variable (Confirmed cases)
y = df['Confirmed'].values.astype(float)

# Compute daily new cases as the difference between consecutive days
y = np.diff(y)
y = y.reshape(-1, 1)  # Ensure y is 2D

# Create a corresponding feature variable (days since the start)
X = np.arange(len(y)).reshape(-1, 1)

# Feature Engineering: Add Daily Growth Rate
daily_growth_rate = np.zeros_like(y)
daily_growth_rate[1:] = (y[1:] - y[:-1]) / (y[:-1] + 1e-5)  # Avoid division by zero
X = np.c_[X, daily_growth_rate]

# Add Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split into train and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.20, random_state=42)

# Scale the data using standardization
scx = StandardScaler()
scy = StandardScaler()

x_train = scx.fit_transform(x_train)
x_test = scx.transform(x_test)
y_train = scy.fit_transform(y_train)
y_test = scy.transform(y_test)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='r2', verbose=2)
grid_search.fit(x_train, y_train.ravel())  # Flatten y_train to 1D as required by SVR
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Fit the best model
best_model.fit(x_train, y_train.ravel())

# Cross-Validation Scores
cv_scores = cross_val_score(best_model, x_train, y_train.ravel(), cv=5, scoring='r2')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV R²:", np.mean(cv_scores))

# Predict the test set
y_pred = best_model.predict(x_test)

# Calculate performance metrics
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = best_model.score(x_test, y_test)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Model Score (R^2):", r2)

# Inverse transform the predictions and data
y_pred = scy.inverse_transform(y_pred.reshape(-1, 1))  # Reshape to 2D
x_test_original = scx.inverse_transform(x_test)[:, 0]  # Retrieve the first column (days)

# Plot the regression fit with original data
plt.scatter(X[:, 0], y, color='magenta', label='Original Data')
plt.scatter(x_test_original, y_pred, color='green', label='Test Predictions')
plt.title('COVID-19 Daily New Cases (Improved SVR Model)')
plt.xlabel('Days')
plt.ylabel('Daily New Cases')
plt.legend()
plt.show()

# Predict daily new cases for future days
future_days = np.array([100, 110, 120]).reshape(-1, 1)
future_days_poly = poly.transform(np.c_[future_days, np.zeros_like(future_days)])  # Add dummy feature
future_days_scaled = scx.transform(future_days_poly)
future_predictions = scy.inverse_transform(best_model.predict(future_days_scaled).reshape(-1, 1))

print("Predicted daily new cases for days 100, 110, 120:", future_predictions.ravel())
