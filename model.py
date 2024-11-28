import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import metrics
import matplotlib.pyplot as plt

# Generate random COVID-19 data
# np.random.seed(42)

# # Simulate 60 days of data
# days = 60
# dates = pd.date_range(start='01/01/2024', periods=days, freq='D')

# # Randomly generating data for Confirmed, Deaths, and Recovered
# confirmed = np.cumsum(np.random.randint(1, 100, size=days))  # Cumulative confirmed cases
# deaths = np.cumsum(np.random.randint(0, 10, size=days))  # Cumulative deaths
# recovered = np.cumsum(np.random.randint(0, 80, size=days))  # Cumulative recovered

# # Create a DataFrame
# data = {
#     'ObservationDate': dates,
#     'Province/State': np.random.choice(['Province1', 'Province2', 'Province3'], days),
#     'Country/Region': ['CountryA'] * days,
#     'Last Update': pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S'),
#     'Confirmed': confirmed,
#     'Deaths': deaths,
#     'Recovered': recovered
# }

# df = pd.DataFrame(data)


np.random.seed(42)

# Simulate data for multiple regions and states
num_rows = 10000
dates = pd.date_range(start='01/01/2024', periods=100, freq='D')
num_provinces = 20  # Number of unique provinces
num_countries = 10  # Number of unique countries

# Generate random data
provinces = [f'Province{i+1}' for i in range(num_provinces)]
countries = [f'Country{i+1}' for i in range(num_countries)]
all_dates = np.random.choice(dates, num_rows)

confirmed = np.cumsum(np.random.randint(1, 100, size=num_rows)) % 100000  # Wrap around at 100,000
deaths = np.cumsum(np.random.randint(0, 10, size=num_rows)) % 5000  # Wrap around at 5,000
recovered = np.cumsum(np.random.randint(0, 80, size=num_rows)) % 80000  # Wrap around at 80,000

# Build DataFrame
data = {
    'ObservationDate': all_dates,
    'Province/State': np.random.choice(provinces, num_rows),
    'Country/Region': np.random.choice(countries, num_rows),
    'Last Update': pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S'),
    'Confirmed': confirmed,
    'Deaths': deaths,
    'Recovered': recovered
}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

y = df['Confirmed'].values.astype(float)

# Create the daily number of cases (difference between consecutive days)
# Create the daily number of cases (difference between consecutive days)
y = np.diff(y)
y = y.reshape(-1, 1)  # Ensure y is a 2D array

# Adjust X to match the length of y (remove the first value of X)
X = np.arange(len(y)).reshape(-1, 1)  # Ensure X is also a 2D array

# Create train and test dataset (80% train, 20% test)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1)

# Scaling the dataset by standardization technique
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
scy = StandardScaler()

x_train = scx.fit_transform(x_train)
x_test = scx.transform(x_test)
y_train = scy.fit_transform(y_train)
y_test = scy.transform(y_test)

# Fit the SVR model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf', epsilon=0.1)
regressor.fit(x_train, y_train.ravel())  # Flatten y_train to 1D as required by SVR

# Predict the test set
y_pred = regressor.predict(x_test)

# Calculate the model performance parameters
from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Model Score (R^2):", regressor.score(x_test, y_test))

# Inverse transform the predictions and data
y_pred = scy.inverse_transform(y_pred.reshape(-1, 1))  # Reshape to 2D after prediction
x_test = scx.inverse_transform(x_test)

# Plot the regression fit with original data
import matplotlib.pyplot as plt
plt.scatter(X, y, color='magenta', label='Original Data')
plt.scatter(x_test, y_pred, color='green', label='Test Data')
plt.title('Covid19 (Support Vector Regression Model)')
plt.xlabel('Days')
plt.ylabel('Daily New Cases')
plt.legend()
plt.show()

# Predict future values (for example, for day 100)
future_day = 100
future_day_scaled = scx.transform([[future_day]])
y_pred_future = scy.inverse_transform(regressor.predict(future_day_scaled).reshape(-1, 1))
print(f"Predicted daily new cases for day {future_day}: {y_pred_future}")


# Predict daily new cases for an additional set of days
sample_days = np.array([70, 80, 90]).reshape(-1, 1)  # Example future days
sample_days_scaled = scx.transform(sample_days)
future_predictions = scy.inverse_transform(regressor.predict(sample_days_scaled))

print("Predicted daily new cases for days 70, 80, 90:", future_predictions)
