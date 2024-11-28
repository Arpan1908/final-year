import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Extract daily new cases
daily_new_cases = np.diff(df['Confirmed'].values.astype(float))  # Daily new cases
X = np.arange(len(daily_new_cases)).reshape(-1, 1)  # Days
y = daily_new_cases.reshape(-1, 1)

# Train-test split and scaling
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1)
scx = StandardScaler()
scy = StandardScaler()
x_train = scx.fit_transform(x_train)
x_test = scx.transform(x_test)
y_train = scy.fit_transform(y_train)
y_test = scy.transform(y_test)

# Train the SVR model
regressor = SVR(kernel='rbf', epsilon=0.1)
regressor.fit(x_train, y_train.ravel())

# Inverse transform predictions for plotting
x_all_scaled = scx.transform(X)  # Scale all X data
y_pred_all = scy.inverse_transform(regressor.predict(x_all_scaled).reshape(-1, 1))  # Predict all Y data

# Set up real-time graph
fig, ax = plt.subplots()
ax.set_title('Real-Time COVID-19 Spread')
ax.set_xlabel('Days')
ax.set_ylabel('Daily New Cases')

# Initial plot elements
line_actual, = ax.plot([], [], 'magenta', label='Actual Data')
line_predicted, = ax.plot([], [], 'green', label='Predicted Data')
alert_points, = ax.plot([], [], 'ro', label='Pandemic Alert')

ax.legend()

# Real-time animation function
def update(frame):
    ax.clear()  # Clear the axes for each frame
    ax.set_title('Real-Time COVID-19 Spread')
    ax.set_xlabel('Days')
    ax.set_ylabel('Daily New Cases')

    # Slice data up to the current frame
    actual_x = X[:frame + 1].flatten()
    actual_y = y[:frame + 1].flatten()
    predicted_y = y_pred_all[:frame + 1].flatten()

    # Update plots
    ax.plot(actual_x, actual_y, 'magenta', label='Actual Data')
    ax.plot(actual_x, predicted_y, 'green', label='Predicted Data')

    # Check for pandemic alert (e.g., daily new cases > 200)
    alert_indices = np.where(actual_y > 200)[0]
    alert_x = actual_x[alert_indices]
    alert_y = actual_y[alert_indices]
    ax.plot(alert_x, alert_y, 'ro', label='Pandemic Alert')

    ax.legend()

# Real-time animation
ani = FuncAnimation(fig, update, frames=len(X), interval=500, repeat=False)

# Display the real-time graph
plt.show()
