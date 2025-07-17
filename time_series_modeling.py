# time_series_modeling.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from utils.data_loader import load_data

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Load and merge datasets
features, sales, stores = load_data()
df = sales.merge(features, on=["Store", "Date", "IsHoliday"])
df = df.merge(stores, on="Store")
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

# Focus on a specific Store and Department
store_id = 1
dept_id = 1
store_dept_df = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)]
store_dept_df = store_dept_df.set_index("Date").sort_index()
weekly_sales = store_dept_df["Weekly_Sales"].resample("W").sum()

# Train-Test Split
train = weekly_sales[:-12]
test = weekly_sales[-12:]

# SARIMAX Modeling
model = sm.tsa.statespace.SARIMAX(train,
                                  order=(1, 1, 1),
                                  seasonal_order=(1, 1, 1, 52),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
results = model.fit()

# Forecast
forecast = results.get_forecast(steps=12)
forecast_ci = forecast.conf_int()

# ✅ Clip negative predictions
forecast_mean = forecast.predicted_mean.clip(lower=0)

# Plot full forecast
plt.figure(figsize=(12, 6))
plt.plot(train, label="Train")
plt.plot(test, label="Actual", color="orange")
plt.plot(forecast_mean, label="Forecast", color="red")
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0].clip(lower=0),
                 forecast_ci.iloc[:, 1].clip(lower=0),
                 color='pink', alpha=0.3)
plt.title(f"SARIMAX Forecast for Store {store_id}, Dept {dept_id}")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/sarimax_forecast.png")
plt.show()

# Plot zoomed forecast
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label="Actual", color='orange')
plt.plot(test.index, forecast_mean, label="Forecast", color='red')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0].clip(lower=0),
                 forecast_ci.iloc[:, 1].clip(lower=0),
                 color='pink', alpha=0.3)
plt.title(f"Zoomed SARIMAX Forecast – Store {store_id}, Dept {dept_id}")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/sarimax_forecast_zoomed.png")
plt.show()

# Evaluation Metrics
mae = mean_absolute_error(test, forecast_mean)
mse = mean_squared_error(test, forecast_mean)
rmse = np.sqrt(mse)
r2 = r2_score(test, forecast_mean)

print("\nSARIMAX Forecast Metrics:")
print(f"MAE:  {mae:,.2f}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")