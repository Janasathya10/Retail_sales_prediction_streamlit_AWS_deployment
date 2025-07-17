import pandas as pd
import numpy as np
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load & preprocess
features, sales, stores = load_data()
df = preprocess_data(features, sales, stores)

# Feature selection
X = df[['Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']]
y = df['Weekly_Sales']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ANN Model
model = Sequential([
    Dense(128, input_dim=X.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30,
          batch_size=128, callbacks=[EarlyStopping(patience=3)])

model.save("models/ann_model.keras")
joblib.dump(scaler, "models/scaler.pkl")