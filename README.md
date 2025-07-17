# Retail_sales_prediction_streamlit_AWS_deployment
Develop a predictive ANN model to forecast department-wide sales for each store over the next year and analyze the impact of markdowns on sales during holiday weeks. Provide actionable insights and recommendations to optimize markdown strategies and inventory management.

## 📁 Project Structure

Retail_Sales_Prediction_Streamlit_AWS_Deployment/
│
├── app/
│ └── streamlit_app.py # Streamlit UI for sales prediction
│
├── data/
│ ├── Features_data_set.csv # Store features & markdowns
│ ├── sales_data_set.csv # Historical sales data
│ └── stores_data_set.csv # Store metadata (type, size)
│
├── models/
│ ├── ann_model.h5 # Saved ANN model
│ ├── ann_model.keras # ANN model in Keras format
│ └── scaler.pkl # Scaler for ANN model input
│
├── notebooks/
│ └── eda_analysis.ipynb # Exploratory data analysis
│
├── plots/
│ ├── sarimax_forecast.png # SARIMAX result - full view
│ └── sarimax_forecast_zoomed.png # SARIMAX result - zoomed view
│
├── utils/
│ ├── data_loader.py # Loads and merges all datasets
│ ├── preprocessing.py # Feature engineering & cleanup
│ └── time_series_modeling.py # SARIMAX modeling per store
│
├── train_model.py # Trains and saves the ANN model
├── Insights & recommendations # Text insights derived from analysis
├── requirements.txt # Required Python packages

## Datasets

### `sales_data_set.csv`
Weekly sales per store and department.  
- `Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`

### `features_data_set.csv`
Extra features that impact sales.  
- Economic: `CPI`, `Unemployment`, `Fuel_Price`  
- Promotions: `MarkDown1`–`MarkDown5`  
- Weather: `Temperature`

### `stores_data_set.csv`
Describes each store’s type and size.  
- `Store`, `Type`, `Size`

## Modeling Approach

### ANN (Artificial Neural Network)
- Predicts weekly sales using historical data + features
- Trained using merged dataset of store, feature, and sales data

### SARIMAX (Time-Series Model)
- Models sales trends for a specific store over time
- Accounts for seasonality and holidays

## Deployment (AWS EC2 + Streamlit)

- Hosted on Ubuntu EC2 instance
- Streamlit app served via Nginx (reverse proxy)
- HTTPS enabled using Certbot and DuckDNS
- Access at: `https://retail-app.duckdns.org`

## Tech Stack

- **Python**, **Pandas**, **NumPy**, **Seaborn**, **Matplotlib**
- **scikit-learn**, **TensorFlow**, **Statsmodels**
- **Streamlit**, **AWS EC2**, **Nginx**, **Certbot**


## 🖼 App Demo Screenshot

![Retail Sales Forecast App](b4aa516b-50f3-4155-8539-cbde2bdfd47f.png)

🔗 **Live App URL**: [https://retail-app.duckdns.org](https://retail-app.duckdns.org)
