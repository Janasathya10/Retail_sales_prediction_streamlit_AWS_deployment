import pandas as pd

def preprocess_data(features, sales, stores):
    # Convert Date column
    features['Date'] = pd.to_datetime(features['Date'])
    sales['Date'] = pd.to_datetime(sales['Date'])

    # Merge datasets
    df = pd.merge(sales, features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = pd.merge(df, stores, on='Store', how='left')

    # Fill NA
    df.fillna(0, inplace=True)

    return df