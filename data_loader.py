import pandas as pd

def load_data():
    features = pd.read_csv('data/Features_data_set.csv')
    sales = pd.read_csv('data/sales_data_set.csv')
    stores = pd.read_csv('data/stores_data_set.csv')
    return features, sales, stores