import pandas as pd

def load_data():
    df_customers = pd.read_csv('olist_customers_dataset.csv')
    df_geolocation = pd.read_csv('olist_geolocation_dataset.csv')
    df_items = pd.read_csv('olist_order_items_dataset.csv')
    df_order_pay = pd.read_csv('olist_order_payments_dataset.csv')
    df_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
    df_orders = pd.read_csv('olist_orders_dataset.csv')
    df_products = pd.read_csv('olist_products_dataset.csv')
    df_sellers = pd.read_csv('olist_sellers_dataset.csv')
    df_category = pd.read_csv('product_category_name_translation.csv')

    return {
        "customers": df_customers,
        "geolocation": df_geolocation,
        "items": df_items,
        "payments": df_order_pay,
        "reviews": df_reviews,
        "orders": df_orders,
        "products": df_products,
        "sellers": df_sellers,
        "category": df_category
    }
# Basic EDA Function

def basic_eda(df, name):

    print(f"\n Dataset: {name}")
    print("=" * 60)

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nSample Data:")
    print(df.sample(5))

    print("\nInfo:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicates:", df.duplicated().sum())

    print("\nUnique Values:")
    print(df.nunique())

    print("\nSummary Stats:")
    print(df.describe())

    print("\nTop Category Values:")
    for col in df.select_dtypes(include='object').columns:
        print(f"\n {col}")
        print(df[col].value_counts().head())


'''
profile = ProfileReport(df)
profile.to_file(f"{name}_report.html")
'''
# Run EDA

def run_eda():
    datasets = load_data()

    for name, df in datasets.items():
        basic_eda(df, name)

if __name__ == "__main__":
    run_eda()