"""
Data Cleaning Pipeline for E-Commerce Dataset
Converted from data_cleaning.ipynb
"""

import pandas as pd
import numpy as np
import unicodedata

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


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Remove accented / non-ASCII characters from a string."""
    return unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('utf-8')


# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaner:
    """
    Loads and cleans all tables in the e-commerce dataset.

    Usage
    -----
    cleaner = DataCleaner()
    cleaner.run_all()

    # Access cleaned dataframes
    df_orders    = cleaner.df_orders
    df_customers = cleaner.df_customers
    ...
    """

    def __init__(self):
        print(" Loading raw data ...")
        data = load_data()

        self.df_customers = data['customers']
        self.df_orders    = data['orders']
        self.df_items     = data['items']
        self.df_payments  = data['payments']
        self.df_reviews   = data['reviews']
        self.df_products  = data['products']
        self.df_sellers   = data['sellers']
        self.df_geo       = data['geolocation']
        self.df_cat       = data['category']

        print(" Data loaded successfully.\n")

    # ── 1. Customers ─────────────────────────────────────────────────────────
    def clean_customers(self) -> pd.DataFrame:
        """
        - Lowercase + strip city names, remove accent characters
        - Uppercase + strip state codes
        - Cast zip code to string
        """
        df = self.df_customers.copy()

        df['customer_city'] = (
            df['customer_city']
            .str.lower()
            .str.strip()
            .apply(normalize_text)
        )
        df['customer_state'] = df['customer_state'].str.upper().str.strip()
        df['customer_zip_code_prefix'] = df['customer_zip_code_prefix'].astype(str)

        self.df_customers = df
        print(f"  ✔ Customers cleaned  →  {df.shape}")
        return df

    # ── 2. Geolocation ───────────────────────────────────────────────────────
    def clean_geolocation(self) -> pd.DataFrame:
        """
        - Cast zip code to string
        - Lowercase + strip + normalize city names
        - Uppercase + strip state codes
        - Aggregate duplicate zips (mean lat/lng, first city/state)
        """
        df = self.df_geo.copy()

        df['geolocation_zip_code_prefix'] = df['geolocation_zip_code_prefix'].astype(str)
        df['geolocation_city'] = (
            df['geolocation_city']
            .str.lower()
            .str.strip()
            .apply(normalize_text)
        )
        df['geolocation_state'] = df['geolocation_state'].str.upper().str.strip()

        df = df.groupby('geolocation_zip_code_prefix').agg(
            geolocation_lat=('geolocation_lat', 'mean'),
            geolocation_lng=('geolocation_lng', 'mean'),
            geolocation_city=('geolocation_city', 'first'),
            geolocation_state=('geolocation_state', 'first'),
        ).reset_index()

        self.df_geo = df
        print(f"  ✔ Geolocation cleaned  →  {df.shape}")
        return df

    # ── 3. Order Items ───────────────────────────────────────────────────────
    def clean_items(self) -> pd.DataFrame:
        """
        - Parse shipping_limit_date as datetime
        """
        df = self.df_items.copy()

        df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'])

        self.df_items = df
        print(f"  ✔ Order items cleaned  →  {df.shape}")
        return df

    # ── 4. Payments ──────────────────────────────────────────────────────────
    def clean_payments(self) -> pd.DataFrame:
        """
        - Lowercase + strip payment_type
        """
        df = self.df_payments.copy()

        df['payment_type'] = df['payment_type'].str.lower().str.strip()

        self.df_payments = df
        print(f"  ✔ Payments cleaned  →  {df.shape}")
        return df

    # ── 5. Orders ────────────────────────────────────────────────────────────
    def clean_orders(self) -> pd.DataFrame:
        """
        - Parse all date columns as datetime
        - Keep only 'delivered' orders
        - Drop rows with missing delivery date
        """
        df = self.df_orders.copy()

        date_cols = [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date',
        ]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])

        df = df[df['order_status'] == 'delivered']
        df = df.dropna(subset=['order_delivered_customer_date'])

        self.df_orders = df
        print(f"  ✔ Orders cleaned  →  {df.shape}")
        return df

    # ── 6. Products ──────────────────────────────────────────────────────────
    def clean_products(self) -> pd.DataFrame:
        """
        - Fill missing category with 'unknown'
        - Fill numeric columns (name/description length, photos) with median
        - Drop rows with missing physical dimensions / weight
        """
        df = self.df_products.copy()

        df['product_category_name'] = df['product_category_name'].fillna('unknown')

        for col in ['product_name_lenght', 'product_description_lenght', 'product_photos_qty']:
            df[col] = df[col].fillna(df[col].median())

        df = df.dropna(subset=[
            'product_weight_g',
            'product_length_cm',
            'product_height_cm',
            'product_width_cm',
        ])

        self.df_products = df
        print(f"  ✔ Products cleaned  →  {df.shape}")
        return df

    # ── 7. Sellers ───────────────────────────────────────────────────────────
    def clean_sellers(self) -> pd.DataFrame:
        """
        - Cast zip code to string
        - Lowercase + strip + normalize city names
        - Uppercase + strip state codes
        """
        df = self.df_sellers.copy()

        df['seller_zip_code_prefix'] = df['seller_zip_code_prefix'].astype(str)
        df['seller_city'] = (
            df['seller_city']
            .str.lower()
            .str.strip()
            .apply(normalize_text)
        )
        df['seller_state'] = df['seller_state'].str.upper().str.strip()

        self.df_sellers = df
        print(f"  ✔ Sellers cleaned  →  {df.shape}")
        return df

    # ── 8. Product Category ──────────────────────────────────────────────────
    def clean_category(self) -> pd.DataFrame:
        """
        - Lowercase + strip both category name columns
        """
        df = self.df_cat.copy()

        df['product_category_name'] = df['product_category_name'].str.lower().str.strip()
        df['product_category_name_english'] = (
            df['product_category_name_english'].str.lower().str.strip()
        )

        self.df_cat = df
        print(f"  ✔ Product category cleaned  →  {df.shape}")
        return df

    # ── 9. Reviews ───────────────────────────────────────────────────────────
    def clean_reviews(self) -> pd.DataFrame:
        """
        - Parse date columns as datetime
        - Drop free-text comment columns (title + message)
        """
        df = self.df_reviews.copy()

        df['review_creation_date']   = pd.to_datetime(df['review_creation_date'])
        df['review_answer_timestamp'] = pd.to_datetime(df['review_answer_timestamp'])

        df = df.drop(columns=['review_comment_title', 'review_comment_message'])

        self.df_reviews = df
        print(f"  ✔ Reviews cleaned  →  {df.shape}")
        return df

    # ── Master runner ────────────────────────────────────────────────────────
    def run_all(self) -> dict:
        """
        Run every cleaning step in the correct order and return
        a dict of all cleaned DataFrames.
        """
        print("=" * 50)
        print("  DATA CLEANING PIPELINE")
        print("=" * 50)

        self.clean_customers()
        self.clean_geolocation()
        self.clean_items()
        self.clean_payments()
        self.clean_orders()
        self.clean_products()
        self.clean_sellers()
        self.clean_category()
        self.clean_reviews()

        print("\n All tables cleaned successfully!")

        return {
            'customers':   self.df_customers,
            'geolocation': self.df_geo,
            'items':       self.df_items,
            'payments':    self.df_payments,
            'orders':      self.df_orders,
            'products':    self.df_products,
            'sellers':     self.df_sellers,
            'category':    self.df_cat,
            'reviews':     self.df_reviews,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaned = cleaner.run_all()

    # Quick sanity check
    for name, df in cleaned.items():
        print(f"  {name:15s}  →  {df.shape}")
