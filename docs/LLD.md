# Low-Level Design (LLD)

## 1. Overview

This document provides a detailed breakdown of the data pipeline, feature engineering, and ML workflow for delay prediction.



---

## 2. Data Pipeline

### 2.1 Raw Data Sources

- orders.csv  
- order_items.csv  
- customers.csv  
- sellers.csv  

---

### 2.2 Data Ingestion Layer

**Function:** `load_data()`

- Reads CSV files into DataFrames  
- Validates schema  
- Handles initial data sanity checks  

---

### 2.3 Preprocessing Layer

**Function:** `convert_dates()`

- Converts timestamp columns to datetime format  
- Handles invalid or missing timestamps  
- Standardizes date formats  

---

### 2.4 Feature Engineering Layer

**Functions:**

- `create_time_features()`
  - Delivery duration
  - Shipping delay
  - Day/week/month features  

- `create_order_features()`
  - Number of items per order
  - Freight value
  - Order-level aggregations  

- `create_target()`
  - Binary target:
    - `is_delayed = 1` if actual delivery > estimated delivery  
    - else `0`

---

### 2.5 Data Integration Layer

**Function:** `merge_all()`

- Joins all datasets:
  - Orders ↔ Order Items  
  - Orders ↔ Customers  
  - Orders ↔ Sellers  
- Ensures correct join keys  
- Maintains row-level consistency  

---

### 2.6 Feature Selection

**Function:** `final_selection()`

- Select relevant features  
- Drop redundant or leakage-prone columns  
- Ensure model-ready dataset  

---

### 2.7 Data Cleaning & Encoding

**Function:** `clean_data()`

- Handle missing values:
  - Imputation (mean/median/mode)  
- Encode categorical variables:
  - One-hot encoding / Label encoding  
- Normalize/scale if required  

---

### 2.8 Final Dataset

- Output file: `final_poc_dataset.csv`  
- Contains:
  - Engineered features  
  - Target variable  

---

## 3. Machine Learning Pipeline

### 3.1 Train/Test Split

- Split dataset into training and testing sets  
- Prefer time-based split to avoid leakage  

---

### 3.2 Model Training

Models used:
- Logistic Regression  
- Random Forest  
- XGBoost  

---

### 3.3 Model Evaluation

Metrics:
- Precision  
- Recall  
- Accuracy  
- F1 Score  
- ROC-AUC  

---

### 3.4 Model Artifact

- Saved model file:
  - `saved_model.pkl`  

---

## 4. Deployment Layer

### 4.1 Batch Prediction
- Scheduled jobs (daily predictions)  

### 4.2 Real-time API
- API endpoint for live predictions  

---

## 5. Prediction Output

- Output variable:
  - `is_delayed = 0 / 1`

---

## 6. Error Handling

- Missing data fallback strategies  
- Logging for failed pipeline steps  
- Validation checks at each stage  

---

## 7. Summary

The LLD provides a detailed, function-level blueprint for implementing the ML pipeline, ensuring reproducibility, scalability, and clarity.
