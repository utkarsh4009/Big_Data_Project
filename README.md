# Big Data Analytics & Delivery Time Prediction

A comprehensive Big Data analytics and machine learning pipeline leveraging **Apache Spark (PySpark)**, **Gradient Boosted Trees (GBT)**, and interactive dashboards to analyze food delivery operations and predict order delivery times.

## 📌 Project Overview
- **Spark SQL Dashboard (`spark_sql_dashboard.py`)**: End-to-end interactive reporting and analytics engine for food delivery metrics, order trends, restaurant operations, and customer analytics.
- **Model Training & Pipeline (`train_new.py`)**: Data ingestion, feature engineering, and training pipeline using PySpark MLlib.
- **Hyperparameter Optimization (`tune_gbt.py`)**: Systematic hyperparameter tuning for GBT Regression models to achieve optimal predictive accuracy.
- **Diagnostics & Evaluation (`gbt_and_diagnostics.py`)**: Model validation metrics, residual analysis, and performance diagnostics.

## 🛠️ Tech Stack
- **Frameworks**: Apache Spark, PySpark, PySpark MLlib
- **Languages**: Python
- **Analytics & Storage**: Spark SQL, Parquet/CSV pipelines

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Apache Spark & Java JDK configured (`SPARK_HOME` and `JAVA_HOME`)

### Running the Dashboard
```bash
python spark_sql_dashboard.py
```

### Training & Tuning the Model
```bash
python train_new.py
python tune_gbt.py
```
