# 🚚 Big Data Analytics & Food Delivery Time Prediction

An end-to-end Big Data Analytics and Machine Learning project that combines **Apache Hadoop, HDFS, Apache Spark, PySpark, Spark SQL, and Spark MLlib** to analyze food delivery operations and predict delivery duration.

The project integrates distributed data processing, regression model training, hyperparameter tuning, interactive analytics, and explainable predictions in a single workflow.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Apache-Spark-orange?logo=apachespark" alt="Apache Spark">
  <img src="https://img.shields.io/badge/Apache-Hadoop-yellow?logo=apachehadoop" alt="Apache Hadoop">
  <img src="https://img.shields.io/badge/ML-GBT%20Regression-green" alt="GBT Regression">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit" alt="Streamlit">
</p>

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Dataset](#-dataset)
- [Data Processing and Feature Engineering](#-data-processing-and-feature-engineering)
- [Machine Learning Models](#-machine-learning-models)
- [Model Performance](#-model-performance)
- [Interactive Dashboard](#-interactive-dashboard)
- [Repository Structure](#-repository-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Project](#-running-the-project)
- [Future Enhancements](#-future-enhancements)
- [Team](#-team)

## 🎯 Project Overview

Food delivery platforms generate data about orders, delivery personnel, restaurants, geographical locations, traffic, weather, and delivery durations. Analyzing these factors can help understand operational performance and estimate delivery times more effectively.

This project develops a Big Data pipeline that processes food delivery records, engineers predictive features, trains regression models, and provides an interactive dashboard for analytical exploration and delivery-time prediction.

The system supports two major objectives:

1. **Big Data Analytics:** Explore delivery operations and compare delivery duration across traffic conditions, weather, vehicle types, cities, order types, distances, and festival conditions.
2. **Machine Learning:** Predict delivery duration for new orders using a tuned Gradient Boosted Trees regression model.

## ✨ Key Features

- **Distributed storage:** Store and access food delivery data through Hadoop Distributed File System (HDFS).
- **Big Data processing:** Process datasets using Apache Spark and PySpark.
- **Data cleaning:** Handle missing values, invalid records, inconsistent strings, and data-type conversions.
- **Feature engineering:** Derive delivery distance, order hour, weekend indicators, traffic scores, and festival indicators.
- **Regression model comparison:** Evaluate Linear Regression, Random Forest Regression, and Gradient Boosted Trees.
- **Hyperparameter optimization:** Tune GBT parameters to improve predictive performance.
- **Spark SQL analytics:** Analyze delivery trends and operational metrics using SQL queries.
- **Interactive dashboard:** Explore analytics and submit new delivery scenarios through a Streamlit interface.
- **Delivery-time prediction:** Estimate delivery duration in minutes for user-specified delivery conditions.
- **Explainable predictions:** Use approximate SHAP-based feature contributions to help interpret individual predictions when the optional SHAP dependency is installed.
- **Model diagnostics:** Evaluate prediction errors and inspect model performance.

## 🏗️ System Architecture

```text
              Food Delivery Dataset
                       |
                       v
              Hadoop / HDFS
              Distributed Storage
                       |
                       v
             Apache Spark / PySpark
                       |
                       v
              Data Preprocessing
                       |
                       v
              Feature Engineering
                       |
                       v
              Feature Encoding
                       |
                       v
               Train/Test Split
                       |
                       v
               Regression Models
             /         |          \
            v          v           v
        Linear      Random       Gradient
       Regression   Forest       Boosted Trees
             \         |          /
                       v
                Model Evaluation
                  RMSE / MAE / R²
                       |
                       v
                GBT Hyperparameter
                     Tuning
                       |
                       v
                 Tuned GBT Model
                       |
              +--------+---------+
              |                  |
              v                  v
         Spark SQL          New-Order
          Analytics         Prediction
              |                  |
              +--------+---------+
                       |
                       v
             Streamlit Dashboard
                       |
                       v
            Tables, Charts and Model
                  Explanations
```

## 🛠️ Technology Stack

| Technology | Role |
|---|---|
| Python | Core programming language |
| Apache Hadoop | Big Data infrastructure |
| HDFS | Distributed dataset storage |
| Apache Spark | Distributed data processing |
| PySpark | Python interface for Spark |
| Spark SQL | Analytical queries and data exploration |
| Spark MLlib | Regression models and ML pipelines |
| StringIndexer | Encoding categorical variables |
| OneHotEncoder | Categorical feature transformation |
| VectorAssembler | Combining features into model vectors |
| RegressionEvaluator | Calculating regression metrics |
| Streamlit | Interactive web dashboard |
| Pandas and NumPy | Tabular data handling and numerical operations |
| SHAP | Approximate feature-contribution explanations |

## 📂 Dataset

The project works with food delivery data containing order, delivery, geographical, and operational attributes.

The dataset includes fields such as:

- Delivery personnel age and ratings
- Restaurant and delivery coordinates
- Order date and time
- Weather conditions
- Road traffic density
- Order type and vehicle type
- Vehicle condition
- Number of multiple deliveries
- Festival information
- City
- Actual delivery duration

The original delivery-time column, `Time_taken(min)`, is processed to produce the numerical target `Delivery_Time_Minutes`.

The repository contains `new_food_data.csv` and `talabat_enhanced_orders.csv`. The training and dashboard scripts reference an HDFS dataset path, so verify which dataset and schema each workflow expects before running it.

Dataset source and licensing details should be checked against the original dataset provider before redistribution.

## ⚙️ Data Processing and Feature Engineering

### 1. Data Ingestion

The Spark workflows read CSV data from HDFS and load it into Spark DataFrames for processing and analysis.

### 2. Data Cleaning

The preprocessing pipeline includes:

- Trimming and standardizing string values
- Handling empty strings and textual `NaN` values
- Converting numerical fields to appropriate data types
- Extracting numerical delivery-time targets
- Filtering records with invalid targets
- Preparing categorical attributes for encoding

### 3. Feature Engineering

The pipeline derives features that describe delivery conditions.

| Feature | Description |
|---|---|
| `Delivery_Distance_km` | Approximate straight-line distance between restaurant and delivery coordinates |
| `Order_Hour` | Hour extracted from the order time |
| `Is_Weekend` | Weekend indicator derived from the order date |
| `Traffic_Score` | Numerical encoding of traffic density |
| `Is_Festival` | Indicator representing festival conditions |

Additional predictors include delivery-person characteristics, vehicle condition, multiple deliveries, weather, traffic, order type, vehicle type, and city.

### 4. Feature Encoding

Spark MLlib transformations convert categorical values into model-compatible representations and assemble the selected features into a vector.

## 🤖 Machine Learning Models

The project implements three regression algorithms and a baseline comparison.

1. **Linear Regression** — establishes a linear predictive relationship between the features and delivery duration.
2. **Random Forest Regression** — combines multiple decision trees to model nonlinear relationships.
3. **Gradient Boosted Trees (GBT)** — builds an ensemble of trees sequentially to improve predictive performance.
4. **Baseline model** — provides a reference point for comparing learned models.

### Model Evaluation

The models are evaluated using:

- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors.
- **MAE (Mean Absolute Error):** Measures average absolute error in minutes.
- **R² (Coefficient of Determination):** Measures the proportion of target variance explained by the model.

Lower RMSE and MAE values and a higher R² score generally indicate better regression performance.

## 📊 Model Performance

The following values are reported in the project report.

| Metric | Initial GBT | Tuned GBT |
|---|---:|---:|
| RMSE | 4.0162 | 3.8722 |
| MAE | 3.1939 min | 3.0810 min |
| R² | 0.8168 | 0.8297 |

### Results

- The initial GBT model outperformed the other initially evaluated regression models.
- Hyperparameter tuning reduced RMSE from 4.0162 to 3.8722.
- MAE decreased from 3.1939 to 3.0810 minutes.
- R² improved from 0.8168 to 0.8297.

**Final reported result:** The tuned GBT model achieved an R² of 0.8297, explaining approximately 82.97% of the variation in delivery duration on the test dataset, with a mean absolute prediction error of approximately 3.08 minutes.

The tuning workflow explores parameters such as `maxDepth`, `maxIter`, and `stepSize`.

## 📈 Interactive Dashboard

The main application is implemented in `spark_sql_dashboard.py` using Streamlit, Spark SQL, and Spark MLlib.

### Dashboard capabilities

#### 1. Delivery Analytics

Explore summary metrics, cleaned records, and analytical results including:

- Average and median delivery duration
- Delivery-time distribution
- Delivery duration by traffic density
- Delivery duration by weather conditions
- City-level delivery comparisons
- Vehicle-type comparisons
- Vehicle condition analysis
- Order-type analysis
- Festival comparisons
- Delivery distance versus delivery duration

#### 2. Delivery-Time Prediction

Enter delivery-related information such as:

- Delivery person's age and rating
- Delivery distance
- Order hour
- Vehicle condition
- Number of multiple deliveries
- Weather and traffic conditions
- Order type and vehicle type
- City
- Weekend and festival indicators

The application uses the saved tuned GBT model to estimate delivery duration in minutes.

#### 3. Prediction Explainability

When SHAP is installed, the dashboard can display approximate feature contributions for an individual prediction. These contributions indicate which input features move the model's estimate above or below its baseline prediction.

SHAP explanations are approximate and should be interpreted as model explanations, not proof of causation.

## 📁 Repository Structure

```text
Big_Data_Project/
│
├── README.md
│
├── gbt_and_diagnostics.py
│   └── Model evaluation and diagnostics
│
├── train_new.py
│   └── Data preprocessing, feature engineering and model training
│
├── tune_gbt.py
│   └── GBT hyperparameter tuning
│
├── spark_sql_dashboard.py
│   └── Streamlit dashboard, Spark SQL analytics and predictions
│
├── saved_tuned_gbt_model/
│   └── Saved tuned GBT pipeline
│
├── saved_gbt_dashboard_model/
│   └── Additional saved model artifacts
│
├── new_food_data.csv
│   └── Food delivery dataset
│
├── talabat_enhanced_orders.csv
│   └── Additional delivery data
│
└── .gitignore
```

This structure describes the main files visible in the repository. The saved model directories contain additional artifacts and are represented here as directories rather than listing every internal file.

## 🚀 Installation and Setup

### Prerequisites

Install and configure:

- Python 3.8 or later, subject to dependency compatibility
- Java JDK compatible with your Spark and Hadoop versions
- Apache Hadoop and HDFS
- Apache Spark with PySpark
- Streamlit
- Pandas and NumPy
- SHAP for optional prediction explanations

Ensure `JAVA_HOME` and `SPARK_HOME` are configured appropriately for your operating system.

### 1. Clone the repository

```bash
git clone https://github.com/utkarsh4009/Big_Data_Project.git
cd Big_Data_Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Python dependencies

Install the libraries required by the scripts:

```bash
python -m pip install pyspark streamlit pandas numpy shap
```

Ensure the installed PySpark version is compatible with your Apache Spark and Java configuration. The exact dependency versions should be pinned after validating the project environment.

### 4. Configure HDFS

The application currently references the following HDFS location:

```text
hdfs://localhost:9000/user/utk60/new_food_data/new_food_data.csv
```

This path is configured in the project scripts. If your HDFS username, NameNode address, port, or dataset location differs, update the relevant paths in the scripts.

Start HDFS using your configured Hadoop installation, then verify that the dataset is accessible.

Example commands:

```bash
hdfs dfs -mkdir -p /user/utk60/new_food_data
hdfs dfs -put new_food_data.csv /user/utk60/new_food_data/
hdfs dfs -ls /user/utk60/new_food_data/
```

If the file already exists in HDFS, verify its location rather than uploading a duplicate.

## ▶️ Running the Project

Run commands from the repository root after configuring Hadoop, Spark, and the dataset path.

### 1. Train the model

```bash
python train_new.py
```

This script reads data from HDFS, preprocesses the records, engineers features, trains regression models, and evaluates their performance.

### 2. Tune the GBT model

```bash
python tune_gbt.py
```

Run the tuning workflow to optimize the GBT model and save the fitted pipeline to the location expected by the dashboard.

### 3. Run model diagnostics

```bash
python gbt_and_diagnostics.py
```

Use this script to inspect the project's model evaluation and diagnostics.

### 4. Launch the dashboard

After confirming that the required dataset and saved tuned model are available, run:

```bash
streamlit run spark_sql_dashboard.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to access the analytics and prediction interface.

**Important:** The dashboard expects the tuned model at `saved_tuned_gbt_model/`. Run the tuning workflow successfully first if that artifact is missing. The scripts also depend on the configured HDFS path and a compatible local Spark environment.

## 🔮 Future Enhancements

Potential directions for extending the project include:

- Integrating live order, traffic, and weather data
- Comparing additional machine learning algorithms
- Improving model monitoring and error analysis
- Adding richer visualizations and operational KPIs
- Deploying the prediction interface as a hosted application
- Improving model interpretability and prediction explanations
- Exploring distributed or streaming processing for continuously arriving delivery data


Developed by:
- Utkarsh Jaiswal

The project covers Big Data storage, data engineering, distributed processing, feature engineering, machine learning, model evaluation, analytical queries, and dashboard integration.


🏁 Conclusion

This project demonstrates how Big Data technologies and machine learning can be integrated to address a practical food delivery problem.

Hadoop HDFS provides the storage layer, Apache Spark supports distributed processing and analytics, and Spark MLlib enables regression model training and optimization. The Streamlit dashboard brings these capabilities together through interactive SQL analytics, delivery-time prediction, and approximate model explanations.

The tuned Gradient Boosted Trees model achieved an RMSE of 3.8722, MAE of 3.0810 minutes, and R² of 0.8297 in the reported evaluation.

The result is an end-to-end academic project combining data engineering, Big Data analytics, predictive modeling, and interactive application development.

---

<p align="center">
  <b>Big Data Analytics • Distributed Computing • Machine Learning • Food Delivery Intelligence</b>
</p>
