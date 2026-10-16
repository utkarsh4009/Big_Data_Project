from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("FoodDelivery_NewDataset_Training") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n" + "=" * 70)
print("FOOD DELIVERY ML MODEL - NEW DATASET")
print("=" * 70)


# ============================================================
# 2. READ CSV FROM HDFS
# ============================================================

hdfs_path = "hdfs://localhost:9000/user/utk60/new_food_data/new_food_data.csv"

print("\nReading dataset from HDFS...")
print(hdfs_path)

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(hdfs_path)

print("\nDataset loaded successfully.")
print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))

print("\nOriginal Schema:")
df.printSchema()


# ============================================================
# 3. CLEAN STRING COLUMNS
# ============================================================

print("\nCleaning string columns...")

string_columns = [
    "ID",
    "Delivery_person_ID",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    "Time_taken(min)"
]

clean_df = df

for c in string_columns:
    clean_df = clean_df.withColumn(
        c,
        F.trim(F.col(c).cast("string"))
    )


# ============================================================
# 4. CLEAN COMMON INVALID STRING VALUES
# ============================================================

# Convert empty strings and NaN strings to NULL
for c in string_columns:
    clean_df = clean_df.withColumn(
        c,
        F.when(
            F.lower(F.trim(F.col(c))) == "nan",
            None
        ).when(
            F.trim(F.col(c)) == "",
            None
        ).otherwise(F.col(c))
    )


# ============================================================
# 5. CONVERT TARGET TO NUMERIC
# ============================================================

print("\nConverting target column...")

clean_df = clean_df.withColumn(
    "Delivery_Time_Minutes",
    F.regexp_extract(
        F.col("Time_taken(min)"),
        r"(\d+)",
        1
    ).cast("double")
)


# ============================================================
# 6. CONVERT NUMERIC COLUMNS SAFELY
# ============================================================

print("\nCleaning numeric columns...")

numeric_columns = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries"
]

for c in numeric_columns:

    # Cast to double
    clean_df = clean_df.withColumn(
        c,
        F.col(c).cast("double")
    )

    # Replace NaN with NULL
    clean_df = clean_df.withColumn(
        c,
        F.nanvl(F.col(c), F.lit(None).cast("double"))
    )


# ============================================================
# 7. REMOVE INVALID TARGET RECORDS
# ============================================================

print("\nRemoving invalid target records...")

clean_df = clean_df.filter(
    F.col("Delivery_Time_Minutes").isNotNull()
)

print(
    "Rows after target cleaning:",
    clean_df.count()
)


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

print("\nPerforming feature engineering...")


# ------------------------------------------------------------
# 8.1 ORDER HOUR
# ------------------------------------------------------------

# Instead of converting NaN into a timestamp,
# extract the hour directly from the time string.

clean_df = clean_df.withColumn(
    "Order_Hour",
    F.when(
        F.col("Time_Orderd").rlike(r"^\d{1,2}:\d{2}(:\d{2})?$"),
        F.regexp_extract(
            F.col("Time_Orderd"),
            r"^(\d{1,2}):",
            1
        ).cast("double")
    ).otherwise(None)
)


# ------------------------------------------------------------
# 8.2 DELIVERY DISTANCE USING HAVERSINE FORMULA
# ------------------------------------------------------------

lat1 = F.radians(
    F.col("Restaurant_latitude")
)

lon1 = F.radians(
    F.col("Restaurant_longitude")
)

lat2 = F.radians(
    F.col("Delivery_location_latitude")
)

lon2 = F.radians(
    F.col("Delivery_location_longitude")
)

a = (
    F.pow(
        F.sin((lat2 - lat1) / 2),
        2
    )
    +
    F.cos(lat1)
    *
    F.cos(lat2)
    *
    F.pow(
        F.sin((lon2 - lon1) / 2),
        2
    )
)

# Prevent numerical problems where a becomes slightly > 1
a = F.least(
    F.lit(1.0),
    F.greatest(F.lit(0.0), a)
)

distance = (
    2
    * F.atan2(
        F.sqrt(a),
        F.sqrt(1 - a)
    )
    * 6371
)

clean_df = clean_df.withColumn(
    "Delivery_Distance_km",
    distance
)


# ------------------------------------------------------------
# 8.3 ORDER DATE
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "Order_Date_Parsed",
    F.when(
        F.col("Order_Date").rlike(r"^\d{1,2}-\d{1,2}-\d{4}$"),
        F.to_date(
            F.col("Order_Date"),
            "dd-MM-yyyy"
        )
    ).otherwise(None)
)


# ------------------------------------------------------------
# 8.4 WEEKEND FEATURE
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "Is_Weekend",
    F.when(
        F.dayofweek(
            F.col("Order_Date_Parsed")
        ).isin([1, 7]),
        1
    ).otherwise(0)
)


# ------------------------------------------------------------
# 8.5 TRAFFIC SCORE
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "Traffic_Score",
    F.when(
        F.lower(
            F.col("Road_traffic_density")
        ) == "low",
        1
    )
    .when(
        F.lower(
            F.col("Road_traffic_density")
        ) == "medium",
        2
    )
    .when(
        F.lower(
            F.col("Road_traffic_density")
        ) == "high",
        3
    )
    .when(
        F.lower(
            F.col("Road_traffic_density")
        ) == "jam",
        4
    )
    .otherwise(0)
)


# ------------------------------------------------------------
# 8.6 FESTIVAL FEATURE
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "Is_Festival",
    F.when(
        F.lower(
            F.col("Festival")
        ) == "yes",
        1
    ).otherwise(0)
)


print("\nFeature engineering completed.")


# ============================================================
# 9. SELECT ML FEATURES
# ============================================================

ml_df = clean_df.select(

    # TARGET
    "Delivery_Time_Minutes",

    # NUMERIC FEATURES
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",

    # ENGINEERED FEATURES
    "Delivery_Distance_km",
    "Order_Hour",
    "Is_Weekend",
    "Traffic_Score",
    "Is_Festival",

    # CATEGORICAL FEATURES
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "City"
)


# ============================================================
# 10. REMOVE NULL AND NaN VALUES
# ============================================================

print("\nRemoving missing and invalid ML records...")

ml_df = ml_df.dropna()

# Additional NaN protection
for c in [
    "Delivery_Time_Minutes",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Delivery_Distance_km",
    "Order_Hour"
]:
    ml_df = ml_df.filter(
        ~F.isnan(F.col(c))
    )


print(
    "\nRows available for ML:",
    ml_df.count()
)

print("\nML Dataset Schema:")
ml_df.printSchema()


# ============================================================
# 11. SHOW BASIC DATA CHECK
# ============================================================

print("\nSample cleaned data:")

ml_df.select(
    "Delivery_Time_Minutes",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Delivery_Distance_km",
    "Order_Hour",
    "Traffic_Score",
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_vehicle",
    "City"
).show(10, truncate=False)


# ============================================================
# 12. CATEGORICAL FEATURE ENCODING
# ============================================================

categorical_columns = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "City"
]

indexers = []

for c in categorical_columns:

    indexers.append(
        StringIndexer(
            inputCol=c,
            outputCol=c + "_Index",
            handleInvalid="keep"
        )
    )


encoder = OneHotEncoder(
    inputCols=[
        c + "_Index"
        for c in categorical_columns
    ],
    outputCols=[
        c + "_Encoded"
        for c in categorical_columns
    ]
)


# ============================================================
# 13. VECTOR ASSEMBLER
# ============================================================

numeric_features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Delivery_Distance_km",
    "Order_Hour",
    "Is_Weekend",
    "Traffic_Score",
    "Is_Festival"
]

encoded_features = [
    c + "_Encoded"
    for c in categorical_columns
]

assembler = VectorAssembler(
    inputCols=numeric_features + encoded_features,
    outputCol="features",
    handleInvalid="skip"
)


# ============================================================
# 14. PREPROCESSING PIPELINE
# ============================================================

preprocessing_pipeline = Pipeline(
    stages=indexers + [
        encoder,
        assembler
    ]
)

print("\nFitting preprocessing pipeline...")

preprocessing_model = preprocessing_pipeline.fit(
    ml_df
)

prepared_df = preprocessing_model.transform(
    ml_df
)

print("Preprocessing completed.")


# ============================================================
# 15. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = prepared_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print(
    "\nTraining rows:",
    train_df.count()
)

print(
    "Testing rows :",
    test_df.count()
)


# ============================================================
# 16. EVALUATORS
# ============================================================

rmse_evaluator = RegressionEvaluator(
    labelCol="Delivery_Time_Minutes",
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol="Delivery_Time_Minutes",
    predictionCol="prediction",
    metricName="mae"
)

r2_evaluator = RegressionEvaluator(
    labelCol="Delivery_Time_Minutes",
    predictionCol="prediction",
    metricName="r2"
)


# ============================================================
# 17. BASELINE MODEL
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL")
print("=" * 70)

mean_delivery_time = train_df.select(
    F.avg("Delivery_Time_Minutes")
).first()[0]

print(
    "Training-set average delivery time:",
    round(mean_delivery_time, 4)
)

baseline_predictions = test_df.withColumn(
    "prediction",
    F.lit(mean_delivery_time)
)

baseline_rmse = rmse_evaluator.evaluate(
    baseline_predictions
)

baseline_mae = mae_evaluator.evaluate(
    baseline_predictions
)

baseline_r2 = r2_evaluator.evaluate(
    baseline_predictions
)

print(
    "Baseline RMSE:",
    round(baseline_rmse, 4)
)

print(
    "Baseline MAE :",
    round(baseline_mae, 4)
)

print(
    "Baseline R2  :",
    round(baseline_r2, 4)
)


# ============================================================
# 18. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("LINEAR REGRESSION")
print("=" * 70)

lr = LinearRegression(
    featuresCol="features",
    labelCol="Delivery_Time_Minutes"
)

lr_model = lr.fit(train_df)

lr_predictions = lr_model.transform(
    test_df
)

lr_rmse = rmse_evaluator.evaluate(
    lr_predictions
)

lr_mae = mae_evaluator.evaluate(
    lr_predictions
)

lr_r2 = r2_evaluator.evaluate(
    lr_predictions
)

print(
    "Linear Regression RMSE:",
    round(lr_rmse, 4)
)

print(
    "Linear Regression MAE :",
    round(lr_mae, 4)
)

print(
    "Linear Regression R2  :",
    round(lr_r2, 4)
)


# ============================================================
# 19. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST")
print("=" * 70)

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="Delivery_Time_Minutes",
    numTrees=100,
    maxDepth=8,
    seed=42
)

rf_model = rf.fit(train_df)

rf_predictions = rf_model.transform(
    test_df
)

rf_rmse = rmse_evaluator.evaluate(
    rf_predictions
)

rf_mae = mae_evaluator.evaluate(
    rf_predictions
)

rf_r2 = r2_evaluator.evaluate(
    rf_predictions
)

print(
    "Random Forest RMSE:",
    round(rf_rmse, 4)
)

print(
    "Random Forest MAE :",
    round(rf_mae, 4)
)

print(
    "Random Forest R2  :",
    round(rf_r2, 4)
)


# ============================================================
# 20. GRADIENT BOOSTED TREES
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTED TREES")
print("=" * 70)

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="Delivery_Time_Minutes",
    maxIter=50,
    maxDepth=5,
    seed=42
)

gbt_model = gbt.fit(train_df)

gbt_predictions = gbt_model.transform(
    test_df
)

gbt_rmse = rmse_evaluator.evaluate(
    gbt_predictions
)

gbt_mae = mae_evaluator.evaluate(
    gbt_predictions
)

gbt_r2 = r2_evaluator.evaluate(
    gbt_predictions
)

print(
    "GBT RMSE:",
    round(gbt_rmse, 4)
)

print(
    "GBT MAE :",
    round(gbt_mae, 4)
)

print(
    "GBT R2  :",
    round(gbt_r2, 4)
)


# ============================================================
# 21. FINAL MODEL COMPARISON
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':30s}"
    f"{'RMSE':>12s}"
    f"{'MAE':>12s}"
    f"{'R2':>12s}"
)

print("-" * 66)

print(
    f"{'Gradient Boosted Trees':30s}"
    f"{gbt_rmse:12.4f}"
    f"{gbt_mae:12.4f}"
    f"{gbt_r2:12.4f}"
)

print(
    f"{'Linear Regression':30s}"
    f"{lr_rmse:12.4f}"
    f"{lr_mae:12.4f}"
    f"{lr_r2:12.4f}"
)

print(
    f"{'Random Forest':30s}"
    f"{rf_rmse:12.4f}"
    f"{rf_mae:12.4f}"
    f"{rf_r2:12.4f}"
)

print(
    f"{'Baseline':30s}"
    f"{baseline_rmse:12.4f}"
    f"{baseline_mae:12.4f}"
    f"{baseline_r2:12.4f}"
)


# ============================================================
# 22. SAMPLE GBT PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE GBT PREDICTIONS")
print("=" * 70)

gbt_predictions.select(
    "Delivery_Time_Minutes",
    "prediction"
).show(10)


# ============================================================
# 23. STOP SPARK
# ============================================================

spark.stop()

print("\nTraining completed successfully.")