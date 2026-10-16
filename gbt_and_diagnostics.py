from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("DeliveryDurationDiagnostics") \
    .getOrCreate()

print("\n==============================================")
print(" FOOD DELIVERY - GBT & FEATURE DIAGNOSTICS")
print("==============================================\n")


# ============================================================
# 2. LOAD DATA FROM HDFS
# ============================================================

print("Loading dataset from HDFS...")

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://localhost:9000/user/utk60/new_food_data/new_food_data.csv")

print("Dataset loaded successfully.")
print("Total records:", df.count())


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\nCreating engineered features...")

feature_base_df = df.withColumn(
    "Traffic_Score",
    when(col("Traffic_Level") == "Low", 1)
    .when(col("Traffic_Level") == "Medium", 2)
    .when(col("Traffic_Level") == "High", 3)
    .otherwise(0)
).withColumn(
    "Price_Per_Item",
    col("Total_Price") / col("Quantity")
)

feature_base_df = feature_base_df.withColumn(
    "Order_Hour",
    F.hour("Order_Time")
).withColumn(
    "Order_Day",
    F.dayofweek("Order_Time")
).withColumn(
    "Order_Month",
    F.month("Order_Time")
).withColumn(
    "Is_Weekend",
    when(
        F.dayofweek("Order_Time").isin([1, 7]),
        1
    ).otherwise(0)
)

print("Feature engineering completed.")


# ============================================================
# 4. FEATURE-TARGET RELATIONSHIP DIAGNOSTICS
# ============================================================

numeric_cols = [
    "Quantity",
    "Total_Price",
    "Delivery_Distance_km",
    "Traffic_Score",
    "Order_Hour",
    "Order_Day",
    "Order_Month",
    "Is_Weekend",
    "Price_Per_Item"
]

print("\n==============================================")
print(" CORRELATION WITH DELIVERY DURATION")
print("==============================================")

for c in numeric_cols:

    stats = feature_base_df.select(
        F.min(c).alias("min_value"),
        F.max(c).alias("max_value")
    ).first()

    if stats["min_value"] == stats["max_value"]:
        print(f"{c:30s} corr = N/A (constant feature)")
    else:
        corr = feature_base_df.stat.corr(
            c,
            "Delivery_Duration_Minutes"
        )

        print(f"{c:30s} corr = {corr:.4f}")


# ============================================================
# 5. CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n==============================================")
print(" MEAN DELIVERY DURATION BY CATEGORY")
print("==============================================")

categorical_cols = [
    "City",
    "Driver_Vehicle",
    "Driver_Availability",
    "Traffic_Level"
]

for c in categorical_cols:

    print(f"\n--- {c} ---")

    feature_base_df.groupBy(c).agg(
        F.round(
            F.avg("Delivery_Duration_Minutes"),
            2
        ).alias("Average_Duration"),
        F.count("*").alias("Order_Count")
    ).orderBy("Average_Duration").show(
        20,
        truncate=False
    )


# ============================================================
# 6. OVERALL MEAN BASELINE
# ============================================================

overall_mean = feature_base_df.select(
    F.avg("Delivery_Duration_Minutes")
).first()[0]

print(
    f"\nOverall Mean Delivery Duration: "
    f"{overall_mean:.2f} minutes"
)


# ============================================================
# 7. PREPARE ML FEATURES
# ============================================================

print("\n==============================================")
print(" PREPARING ML FEATURES")
print("==============================================")

categorical_cols = [
    "City",
    "Driver_Vehicle",
    "Driver_Availability"
]

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(
        inputCol=c + "_idx",
        outputCol=c + "_ohe"
    )
    for c in categorical_cols
]

feature_cols = numeric_cols + [
    c + "_ohe"
    for c in categorical_cols
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

pipeline = Pipeline(
    stages=indexers + encoders + [assembler]
)

prepared_df = pipeline.fit(
    feature_base_df
).transform(
    feature_base_df
)

print("ML feature preparation completed.")


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = prepared_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\nTraining records:", train_df.count())
print("Testing records :", test_df.count())


# ============================================================
# 9. EVALUATORS
# ============================================================

evaluator_rmse = RegressionEvaluator(
    labelCol="Delivery_Duration_Minutes",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="Delivery_Duration_Minutes",
    predictionCol="prediction",
    metricName="mae"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="Delivery_Duration_Minutes",
    predictionCol="prediction",
    metricName="r2"
)


def evaluate(model_name, predictions):

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    print(f"\n{model_name}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  R2   = {r2:.4f}")

    return rmse, mae, r2


# ============================================================
# 10. GRADIENT BOOSTED TREES
# ============================================================

print("\n==============================================")
print(" TRAINING GRADIENT BOOSTED TREES")
print("==============================================")

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="Delivery_Duration_Minutes",
    maxIter=50,
    maxDepth=5,
    seed=42
)

gbt_model = gbt.fit(train_df)

gbt_preds = gbt_model.transform(test_df)

gbt_results = evaluate(
    "Gradient Boosted Trees",
    gbt_preds
)


# ============================================================
# 11. GBT FEATURE IMPORTANCE
# ============================================================

print("\n==============================================")
print(" GBT FEATURE IMPORTANCES")
print("==============================================")

importance_values = (
    gbt_model
    .featureImportances
    .toArray()
)

importance_pairs = sorted(
    zip(feature_cols, importance_values),
    key=lambda x: -x[1]
)

for name, score in importance_pairs:

    print(
        f"{name:30s} {score:.4f}"
    )


# ============================================================
# 12. LINEAR REGRESSION
# ============================================================

print("\n==============================================")
print(" TRAINING LINEAR REGRESSION")
print("==============================================")

lr = LinearRegression(
    featuresCol="features",
    labelCol="Delivery_Duration_Minutes"
)

lr_model = lr.fit(train_df)

lr_preds = lr_model.transform(test_df)

lr_results = evaluate(
    "Linear Regression",
    lr_preds
)


# ============================================================
# 13. RANDOM FOREST
# ============================================================

print("\n==============================================")
print(" TRAINING RANDOM FOREST")
print("==============================================")

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="Delivery_Duration_Minutes",
    numTrees=100,
    maxDepth=8,
    seed=42
)

rf_model = rf.fit(train_df)

rf_preds = rf_model.transform(test_df)

rf_results = evaluate(
    "Random Forest",
    rf_preds
)


# ============================================================
# 14. BASELINE MODEL
# ============================================================

print("\n==============================================")
print(" BASELINE MODEL")
print("==============================================")

baseline_preds = test_df.withColumn(
    "prediction",
    F.lit(overall_mean)
)

baseline_results = evaluate(
    "Baseline (Predict Overall Mean)",
    baseline_preds
)


# ============================================================
# 15. FINAL MODEL COMPARISON
# ============================================================

print("\n==============================================")
print(" FINAL MODEL COMPARISON")
print("==============================================")

print(
    f"{'Model':30s}"
    f"{'RMSE':>12s}"
    f"{'MAE':>12s}"
    f"{'R2':>12s}"
)

print("-" * 66)

print(
    f"{'Gradient Boosted Trees':30s}"
    f"{gbt_results[0]:12.4f}"
    f"{gbt_results[1]:12.4f}"
    f"{gbt_results[2]:12.4f}"
)

print(
    f"{'Linear Regression':30s}"
    f"{lr_results[0]:12.4f}"
    f"{lr_results[1]:12.4f}"
    f"{lr_results[2]:12.4f}"
)

print(
    f"{'Random Forest':30s}"
    f"{rf_results[0]:12.4f}"
    f"{rf_results[1]:12.4f}"
    f"{rf_results[2]:12.4f}"
)

print(
    f"{'Baseline':30s}"
    f"{baseline_results[0]:12.4f}"
    f"{baseline_results[1]:12.4f}"
    f"{baseline_results[2]:12.4f}"
)


# ============================================================
# 16. SAMPLE PREDICTIONS
# ============================================================

print("\n==============================================")
print(" SAMPLE GBT PREDICTIONS")
print("==============================================")

gbt_preds.select(
    "Delivery_Duration_Minutes",
    "prediction"
).show(
    10,
    truncate=False
)


# ============================================================
# 17. END
# ============================================================

print("\n==============================================")
print(" ANALYSIS COMPLETED SUCCESSFULLY")
print("==============================================")

spark.stop()