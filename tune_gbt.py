import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("FoodDelivery_GBT_Tuning") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n" + "=" * 75)
print("FOOD DELIVERY - GBT HYPERPARAMETER TUNING")
print("=" * 75)


# ============================================================
# 2. READ DATA FROM HDFS
# ============================================================

hdfs_path = (
    "hdfs://localhost:9000/user/utk60/"
    "new_food_data/new_food_data.csv"
)

print("\nReading dataset from HDFS...")
print(hdfs_path)

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(hdfs_path)

print("\nRows:", df.count())
print("Columns:", len(df.columns))


# ============================================================
# 3. CLEAN STRING COLUMNS
# ============================================================

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

    clean_df = clean_df.withColumn(
        c,
        F.when(
            F.lower(F.trim(F.col(c))) == "nan",
            None
        )
        .when(
            F.trim(F.col(c)) == "",
            None
        )
        .otherwise(F.col(c))
    )


# ============================================================
# 4. TARGET
# ============================================================

clean_df = clean_df.withColumn(
    "Delivery_Time_Minutes",
    F.regexp_extract(
        F.col("Time_taken(min)"),
        r"(\d+)",
        1
    ).cast("double")
)


# ============================================================
# 5. NUMERIC COLUMNS
# ============================================================

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

    clean_df = clean_df.withColumn(
        c,
        F.col(c).cast("double")
    )

    clean_df = clean_df.withColumn(
        c,
        F.nanvl(
            F.col(c),
            F.lit(None).cast("double")
        )
    )


# ============================================================
# 6. REMOVE INVALID TARGET
# ============================================================

clean_df = clean_df.filter(
    F.col("Delivery_Time_Minutes").isNotNull()
)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

print("\nPerforming feature engineering...")


# ------------------------------------------------------------
# ORDER HOUR
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "Order_Hour",
    F.when(
        F.col("Time_Orderd").rlike(
            r"^\d{1,2}:\d{2}(:\d{2})?$"
        ),
        F.regexp_extract(
            F.col("Time_Orderd"),
            r"^(\d{1,2}):",
            1
        ).cast("double")
    ).otherwise(None)
)


# ------------------------------------------------------------
# DELIVERY DISTANCE
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
# DATE
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "Order_Date_Parsed",
    F.when(
        F.col("Order_Date").rlike(
            r"^\d{1,2}-\d{1,2}-\d{4}$"
        ),
        F.to_date(
            F.col("Order_Date"),
            "dd-MM-yyyy"
        )
    ).otherwise(None)
)


# ------------------------------------------------------------
# WEEKEND
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
# TRAFFIC SCORE
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "Traffic_Score",
    F.when(
        F.lower(F.col("Road_traffic_density")) == "low",
        1
    )
    .when(
        F.lower(F.col("Road_traffic_density")) == "medium",
        2
    )
    .when(
        F.lower(F.col("Road_traffic_density")) == "high",
        3
    )
    .when(
        F.lower(F.col("Road_traffic_density")) == "jam",
        4
    )
    .otherwise(0)
)


# ------------------------------------------------------------
# FESTIVAL
# ------------------------------------------------------------

clean_df = clean_df.withColumn(
    "Is_Festival",
    F.when(
        F.lower(F.col("Festival")) == "yes",
        1
    ).otherwise(0)
)


# ============================================================
# 8. ML DATASET
# ============================================================

ml_df = clean_df.select(

    "Delivery_Time_Minutes",

    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",

    "Delivery_Distance_km",
    "Order_Hour",
    "Is_Weekend",
    "Traffic_Score",
    "Is_Festival",

    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "City"
)

ml_df = ml_df.dropna()


# Remove NaN from numeric features

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


print("\nRows available for ML:", ml_df.count())


# ============================================================
# 9. CATEGORICAL FEATURES
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
# 10. VECTOR ASSEMBLER
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
# 11. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = ml_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\nTraining rows:", train_df.count())
print("Testing rows :", test_df.count())


# ============================================================
# 12. GBT MODEL
# ============================================================

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="Delivery_Time_Minutes",
    seed=42
)


# ============================================================
# 13. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    stages=indexers + [
        encoder,
        assembler,
        gbt
    ]
)


# ============================================================
# 14. HYPERPARAMETER GRID
# ============================================================

param_grid = ParamGridBuilder() \
    .addGrid(
        gbt.maxDepth,
        [4, 5, 6]
    ) \
    .addGrid(
        gbt.maxIter,
        [30, 50]
    ) \
    .addGrid(
        gbt.stepSize,
        [0.05, 0.1]
    ) \
    .build()

print("\nNumber of parameter combinations:",
      len(param_grid))


# ============================================================
# 15. TRAIN / VALIDATION
# ============================================================

evaluator = RegressionEvaluator(
    labelCol="Delivery_Time_Minutes",
    predictionCol="prediction",
    metricName="rmse"
)

tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    trainRatio=0.8,
    seed=42,
    parallelism=2
)


# ============================================================
# 16. RUN TUNING
# ============================================================

print("\n" + "=" * 75)
print("STARTING GBT HYPERPARAMETER TUNING")
print("=" * 75)

tvs_model = tvs.fit(train_df)

print("\nGBT tuning completed.")


# ============================================================
# 17. BEST MODEL
# ============================================================

best_model = tvs_model.bestModel

best_gbt = best_model.stages[-1]

print("\n" + "=" * 75)
print("BEST GBT PARAMETERS")
print("=" * 75)

print(
    "maxDepth :",
    best_gbt.getMaxDepth()
)

print(
    "maxIter  :",
    best_gbt.getMaxIter()
)

print(
    "stepSize :",
    best_gbt.getStepSize()
)


# ============================================================
# 18. FINAL TEST PREDICTIONS
# ============================================================

predictions = tvs_model.transform(
    test_df
)


# ============================================================
# 19. EVALUATE
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


rmse = rmse_evaluator.evaluate(
    predictions
)

mae = mae_evaluator.evaluate(
    predictions
)

r2 = r2_evaluator.evaluate(
    predictions
)


# ============================================================
# SAVE THE EXACT TUNED PIPELINE MODEL FOR STREAMLIT
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saved_tuned_gbt_model"
)
MODEL_URI = Path(MODEL_PATH).resolve().as_uri()

print("\\nSaving the tuned GBT pipeline model...")
best_model.write().overwrite().save(MODEL_URI)
print("Saved model folder:", MODEL_PATH)
print("This folder can now be loaded by spark_sql_dashboard.py.")


# ============================================================
# 20. FINAL RESULT
# ============================================================

print("\n" + "=" * 75)
print("TUNED GBT FINAL TEST RESULTS")
print("=" * 75)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "MAE :",
    round(mae, 4)
)

print(
    "R2  :",
    round(r2, 4)
)


# ============================================================
# 21. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("SAMPLE TUNED GBT PREDICTIONS")
print("=" * 75)

predictions.select(
    "Delivery_Time_Minutes",
    "prediction"
).show(10)


# ============================================================
# 22. STOP SPARK
# ============================================================

spark.stop()

print("\nTuning and validation completed successfully.")