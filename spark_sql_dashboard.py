import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

try:
    import shap
except ImportError:
    shap = None
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor

HDFS_PATH = "hdfs://localhost:9000/user/utk60/new_food_data/new_food_data.csv"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_tuned_gbt_model")
# Spark's default filesystem may be HDFS; explicitly use the local Windows filesystem
# when saving/loading the model artifact.
MODEL_URI = Path(MODEL_PATH).resolve().as_uri()

st.set_page_config(page_title="Food Delivery Analytics & Prediction", page_icon="🚚", layout="wide")
st.title("Food Delivery Analytics & Delivery-Time Prediction")
st.caption("Apache Hadoop HDFS • Apache Spark SQL • Spark MLlib")
st.sidebar.header("Dashboard controls")
st.sidebar.code(HDFS_PATH, language="text")

@st.cache_resource
def get_spark():
    return (SparkSession.builder.appName("FoodDelivery_Analytics_Prediction_Dashboard")
            .master("local[*]").getOrCreate())

@st.cache_resource(show_spinner="Loading and preparing the HDFS dataset...")
def load_data():
    spark = get_spark()
    df = (spark.read.option("header", "true").option("inferSchema", "true").csv(HDFS_PATH))
    string_cols = ["Weatherconditions", "Road_traffic_density", "Type_of_order", "Type_of_vehicle",
                   "Festival", "City", "Order_Date", "Time_Orderd", "Time_Order_picked",
                   "Time_taken(min)", "Delivery_person_ID", "ID"]
    for c in string_cols:
        if c in df.columns:
            df = df.withColumn(c, F.trim(F.col(c).cast("string")))
            df = df.withColumn(c, F.when(F.col(c).isNull() | (F.col(c) == "") |
                                         (F.lower(F.col(c)) == "nan"), None).otherwise(F.col(c)))
    if "Time_taken(min)" not in df.columns:
        raise ValueError("Expected target column 'Time_taken(min)' was not found in the HDFS CSV.")
    df = df.withColumn("Delivery_Time_Minutes", F.expr("try_cast(regexp_extract(`Time_taken(min)`, '(\\\\d+)', 1) as double)"))
    numeric_cols = ["Delivery_person_Age", "Delivery_person_Ratings", "Restaurant_latitude", "Restaurant_longitude",
                    "Delivery_location_latitude", "Delivery_location_longitude", "Vehicle_condition", "multiple_deliveries"]
    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, F.expr(f"try_cast(`{c}` as double)"))
            df = df.withColumn(c, F.nanvl(F.col(c), F.lit(None).cast("double")))
    coords = {"Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude"}
    if coords.issubset(set(df.columns)):
        lat1, lon1 = F.radians("Restaurant_latitude"), F.radians("Restaurant_longitude")
        lat2, lon2 = F.radians("Delivery_location_latitude"), F.radians("Delivery_location_longitude")
        a = F.pow(F.sin((lat2-lat1)/2), 2) + F.cos(lat1)*F.cos(lat2)*F.pow(F.sin((lon2-lon1)/2), 2)
        a = F.least(F.lit(1.0), F.greatest(F.lit(0.0), a))
        df = df.withColumn("Delivery_Distance_km", 2 * F.atan2(F.sqrt(a), F.sqrt(1-a)) * 6371)
    else:
        df = df.withColumn("Delivery_Distance_km", F.lit(None).cast("double"))
    if "Time_Orderd" in df.columns:
        df = df.withColumn("Order_Hour", F.when(F.col("Time_Orderd").rlike(r"^\d{1,2}:\d{2}(:\d{2})?$"),
                                 F.regexp_extract("Time_Orderd", r"^(\d{1,2}):", 1).cast("double")))
    else:
        df = df.withColumn("Order_Hour", F.lit(None).cast("double"))
    if "Order_Date" in df.columns:
        df = df.withColumn("Is_Weekend", F.when(F.dayofweek(F.to_date("Order_Date", "dd-MM-yyyy")).isin([1, 7]), 1.0).otherwise(0.0))
    else:
        df = df.withColumn("Is_Weekend", F.lit(0.0))
    if "Road_traffic_density" in df.columns:
        df = df.withColumn("Traffic_Score", F.when(F.lower("Road_traffic_density").contains("low"), 1.0)
                           .when(F.lower("Road_traffic_density").contains("medium"), 2.0)
                           .when(F.lower("Road_traffic_density").contains("high"), 3.0)
                           .when(F.lower("Road_traffic_density").contains("jam"), 4.0))
    else:
        df = df.withColumn("Traffic_Score", F.lit(None).cast("double"))
    if "Festival" in df.columns:
        df = df.withColumn("Is_Festival", F.when(F.lower("Festival").contains("yes"), 1.0)
                           .when(F.lower("Festival").contains("no"), 0.0))
    else:
        df = df.withColumn("Is_Festival", F.lit(0.0))
    # Ensure every predictor column exists, even if a source CSV lacks one.
    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition", "multiple_deliveries"]:
        if c not in df.columns: df = df.withColumn(c, F.lit(None).cast("double"))
    for c in ["Weatherconditions", "Road_traffic_density", "Type_of_order", "Type_of_vehicle", "City"]:
        if c not in df.columns: df = df.withColumn(c, F.lit("Unknown"))
    df = df.filter(F.col("Delivery_Time_Minutes").isNotNull() & (~F.isnan("Delivery_Time_Minutes")))
    df.createOrReplaceTempView("food_delivery")
    return df

NUMERIC_FEATURES = ["Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition", "multiple_deliveries",
                    "Delivery_Distance_km", "Order_Hour", "Is_Weekend", "Traffic_Score", "Is_Festival"]
CATEGORICAL_FEATURES = ["Weatherconditions", "Road_traffic_density", "Type_of_order", "Type_of_vehicle", "City"]

@st.cache_resource(show_spinner="Loading saved model or training the tuned GBT model for the first time...")
def get_model(_df):
    # Load the exact fitted pipeline saved by tune_gbt.py.
    # Do not train a different model inside the dashboard.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "The tuned GBT model was not found at: "
            + MODEL_PATH
            + ". Run tune_gbt.py successfully first, then restart Streamlit."
        )
    return PipelineModel.load(MODEL_URI), "Loaded the tuned GBT model saved by tune_gbt.py"


# SHAP uses a model-agnostic KernelExplainer around the saved Spark ML pipeline.
# Categorical values are represented by stable integer codes internally and decoded
# back to their original labels before calling Spark. Explanations are approximate.
EXPLAIN_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

@st.cache_resource(show_spinner="Preparing SHAP explanation background...")
def get_shap_context(_df):
    if shap is None:
        return None
    # Small, representative background keeps Kernel SHAP practical on a laptop.
    sample_rows = (_df.select(*EXPLAIN_FEATURES)
                   .na.drop(subset=["Delivery_Distance_km"])
                   .limit(200).toPandas())
    if sample_rows.empty:
        raise ValueError("No usable rows were found to build the SHAP background.")
    # Fill numeric nulls using medians from the sample.
    for c in NUMERIC_FEATURES:
        sample_rows[c] = pd.to_numeric(sample_rows[c], errors="coerce")
        median = sample_rows[c].median()
        sample_rows[c] = sample_rows[c].fillna(0.0 if pd.isna(median) else float(median))
    category_maps = {}
    for c in CATEGORICAL_FEATURES:
        vals = _df.select(c).where(F.col(c).isNotNull()).distinct().limit(100).toPandas()[c].astype(str).tolist()
        vals = sorted(set(vals + ["Unknown"]))
        category_maps[c] = {v: i for i, v in enumerate(vals)}
        sample_rows[c] = sample_rows[c].fillna("Unknown").astype(str)
        # Ensure values from the background have a code.
        for v in sample_rows[c].unique():
            if v not in category_maps[c]:
                category_maps[c][v] = len(category_maps[c])
    encoded = sample_rows.copy()
    for c in CATEGORICAL_FEATURES:
        encoded[c] = encoded[c].map(category_maps[c]).fillna(category_maps[c]["Unknown"]).astype(float)
    return encoded[EXPLAIN_FEATURES].head(8).to_numpy(dtype=float), category_maps

def _decode_shap_matrix(matrix, category_maps):
    decoded_rows = []
    for values in np.asarray(matrix):
        row = {}
        for i, c in enumerate(EXPLAIN_FEATURES):
            value = float(values[i])
            if c in CATEGORICAL_FEATURES:
                mapping = category_maps[c]
                code = int(round(value))
                code = max(0, min(code, len(mapping) - 1))
                # Reverse the stable integer encoding used for the explanation.
                reverse = {idx: label for label, idx in mapping.items()}
                row[c] = reverse.get(code, "Unknown")
            else:
                row[c] = value
        decoded_rows.append(row)
    return decoded_rows

def explain_prediction_with_shap(model, background, category_maps, input_row):
    """Return SHAP values for one input using a small background and bounded samples."""
    encoded_input = dict(input_row)
    for c in CATEGORICAL_FEATURES:
        mapping = category_maps[c]
        if input_row[c] not in mapping:
            mapping[input_row[c]] = len(mapping)
        encoded_input[c] = float(mapping[input_row[c]])
    x = np.array([[float(encoded_input[c]) for c in EXPLAIN_FEATURES]], dtype=float)

    def predict_encoded(matrix):
        decoded = _decode_shap_matrix(matrix, category_maps)
        # Spark accepts the whole perturbation batch at once, avoiding per-row jobs.
        sdf = get_spark().createDataFrame(decoded)
        predictions = model.transform(sdf).select("prediction").toPandas()["prediction"]
        return predictions.to_numpy(dtype=float)

    explainer = shap.KernelExplainer(predict_encoded, background, feature_names=EXPLAIN_FEATURES)
    values = explainer.shap_values(x, nsamples=80, silent=True)
    values = np.asarray(values)
    if values.ndim > 1:
        values = values.reshape(-1, values.shape[-1])[0]
    base_value = np.asarray(explainer.expected_value).reshape(-1)[0]
    result = pd.DataFrame({"Feature": EXPLAIN_FEATURES, "SHAP value (minutes)": values,
                           "Selected value": [input_row[c] for c in EXPLAIN_FEATURES]})
    result["Absolute impact"] = result["SHAP value (minutes)"].abs()
    return float(base_value), result.sort_values("Absolute impact", ascending=False)

QUERIES = {
    "Traffic density": """SELECT Road_traffic_density AS Traffic_Level, COUNT(*) AS Total_Orders, ROUND(AVG(Delivery_Time_Minutes),2) AS Avg_Delivery_Minutes, ROUND(AVG(Delivery_Distance_km),2) AS Avg_Distance_Km FROM food_delivery WHERE Road_traffic_density IS NOT NULL GROUP BY Road_traffic_density ORDER BY Avg_Delivery_Minutes DESC""",
    "Weather conditions": """SELECT Weatherconditions AS Weather, COUNT(*) AS Total_Orders, ROUND(AVG(Delivery_Time_Minutes),2) AS Avg_Delivery_Minutes FROM food_delivery WHERE Weatherconditions IS NOT NULL GROUP BY Weatherconditions ORDER BY Avg_Delivery_Minutes DESC""",
    "City comparison": """SELECT City, COUNT(*) AS Total_Orders, ROUND(AVG(Delivery_Time_Minutes),2) AS Avg_Delivery_Minutes, ROUND(AVG(Delivery_Distance_km),2) AS Avg_Distance_Km FROM food_delivery WHERE City IS NOT NULL GROUP BY City ORDER BY Avg_Delivery_Minutes DESC""",
    "Vehicle type": """SELECT Type_of_vehicle AS Vehicle_Type, COUNT(*) AS Total_Orders, ROUND(AVG(Delivery_Time_Minutes),2) AS Avg_Delivery_Minutes FROM food_delivery WHERE Type_of_vehicle IS NOT NULL GROUP BY Type_of_vehicle ORDER BY Avg_Delivery_Minutes DESC""",
    "Vehicle condition": """SELECT Vehicle_condition, COUNT(*) AS Total_Orders, ROUND(AVG(Delivery_Time_Minutes),2) AS Avg_Delivery_Minutes FROM food_delivery WHERE Vehicle_condition IS NOT NULL GROUP BY Vehicle_condition ORDER BY Vehicle_condition""",
    "Order type": """SELECT Type_of_order AS Order_Type, COUNT(*) AS Total_Orders, ROUND(AVG(Delivery_Time_Minutes),2) AS Avg_Delivery_Minutes FROM food_delivery WHERE Type_of_order IS NOT NULL GROUP BY Type_of_order ORDER BY Avg_Delivery_Minutes DESC""",
    "Delivery-time distribution": """SELECT Delivery_Time_Minutes, COUNT(*) AS Order_Count FROM food_delivery GROUP BY Delivery_Time_Minutes ORDER BY Delivery_Time_Minutes""",
    "Distance vs delivery time": """SELECT ROUND(Delivery_Distance_km,0) AS Distance_Band_Km, COUNT(*) AS Total_Orders, ROUND(AVG(Delivery_Time_Minutes),2) AS Avg_Delivery_Minutes FROM food_delivery WHERE Delivery_Distance_km IS NOT NULL AND Delivery_Distance_km >= 0 GROUP BY ROUND(Delivery_Distance_km,0) ORDER BY Distance_Band_Km""",
    "Festival comparison": """SELECT Festival, COUNT(*) AS Total_Orders, ROUND(AVG(Delivery_Time_Minutes),2) AS Avg_Delivery_Minutes FROM food_delivery WHERE Festival IS NOT NULL GROUP BY Festival ORDER BY Avg_Delivery_Minutes DESC"""
}

try:
    spark = get_spark()
    df = load_data()
    rows = df.count()
    avg_time = df.agg(F.avg("Delivery_Time_Minutes")).first()[0]
    median_time = df.approxQuantile("Delivery_Time_Minutes", [0.5], 0.01)[0]
    avg_distance = df.agg(F.avg("Delivery_Distance_km")).first()[0]
    a, b, c, d = st.columns(4)
    a.metric("Usable records", f"{rows:,}")
    b.metric("Columns after feature engineering", str(len(df.columns)))
    c.metric("Average delivery time", f"{avg_time:.2f} min")
    d.metric("Median delivery time", f"{median_time:.2f} min")
    st.caption(f"Average calculated straight-line distance: {avg_distance:.2f} km")

    tab_predict, tab_analytics = st.tabs(["🔮 Predict delivery time", "📊 Spark SQL analytics"])
    with tab_predict:
        st.subheader("Predict a new delivery")
        st.write("Choose order and delivery conditions. The trained GBT model estimates the delivery duration in minutes.")
        st.info("Use realistic values. Distance is straight-line distance calculated from restaurant and delivery coordinates in the dataset; enter an estimated distance if you do not have coordinates.")
        with st.form("prediction_form"):
            p1, p2, p3 = st.columns(3)
            age = p1.slider("Delivery person's age", 18, 60, 30)
            rating = p2.slider("Delivery person's rating", 1.0, 5.0, 4.5, 0.1)
            distance = p3.number_input("Delivery distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
            hour = p1.slider("Order hour (0–23)", 0, 23, 19)
            vehicle_condition = p2.selectbox("Vehicle condition", [0, 1, 2, 3], index=2, help="Choose the condition value used in the dataset.")
            multiple = p3.selectbox("Multiple deliveries", [0, 1, 2, 3], index=1)
            weather_values = ["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy"]
            traffic_values = ["Low", "Medium", "High", "Jam"]
            order_values = ["Snack", "Meal", "Drinks", "Buffet"]
            vehicle_values = ["motorcycle", "scooter", "electric_scooter", "bicycle"]
            city_values = ["Metropolitian", "Urban", "Semi-Urban"]
            weather = p1.selectbox("Weather", weather_values)
            traffic = p2.selectbox("Road traffic", traffic_values, index=1)
            order_type = p3.selectbox("Order type", order_values)
            vehicle_type = p1.selectbox("Vehicle type", vehicle_values)
            city = p2.selectbox("City category", city_values)
            festival = p3.selectbox("Festival day?", ["No", "Yes"])
            weekend = p1.selectbox("Weekend?", ["No", "Yes"])
            submitted = st.form_submit_button("Predict delivery time", type="primary")
        if submitted:
            try:
                model, model_status = get_model(df)
                traffic_score = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Jam": 4.0}[traffic]
                row = {"Delivery_person_Age": float(age), "Delivery_person_Ratings": float(rating),
                       "Vehicle_condition": float(vehicle_condition), "multiple_deliveries": float(multiple),
                       "Delivery_Distance_km": float(distance), "Order_Hour": float(hour),
                       "Is_Weekend": 1.0 if weekend == "Yes" else 0.0,
                       "Traffic_Score": traffic_score, "Is_Festival": 1.0 if festival == "Yes" else 0.0,
                       "Weatherconditions": weather, "Road_traffic_density": traffic,
                       "Type_of_order": order_type, "Type_of_vehicle": vehicle_type, "City": city}
                input_df = spark.createDataFrame([row])
                # Ensure input uses the same numeric feature columns and categories as training.
                prediction = model.transform(input_df).select("prediction").first()[0]
                st.success(f"Estimated delivery time: {prediction:.1f} minutes")
                st.caption(f"Model status: {model_status}. This is an estimate, not a guaranteed delivery time.")

                st.markdown("### Why did the model make this prediction?")
                if shap is None:
                    st.warning("SHAP is not installed in this Python environment. Stop Streamlit and run: python -m pip install shap")
                else:
                    with st.spinner("Calculating approximate SHAP contributions..."):
                        shap_context = get_shap_context(df)
                        background, category_maps = shap_context
                        baseline, shap_table = explain_prediction_with_shap(model, background, category_maps, row)
                    st.metric("SHAP baseline (minutes)", f"{baseline:.1f}")
                    st.caption("Each SHAP value shows how a feature moves the prediction up or down from the baseline. Positive values increase predicted time; negative values decrease it. Kernel SHAP is approximate and may take a little time.")
                    plot_df = shap_table.head(10).sort_values("SHAP value (minutes)")
                    st.markdown("#### Largest feature contributions")
                    st.bar_chart(plot_df.set_index("Feature")[["SHAP value (minutes)"]])
                    st.markdown("#### Contribution details")
                    st.dataframe(shap_table[["Feature", "Selected value", "SHAP value (minutes)"]].round(3), use_container_width=True)
                    top = shap_table.head(3)
                    for _, item in top.iterrows():
                        direction = "increased" if item["SHAP value (minutes)"] > 0 else "reduced"
                        st.write(f"• **{item['Feature']}** ({item['Selected value']}) {direction} the prediction by approximately {abs(item['SHAP value (minutes)']):.2f} minutes.")
            except Exception as e:
                st.error("Prediction failed. Check the error below and share it if you need help.")
                st.exception(e)
    with tab_analytics:
        st.subheader("Run Spark SQL analytics")
        choice = st.sidebar.selectbox("Choose an analysis", list(QUERIES) + ["Preview cleaned data"])
        if choice == "Preview cleaned data":
            st.dataframe(df.limit(100).toPandas(), use_container_width=True)
            st.caption("Preview is limited to 100 rows.")
        else:
            sql = QUERIES[choice]
            st.markdown("#### SQL executed")
            st.code(sql, language="sql")
            pdf = spark.sql(sql).toPandas()
            st.markdown("#### Results")
            st.dataframe(pdf, use_container_width=True)
            if not pdf.empty:
                if choice == "Delivery-time distribution":
                    st.bar_chart(pdf.set_index("Delivery_Time_Minutes")[["Order_Count"]])
                elif choice == "Distance vs delivery time":
                    st.line_chart(pdf.set_index("Distance_Band_Km")[["Avg_Delivery_Minutes"]])
                else:
                    category = next((x for x in pdf.columns if x not in {"Total_Orders", "Avg_Delivery_Minutes", "Avg_Distance_Km"}), None)
                    if category and "Avg_Delivery_Minutes" in pdf.columns:
                        st.bar_chart(pdf.set_index(category)[["Avg_Delivery_Minutes"]])
                    if category and "Total_Orders" in pdf.columns:
                        st.markdown("#### Order counts")
                        st.bar_chart(pdf.set_index(category)[["Total_Orders"]])
    with st.expander("Show Spark schema"):
        st.code(df._jdf.schema().treeString(), language="text")
    st.sidebar.success("Connected to HDFS through Spark")
    st.sidebar.caption("Spark SQL runs in the local Spark session. Prediction uses Spark MLlib GBT regression.")
except Exception as e:
    st.error("Dashboard could not load or query the HDFS dataset.")
    st.exception(e)
    st.info("Check that HDFS is running, the CSV exists at the HDFS path, and Spark environment variables are set in the CMD used to start Streamlit.")
