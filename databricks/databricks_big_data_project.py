# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Big Data Project
# MAGIC

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/diabetes.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# Read data
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("BigDataProject").getOrCreate()

df = spark.read.format(file_type) \
    .option("inferSchema", infer_schema) \
    .option("header", first_row_is_header) \
    .option("sep", delimiter) \
    .load(file_location)

df.show(5)
df.printSchema()

# COMMAND ----------

# Data Cleaning - Replace 0 values with median
from pyspark.sql.functions import when, col

columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col_name in columns_to_fix:
    median_value = df.filter(df[col_name] > 0).approxQuantile(col_name, [0.5], 0.0)[0]
    df = df.withColumn(col_name, when(df[col_name] == 0, median_value).otherwise(df[col_name]))

df.show(5)

# COMMAND ----------

# Distribution of Diabetes Outcome
from pyspark.sql.functions import count

df.groupBy("Outcome").agg(count("Outcome").alias("count")).show()

# COMMAND ----------

# Prepare Data
from pyspark.ml.feature import VectorAssembler

feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

df_transformed = assembler.transform(df)
df_transformed = df_transformed.select("features", "Outcome")

df_transformed.show(5)

# COMMAND ----------

# Train ML Models
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="Outcome"),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="Outcome"),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="Outcome", numTrees=10),
    "Gradient Boosted Trees": GBTClassifier(featuresCol="features", labelCol="Outcome")
}

evaluator = MulticlassClassificationEvaluator(labelCol="Outcome", metricName="accuracy")

best_model = None
best_model_name = ""
best_accuracy = 0

for name, model in models.items():
    trained_model = model.fit(df_transformed)
    predictions = trained_model.transform(df_transformed)
    accuracy = evaluator.evaluate(predictions)
    print(f"{name} Accuracy: {accuracy:.4f}")

    # Store the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = trained_model
        best_model_name = name

print(f"Best Model: {best_model_name} with {best_accuracy:.4f} accuracy")

# COMMAND ----------

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

pdf = df.toPandas()

# Glucose Distribution
sns.histplot(pdf["Glucose"], bins=20)
plt.xlabel("Glucose Level")
plt.ylabel("Frequency")
plt.title("Glucose Distribution")
plt.show()

# Outcome Distribution
sns.histplot(pdf[pdf["Outcome"] == 1]["Glucose"], color="red", label="Diabetic", kde=True)
sns.histplot(pdf[pdf["Outcome"] == 0]["Glucose"], color="blue", label="Non-Diabetic", kde=True)

plt.xlabel("Glucose Level")
plt.ylabel("Frequency")
plt.title("Glucose Distribution by Diabetes Outcome")
plt.legend()
plt.show()
