# Databricks notebook source
# MAGIC %md # MLflow Demo
# MAGIC 
# MAGIC See the [docs](https://mlflow.org/).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 300px; margin: 20px"/></div>

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)
X_train.head()

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.autolog(disable=True) # Disabling b/c enabled by default on this workspace

with mlflow.start_run(run_name="Basic RF Experiment") as run:
  # Create model, train it, and create predictions
  rf = RandomForestRegressor(random_state=42)
  rf.fit(X_train, y_train)
  predictions = rf.predict(X_test)

  # Log model
  mlflow.sklearn.log_model(rf, "random_forest_model")

  # Log metrics
  mse = mean_squared_error(y_test, predictions)
  mlflow.log_metric("mse", mse)

  run_id = run.info.run_id
  experiment_id = run.info.experiment_id

  print(f"Inside MLflow Run with run_id `{run_id}` and experiment_id `{experiment_id}`")

# COMMAND ----------

# MAGIC %md ### Autologging

# COMMAND ----------

mlflow.autolog()

rf.fit(X_train, y_train)
