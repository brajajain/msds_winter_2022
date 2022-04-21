# Databricks notebook source
from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, StringType, LongType, FloatType, DateType
import pyspark.sql.functions as f
udf = f.udf
from pyspark.ml.feature import Binarizer, StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pathlib import Path
from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, LongType, TimestampType
import pyspark.sql.functions as f
udf = f.udf
from pyspark.ml.feature import Binarizer, StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.regression import RandomForestRegressor
import pandas as pd
import datetime
from sklearn.metrics import accuracy_score

sc = spark.sparkContext

# COMMAND ----------

dbutils.fs.mkdirs("/FileStore/tables/assignment_2")

# COMMAND ----------

# CSV options
file_type = "csv"
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# helper function
def create_spark_df(path:str, schema=None, target_columns:list=None):
    df = (spark.read.format(file_type)
            .schema(schema)
            .option("header", first_row_is_header)
            .option("sep", delimiter)
            .option("mode", "dropMalformed")
            .option("ignoreLeadingWhiteSpace", True)
            .option("ignoreTrailingWhiteSpace", True)
            .load(path)
         )
    if target_columns:
        df = df.select(*target_columns)
    
    return df

root_dir = Path("/FileStore/tables/assignment_2")
data = str(root_dir/ "Occupancy_Estimation.csv")

# COMMAND ----------

data_schema = StructType(
    [
        StructField('Date', StringType(), True),
        StructField('Time', StringType(), True),
        StructField('S1_Temp', FloatType(), True),
        StructField('S2_Temp', FloatType(), True),
        StructField('S3_Temp', FloatType(), True),
        StructField('S4_Temp', FloatType(), True),
        StructField('S1_Light', FloatType(), True),
        StructField('S2_Light', FloatType(), True),
        StructField('S3_Light', FloatType(), True),
        StructField('S4_Light', FloatType(), True),
        StructField('S1_Sound', FloatType(), True),
        StructField('S2_Sound', FloatType(), True),
        StructField('S3_Sound', FloatType(), True),
        StructField('S4_Sound', FloatType(), True),
        StructField('S5_CO2', FloatType(), True),
        StructField('S5_CO2_Slope', FloatType(), True),
        StructField('S6_PIR', FloatType(), True),
        StructField('S7_PIR', FloatType(), True),
        StructField('Room_Occupancy_Count', LongType(), True),
    ]
)



data_df = create_spark_df(data, schema=data_schema)

# Convert Dates
date_convert_udf = udf (lambda x,y: datetime.datetime.strptime(f"{x} {y}", '%Y/%m/%d %H:%M:%S'), TimestampType())

data_df = (data_df
    .withColumn("timestamp", date_convert_udf(data_df["Date"], data_df["Time"]))
    .drop("Date","Time")
)

data_df.printSchema()

# COMMAND ----------

def split_data(data, test_size=.2):
    data_size = data.count()
    split_index = int(data_size * test_size)
    #test data
    test_data = data.limit(split_index)
    #train data
    train_data = data.withColumn("index", f.monotonically_increasing_id())
    train_data = train_data.orderBy(f.desc("index")).drop("index").limit(data_size-split_index)
    
    return test_data, train_data

test_data, train_data = split_data(data_df)

print("Original data size:", data_df.count())
print("Train data size:", train_data.count())
print("Test data size:", test_data.count())

# COMMAND ----------

vec_assembler = VectorAssembler()

feat_cols = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']

vec_assembler.setInputCols(feat_cols).setOutputCol("features")

rf = RandomForestRegressor(numTrees=100, maxDepth=15, featuresCol='features', labelCol = 'Room_Occupancy_Count')

pipeline = Pipeline(stages=[vec_assembler, rf])

model = pipeline.fit(train_data)

# COMMAND ----------

def transform_predictions(x):
    """Transforms occupancy predictions into discrete values which match expected possible outcomes.
    Rounds predicted value to nearest integer unless predicted value is greater than 3. In which case, assigns 3.
    """
    if x < 0.5:
        return 0
    elif x < 1.5:
        return 1
    elif x < 2.5:
        return 2
    else:
        return 3

transform_predictions_udf = udf(transform_predictions, LongType())

print("Training Data Performance")
train_predictions_df = model.transform(train_data).select("Room_Occupancy_Count", transform_predictions_udf("prediction")).toPandas()
display(pd.crosstab(train_predictions_df["Room_Occupancy_Count"], train_predictions_df["transform_predictions(prediction)"]))
print("Accuracy Score:", accuracy_score(train_predictions_df["Room_Occupancy_Count"], train_predictions_df["transform_predictions(prediction)"]))

print("Testing Data Performance")
test_predictions_df = model.transform(test_data).select("Room_Occupancy_Count", transform_predictions_udf("prediction")).toPandas()
display(pd.crosstab(test_predictions_df["Room_Occupancy_Count"], test_predictions_df["transform_predictions(prediction)"]))
print("Accuracy Score:", accuracy_score(test_predictions_df["Room_Occupancy_Count"], test_predictions_df["transform_predictions(prediction)"]))

# COMMAND ----------

PARTITIONS = 10

test_data_partitioned = test_data.repartition(PARTITIONS)

def create_stream_contents(df, stream_dir_path):
    print("Removing existing...")
    dbutils.fs.rm(stream_dir_path, True)
    print("Existing removed!")
    print(f"writing files at {stream_dir_path}")
    df.write.format("csv").option("header",True).save(stream_dir_path)

test_data_stream_path = str(root_dir/"data_stream")
create_stream_contents(test_data_partitioned, test_data_stream_path)

# COMMAND ----------

def create_source_stream(path:str, schema, target_columns:list=None):
    source_stream = (spark.readStream.format(file_type)
                        .schema(schema)
                        .option("header", first_row_is_header)
                        .option("sep", delimiter)
                        .option("mode", "dropMalformed")
                        .option("ignoreLeadingWhiteSpace", True)
                        .option("ignoreTrailingWhiteSpace", True)
                        .option("maxFilesPerTrigger", 1)
                        .load(path))

    return source_stream

stream_schema = StructType(
    [
        StructField('S1_Temp', FloatType(), True),
        StructField('S2_Temp', FloatType(), True),
        StructField('S3_Temp', FloatType(), True),
        StructField('S4_Temp', FloatType(), True),
        StructField('S1_Light', FloatType(), True),
        StructField('S2_Light', FloatType(), True),
        StructField('S3_Light', FloatType(), True),
        StructField('S4_Light', FloatType(), True),
        StructField('S1_Sound', FloatType(), True),
        StructField('S2_Sound', FloatType(), True),
        StructField('S3_Sound', FloatType(), True),
        StructField('S4_Sound', FloatType(), True),
        StructField('S5_CO2', FloatType(), True),
        StructField('S5_CO2_Slope', FloatType(), True),
        StructField('S6_PIR', FloatType(), True),
        StructField('S7_PIR', FloatType(), True),
        StructField('Room_Occupancy_Count', IntegerType(), True),
        StructField('timestamp', TimestampType(), True),
    ]
)
source_stream = create_source_stream(path=test_data_stream_path, schema=stream_schema)

# COMMAND ----------

model_predictions = model.transform(source_stream).select("Room_Occupancy_Count", transform_predictions_udf("prediction"))

# COMMAND ----------

def create_sink(df, query_name):
    return (df
            .writeStream
            .outputMode("append")
            .format("memory")
            .queryName(query_name)
            .trigger(processingTime="10 seconds")
            .start())

predictions_sink = create_sink(model_predictions, "model_predictions")

# COMMAND ----------

stream_predictions_df = spark.sql("""SELECT * FROM model_predictions""").toPandas()
display(pd.crosstab(stream_predictions_df["Room_Occupancy_Count"], stream_predictions_df["transform_predictions(prediction)"]))
print("Accuracy Score:", accuracy_score(stream_predictions_df["Room_Occupancy_Count"], stream_predictions_df["transform_predictions(prediction)"]))

# COMMAND ----------


