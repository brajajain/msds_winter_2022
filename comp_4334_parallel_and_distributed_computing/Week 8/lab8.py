# Databricks notebook source
# Raja Jain
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

sc = spark.sparkContext

# COMMAND ----------

# CSV options
file_type = "csv"
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# helper function
def create_spark_df(path:str, schema, target_columns:list=None):
    df = (spark.read.format(file_type)
          .schema(schema)
                .option("header", first_row_is_header)
                .option("sep", delimiter)
                .option("mode", "dropMalformed")
                .option("ignoreLeadingWhiteSpace", True)
                .option("ignoreTrailingWhiteSpace", True)
                .load(path))
    if target_columns:
        df = df.select(*target_columns)
    
    return df

root_dir = Path("/FileStore/tables/fifa_tweets")

# COMMAND ----------

fifaSchema = StructType(
    [
        StructField("ID", LongType(), True),
        StructField("lang", StringType(), True),
        StructField("Date", TimestampType(), True),
        StructField("Source", StringType(), True),
        StructField("len", LongType(), True),
        StructField("Orig_Tweet", StringType(), True),
        StructField("Tweet", StringType(), True),
        StructField("Likes", LongType(), True),
        StructField("RTs", LongType(), True),
        StructField("Hashtags", StringType(), True),
        StructField("UserMentionNames", StringType(), True),
        StructField("UserMentionID", StringType(), True),
        StructField("Name", StringType(), True),
        StructField("Place", StringType(), True),
        StructField("Followers", LongType(), True),
        StructField("Friends", LongType(), True),
    ]
)

PARTITIONS = 10
fifa_data = create_spark_df(str(root_dir/"FIFA.csv"), schema = fifaSchema).repartition(PARTITIONS)
fifa_data.printSchema()

# COMMAND ----------

def process_fifa_data(df):
    df = df.select(*["ID","Date","Hashtags"])
    df = df.filter(f.col("Hashtags").isNotNull())
    df = df.withColumn("Hashtags", f.explode(f.split('Hashtags', ',')))
    return (df.select(f.col("ID"),f.col("Date"),f.col("Hashtags"))
            .withWatermark('Date','1 minutes')
             .groupBy(f.window("Date","60 minutes","30 minutes"), f.col("Hashtags"))
             .agg(f.count("Hashtags").alias("trending"))
             .filter(f.col("trending")>100))

fifa_static_data = process_fifa_data(fifa_data)
(fifa_static_data.show())

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Streaming Files

# COMMAND ----------

def create_stream_contents(df, stream_dir_path):
    print("Removing existing...")
    dbutils.fs.rm(stream_dir_path, True)
    print("Existing removed!")
    print(f"writing files at {stream_dir_path}")
    df.write.format("csv").option("header",True).save(stream_dir_path)

fifa_stream_path = str(root_dir/"fifa_stream")
create_stream_contents(fifa_data, fifa_stream_path)

# fifa_sorted = fifa_data.orderBy(f.col("Date")).repartition(PARTITIONS).persist()
# fifa_stream_sorted_path = str(root_dir/"fifa_stream_sorted")
# create_stream_contents(fifa_sorted, fifa_stream_sorted_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # Source

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

source_stream = create_source_stream(path=fifa_stream_path, schema=fifaSchema)

# COMMAND ----------

# MAGIC %md
# MAGIC # Query

# COMMAND ----------

trending_hashtags = process_fifa_data(source_stream)

# COMMAND ----------

# MAGIC %md
# MAGIC # Sink

# COMMAND ----------

def create_sink(df, query_name):
    return (df
            .writeStream.outputMode("complete")
            .format("memory")
            .queryName(query_name)
            .trigger(processingTime="10 seconds")
            .start())

fifa_sink = create_sink(trending_hashtags, "trendingHashtags")

# COMMAND ----------

current = spark.sql(
"""
SELECT * FROM trendingHashtags
""")
current.orderBy("window").show(current.count(), False)
