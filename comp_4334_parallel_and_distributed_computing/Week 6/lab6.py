# Databricks notebook source
# Raja Jain

from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, StringType, IntegerType
import pyspark.sql.functions as f

sc = spark.sparkContext


# file paths
master_path = "/FileStore/tables/Master.csv"
all_star_path = "/FileStore/tables/AllstarFull.csv"
teams_path = "/FileStore/tables/Teams.csv"


# CSV options
file_type = "csv"
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# helper function
def create_spark_df(path: str, target_columns: list = None):
    df = (
        spark.read.format(file_type)
        .option("inferSchema", infer_schema)
        .option("header", first_row_is_header)
        .option("sep", delimiter)
        .option("ignoreLeadingWhiteSpace", True)
        .option("ignoreTrailingWhiteSpace", True)
        .load(path)
    )
    if target_columns:
        df = df.select(*target_columns)

    return df


master_df = create_spark_df(master_path, target_columns=["playerID", "nameFirst", "nameLast"])
all_star_df = create_spark_df(all_star_path, target_columns=["playerID", "teamID"])
teams_df = create_spark_df(teams_path, target_columns=["name", "teamID"])


joined_df = (
    teams_df.join(all_star_df, ["teamID"])
    .join(master_df, ["playerID"])
    .distinct()
    .select("playerID", "teamID", "nameFirst", "nameLast", f.col("name").alias("teamName"))
)

joined_df.printSchema()

joined_df.show()


dir_name = f"/FileStore/tables/lab6_all_star_teams_parquet_data"
dbutils.fs.mkdirs(dir_name)
# write parquet
(joined_df.write.format("parquet").mode("overwrite").partitionBy("teamName").save(dir_name))


# readparquet
parquet_df = spark.read.format("parquet").load(dir_name)

parquet_df.select("nameFirst", "nameLast").filter(f.col("teamName") == "Colorado Rockies").show(n=50, truncate=False)

parquet_df.select("nameFirst", "nameLast").filter(f.col("teamName") == "Colorado Rockies").count()
