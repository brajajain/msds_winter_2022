# Databricks notebook source
# Raja Jain

from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, StringType, IntegerType
import pyspark.sql.functions as f

sc = spark.sparkContext
file_location = "/FileStore/tables/Master.csv"

raw_text_rdd = sc.textFile(file_location)
split_text_rdd = raw_text_rdd.map(lambda x: x.split(",")).persist()

columns = split_text_rdd.first()

raw_data_rdd = split_text_rdd.filter(lambda x: x != columns).persist()
raw_data_rdd = raw_data_rdd.filter(lambda x: any([True for i in x if i == ""]) is False)


rdd_rows = raw_data_rdd.map(
    lambda x: Row(playedID=x[0], birthCountry=x[4], birthState=x[5], height=int(x[17]))
)

rdd_struct = StructType(
    [
        StructField("playerID", StringType(), True),
        StructField("birthCountry", StringType(), True),
        StructField("birthState", StringType(), True),
        StructField("height", IntegerType(), False),
    ]
)

df = spark.createDataFrame(rdd_rows, rdd_struct)

df.createOrReplaceTempView("master")

def run_queries(query:str, df_query):
    print("* "*10, "SQL QUERY RESULTS", "* "*10)
    query_results = spark.sql(query).collect()
    print(query_results)
    
    print()
    
    print("* "*10, "DATA FRAME FUNCTIONS RESULTS", "* "*10)
    df_results = df_query.collect()
    print(df_results)
    
display(df)


# COMMAND ----------

colorado_query = """
SELECT birthState, COUNT(playerID) AS count_player_ID 
FROM master 
WHERE birthState = 'CO' 
GROUP BY birthState
"""

df_colorado_query = (df
.filter(f.col('birthState') == 'CO')
.groupBy('birthState')
.agg(f.count('playerID'))
)
                     
run_queries(colorado_query, df_colorado_query)

# COMMAND ----------

run_queries(
"""
SELECT birthCountry, AVG(height) AS average_player_height
FROM master 
GROUP BY birthCountry
ORDER BY average_player_height DESC
"""
,
    
(
df
.select('height', 'birthCountry')
.groupBy(f.col('birthCountry'))
.agg(f.avg(f.col('height')))
.sort('avg(height)', ascending=False)
)
)


# COMMAND ----------


