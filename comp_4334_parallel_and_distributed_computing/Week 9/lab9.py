# Databricks notebook source
# Raja Jain
import re
from pathlib import Path
import pyspark.sql.functions as f
udf = f.udf
from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, LongType, TimestampType
from graphframes import *
from graphframes.lib import Pregel

sc = spark.sparkContext

# CSV options
file_type = "csv"
infer_schema = "false"
delimiter = ","

# helper function
def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def rename_cols(df, new_cols_str):
    new_col_names = new_cols_str.split(", ")
    new_col_names = list(map(camel_to_snake, new_col_names))
    for exists, new in zip(df.columns, new_col_names):
        df = df.withColumnRenamed(exists, new)
    return df

def create_spark_df(path:str, first_row_is_header="true",target_columns:list=None,new_cols_str:str=None):
    df = (spark.read.format(file_type)
                .option("header", first_row_is_header)
                .option("sep", delimiter)
                .option("mode", "dropMalformed")
                .option("ignoreLeadingWhiteSpace", True)
                .option("ignoreTrailingWhiteSpace", True)
                .load(path))

    if new_cols_str:
        df = rename_cols(df, new_cols_str)
    
    if target_columns:
        df = df.select(*target_columns)
    
    return df

root_dir = Path("/FileStore/tables/flights")

# COMMAND ----------

airports_col_names = "airportID, name, city, country, IATA, ICAO, Lat, Long, Alt, timeZone, DST, databaseTimeZone, type, source"
airports = create_spark_df(str(root_dir / 'airports.csv'), first_row_is_header="false", new_cols_str=airports_col_names, target_columns=['country', 'iata'])
airports = airports.filter(f.col("country").contains("United States")).filter(f.col('iata')!='\\N')
airports = airports.drop_duplicates(subset=['iata'])
airports = airports.withColumn("id", f.monotonically_increasing_id())
airports = airports.select("id", "iata")

airports.show()

# COMMAND ----------

routes_col_names_str = "airline, airlineID, sourceAirport, sourceAirportID, destinationAirport, destinationAirportID, codeshare, stops, planeType"
routes = create_spark_df(str(root_dir / 'routes.csv'), new_cols_str=routes_col_names_str, target_columns=['source_airport', 'destination_airport'])
routes = routes.drop_duplicates()


us_airport_iatas = airports.select(f.col('iata')).rdd.map(lambda row : row[0]).collect()
for col in routes.columns:
    routes = routes.filter(f.col(col).isin(us_airport_iatas))

routes = rename_cols(routes, "src, dst")
routes.show()

# COMMAND ----------

us_airports = routes.select('src').rdd.map(lambda r: r[0]).collect()
us_airports.extend(routes.select('dst').rdd.map(lambda r: r[0]).collect())
us_airports = spark.createDataFrame(enumerate(set(us_airports)), ['index','id'])

g = GraphFrame(us_airports, routes)

g.vertices.show()
g.edges.show()

# COMMAND ----------

print("Number of US airports:", g.vertices.count())
print("Number of US to US routes:", g.edges.count())

# COMMAND ----------

denver_edges = g.edges.filter("src = 'DEN' or dst = 'DEN'")
g_denver = GraphFrame(g.vertices, denver_edges)
g_denver = g_denver.dropIsolatedVertices()

# COMMAND ----------

g_denver.find("(a)-[]->(b); !(b)-[]->(a)").show()

# COMMAND ----------

g_denver.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[cd]->(d)").show()

# COMMAND ----------

denver_shortest_paths = g.shortestPaths(landmarks=["DEN"]).persist()

# COMMAND ----------

denver_shortest_paths.show()

# COMMAND ----------

denver_shortest_paths.filter(f.col("distances")['DEN']==4).show()
