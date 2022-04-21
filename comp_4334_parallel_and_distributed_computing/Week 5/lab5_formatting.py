# Raja Jain

from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, StringType, IntegerType


sc = spark.sparkContext
file_location = "/FileStore/tables/Master.csv"

raw_text_rdd = sc.textFile(file_location)
split_text_rdd = raw_text_rdd.map(lambda x: x.split(",")).persist()

columns = split_text_rdd.first()
raw_data_rdd = split_text_rdd.filter(lambda x: x != columns).persist()
raw_data_rdd = raw_data_rdd.filter(lambda x: sum([True for i in x if i == ""]) == 0)
raw_data_rdd = raw_data_rdd.map(lambda x: [i.strip(" '") for i in x])

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

display(df)
