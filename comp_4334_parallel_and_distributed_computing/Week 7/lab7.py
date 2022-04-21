# Databricks notebook source
# Raja Jain

from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, StringType, IntegerType
import pyspark.sql.functions as f

udf = f.udf
from pyspark.ml.feature import Binarizer, StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

sc = spark.sparkContext


# file paths
test_path = "/FileStore/tables/heartTesting.csv"
train_path = "/FileStore/tables/heartTraining.csv"


# CSV options
file_type = "csv"
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# helper function
def create_spark_df(path: str, schema, target_columns: list = None):
    df = (
        spark.read.format(file_type)
        .schema(schema)
        .option("header", first_row_is_header)
        .option("sep", delimiter)
        .option("ignoreLeadingWhiteSpace", True)
        .option("ignoreTrailingWhiteSpace", True)
        .load(path)
    )
    if target_columns:
        df = df.select(*target_columns)

    return df


schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("sex", StringType(), True),
        StructField("chol", IntegerType(), True),
        StructField("pred", StringType(), False),
    ]
)

train_df = create_spark_df(train_path, schema=schema)
test_df = create_spark_df(test_path, schema=schema)


def age_descritizer(age: int):
    category = "below_40"
    if age > 40 and age <= 49:
        category = "40_49"
    elif age > 50 and age <= 59:
        category = "50_59"
    elif age > 60 and age <= 69:
        category = "60_69"
    elif age > 70:
        category = "70_above"

    return category


age_descritizer_udf = udf(age_descritizer, StringType())


class AgeDescritizer(Transformer):
    def __init__(self, inputCol="val1", outputCol="val2"):
        super().__init__()
        self.inCol = inputCol
        self.outCol = outputCol

    def _transform(self, df):
        return df.withColumn(self.outCol, age_descritizer_udf(f.col(self.inCol))).drop(self.inCol)


class AgeDummies(Transformer):
    def __init__(self, inputCol="ageCat", outputCols=None):
        super().__init__()
        self.inCol = inputCol
        self.outCol = outputCols

    def _transform(self, df):
        return df.select(*[*self.outCol, *self.__get_types_expr__(df)])

    def __get_types_expr__(self, df):
        types = df.select(self.inCol).distinct().rdd.flatMap(lambda x: x).collect()
        types_expr = [f.when(f.col(self.inCol) == ty, 1).otherwise(0).alias(f"{self.inCol}_" + ty) for ty in types]
        return types_expr


age_transform = AgeDescritizer(inputCol="age", outputCol="ageCat")

age_dummies_transform = AgeDummies(outputCols=["id", "sex", "chol", "pred"])

indexer = StringIndexer()
indexer.setInputCols(["sex", "pred"]).setOutputCols(["sexCat", "predCat"])

vecAssembler = VectorAssembler()
vecAssembler.setInputCols(
    ["sexCat", "chol", "ageCat_below_40", "ageCat_60_69", "ageCat_40_49", "ageCat_50_59", "ageCat_70_above"]
).setOutputCol("features")

lr = LogisticRegression(featuresCol="features", labelCol="predCat")


pipeline = Pipeline(stages=[age_transform, age_dummies_transform, indexer, vecAssembler, lr])
model = pipeline.fit(train_df)


model.transform(train_df).show()


train_results = model.transform(train_df).select("predCat", "rawPrediction", "probability", "prediction")

train_results.show()

print("Train AUC ROC:")
BinaryClassificationMetrics(train_results.select("predCat", "prediction").rdd.map(tuple)).areaUnderROC


test_results = model.transform(test_df).select("predCat", "rawPrediction", "probability", "prediction")
test_results.show()

print("Test AUC ROC:")
BinaryClassificationMetrics(test_results.select("predCat", "prediction").rdd.map(tuple)).areaUnderROC
