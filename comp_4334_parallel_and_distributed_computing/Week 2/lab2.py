## Raja Jain | 2021-01-09
# Databricks notebook source
# MAGIC %md
# MAGIC # COMP 4334, Lab 2
# MAGIC ## Raja Jain | 2021-01-09

# COMMAND ----------

# MAGIC %md
# MAGIC # Prime Exercise
# MAGIC ## In one databricks notebook command, create Spark code that counts the number of primes in a given range. Start by first creating a Python list of all the numbers in the range 100..10,000. Then use Spark commands to create a parallel RDD from this list. Using only Spark map, filter, reduce and/or count, count the number of primes in this range in parallel. You may use lambdas or standard Python functions in your maps/filters/reductions.

# COMMAND ----------

sc = spark.sparkContext

# create list of numbers in which to look for primes
nums = list(range(100, 10001))
nums[-1]
print(nums[0], nums[-1])

# parallelize
nums_rdd = sc.parallelize(nums)
print(nums_rdd)


def is_prime(num):
    for i in range(2, num):
        if num % i == 0:
            return False
    return True


primes = nums_rdd.filter(lambda x: is_prime(x))

print("Count of prime numbers between 10 - 10,000:", primes.count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Celsius exercise
# MAGIC ## In one databricks notebook command, create Spark code that works with temperature in the following way. Start with creating 1000 random Fahrenheit temperatures between 0..100 degrees F. This should be done in a standard Python list. Normally, we would load this data from 1000 different observations, but for this lab we will simply generate random test data. Next use Spark RDDs (only single ones – no pairRDDs) and only the Spark commands map, filter, reduce and/or count to first convert the full list to Celsius. Then find the average of all the Celsius values above freezing. You should print that average. You are only to use lambdas in your maps/filters/reductions. And you should persist RDDs if helps reduce computations.

# COMMAND ----------

import numpy as np

farenheit_temps = np.random.uniform(0, 100, 10000)

farenheit_temps_rdd = sc.parallelize(farenheit_temps)

farenheit_to_celcius_conversion = lambda x: (x - 32) * (5 / 9)

celcius_temps = farenheit_temps_rdd.map(farenheit_to_celcius_conversion)

celcius_temps_above_freezing = celcius_temps.filter(lambda x: x > 0)

print("Average of all of the Celcius values above freezing:", celcius_temps_above_freezing.mean())

# COMMAND ----------
