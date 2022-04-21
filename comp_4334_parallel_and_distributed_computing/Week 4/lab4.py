# Databricks notebook source
demo_data = ["a b c", "b a a", "c b", "d a"]
# helper functions to process url data
sc = spark.sparkContext
demo_rdd = sc.parallelize(demo_data, 3)

# COMMAND ----------

def return_tuple_from_url_list(url_list):
    # split urls by space
    split_urls = url_list.split(" ")
    return split_urls[0], split_urls[1:]


def find_neighboring_links(data):
    # convert to rdd
    data = data.map(return_tuple_from_url_list)
    data = data.mapValues(lambda x: sorted(list(set(x))))
    return data.sortByKey()

def compute_initial_score(data):
    distinct_pages = data.map(lambda a: a.split(" ")).flatMap(lambda x: x).distinct().persist()
    initial_scores = 1 / distinct_pages.count()
    return distinct_pages.map(lambda x: (x, initial_scores))

def compute_neighbor_contributions(x):
    """compute contributions from neighbors and return a list containing proportional contribution per neighbor"""
    n = len(x[0])
    contribution = x[1] / n
    return [(x, contribution) for x in x[0]]

def compute_page_rank(data, iterations:int):
    links = find_neighboring_links(data)
    rankings = compute_initial_score(data)
    print("Initial links:",links.collect())
    print("Initial rankings:",rankings.collect())
    joined_rdd = links.join(rankings)
    for i in range(iterations): 
        print("Iteration:",i)    
        print("Joined RDD:", joined_rdd.collect())
        
        # compute neighbor contributions and return as a flattened list rather than a list of lists
        neighbor_contributions = (joined_rdd.mapValues(lambda x: compute_neighbor_contributions(x))
                                  .values()
                                  .flatMap(lambda x: x)
                                 )
        print("Neighbor contributions:", neighbor_contributions.collect())
        
        # compute new rankings 
        rankings = neighbor_contributions.reduceByKey(lambda a,b: a+b)
        print("New rankings:", rankings.collect())
        
        # rejoin datasets
        joined_rdd = links.join(rankings)
        
    print("Final sorted rankings:")
    rankings = rankings.sortByKey()
    for x in rankings.collect():
        print(f"{x[0]} has rank: {x[1]}")
        
def compute_page_rank_no_print(data, iterations:int):
    links = find_neighboring_links(data)
    rankings = compute_initial_score(data)
    joined_rdd = links.join(rankings)
    for i in range(iterations): 
        # compute neighbor contributions and return as a flattened list rather than a list of lists
        neighbor_contributions = (joined_rdd.mapValues(lambda x: compute_neighbor_contributions(x))
                                  .values()
                                  .flatMap(lambda x: x)
                                 )
        # compute new rankings 
        rankings = neighbor_contributions.reduceByKey(lambda a,b: a+b)
        
        # rejoin datasets
        joined_rdd = links.join(rankings)
        
    print("Final sorted rankings:")
    rankings = rankings.sortBy(lambda x: x[1], ascending=False, numPartitions=3)
    for x in rankings.collect():
        print(f"{x[0]} has rank: {x[1]}")
    return rankings

# COMMAND ----------

compute_page_rank(demo_rdd, 10)

# COMMAND ----------

short_files = "dbfs:///FileStore/tables/lab3short/"
short_rdd = sc.textFile(short_files)
compute_page_rank_no_print(short_rdd, 10)
