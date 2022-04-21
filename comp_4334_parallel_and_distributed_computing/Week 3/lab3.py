# Raja Jain


def copy_files_with_subname_to_dir(subname: str = None, num_files: int = None):
    directory_name = f"FileStore/tables/lab3{subname}"
    dbutils.fs.mkdirs(directory_name)
    for i in range(num_files):
        file_name = f"FileStore/tables/{subname}Lab3data{i}.txt"
        print(f"Copying\n    file: {file_name}\n    to directory: {directory_name}")
        dbutils.fs.cp(file_name, directory_name)


# COMMAND ----------

# copy short data
copy_files_with_subname_to_dir("short", 2)

# COMMAND ----------

# copy full data
copy_files_with_subname_to_dir("full", 4)

# COMMAND ----------

# helper functions to process url data
sc = spark.sparkContext


def return_tuple_from_url_list(url_list):
    # split urls by space
    split_urls = url_list.split(" ")
    return split_urls[0], split_urls[1:]


def find_reference_web_pages(data_path: str = None):
    # import raw data
    raw_data = sc.textFile(data_path)
    # convert to rdd
    url_pair_rdd = raw_data.map(return_tuple_from_url_list)
    # expands k,v into tuples where [(k, v1), (k, v2)]
    flattened_pair_rdd = url_pair_rdd.flatMapValues(lambda x: x)
    # swap positions of tuples to set referenced webpages as keys for later use in an aggregation function
    swap_tuples = flattened_pair_rdd.map(lambda x: (x[1], x[0]))
    # aggregation operation, group data by referenced webpage and return list of webpages that referenced it
    referenced_web_pages = swap_tuples.groupByKey().mapValues(
        lambda x: sorted(list(set(x)))
    )
    referenced_web_pages = referenced_web_pages.sortByKey()
    return referenced_web_pages


# COMMAND ----------

# get results for short data
short_data_dir_path = "FileStore/tables/lab3short"

short_data_web_page_references = find_reference_web_pages(short_data_dir_path)
short_data_web_page_references = short_data_web_page_references.collect()
for res in short_data_web_page_references:
    print(res)

# COMMAND ----------

# get results for full data
full_data_dir_path = "FileStore/tables/lab3full"

full_data_web_page_references = find_reference_web_pages(full_data_dir_path)
print(full_data_web_page_references.count())
full_data_web_page_references_take_10 = full_data_web_page_references.take(10)
for res in full_data_web_page_references_take_10:
    print(res)
