import time
from pyspark import SparkContext, Row
import sys
from graphframes import GraphFrame
from pyspark.sql import SparkSession

sc = SparkContext('local[*]', 'task1')
spark = SparkSession.builder.appName(sc.appName).getOrCreate()

sc.setLogLevel("WARN")

filter_threshold = int(sys.argv[1])
#filter_threshold = 2
input_file = sys.argv[2]
#input_file = "../resource/asnlib/publicdata/ub_sample_data.csv"
output_file = sys.argv[3]
#output_file = "./output1.txt"

start_time = time.time()

reviews_data = sc.textFile(input_file).filter(lambda x: "user_id" not in x).map(lambda line: line.split(",")).persist()
reviews_data = reviews_data.map(lambda x: (x[0], x[1]))
user_business_data = reviews_data.groupByKey().mapValues(set)
user_pairs_rdd = user_business_data.cartesian(user_business_data).filter(lambda x: x[0][0] != x[1][0])
common_businesses_rdd = user_pairs_rdd.map(lambda x: (x[0][0], x[1][0], len(x[0][1].intersection(x[1][1]))))
filtered_common_businesses_rdd = common_businesses_rdd.filter(lambda x: x[2] >= filter_threshold)

user_ids_rdd = user_business_data.map(lambda x: x[0]).distinct()

edges_forward = filtered_common_businesses_rdd.map(lambda x: Row(x[0], x[1])).persist()
edges_reverse = filtered_common_businesses_rdd.map(lambda x: Row(x[1], x[0])).persist()
edges_df = spark.createDataFrame(sc.union([edges_forward, edges_reverse]).distinct(), ["src", "dst"])

vertices_src = filtered_common_businesses_rdd.map(lambda x: Row(x[0])).persist()
vertices_dst = filtered_common_businesses_rdd.map(lambda x: Row(x[1])).persist()
vertices_df = spark.createDataFrame(sc.union([vertices_src, vertices_dst]).distinct(), ["id"])

g = GraphFrame(vertices_df, edges_df)

communities = g.labelPropagation(maxIter=5)
output = communities.rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: sorted(list(x))) \
    .sortBy(lambda x: (len(x[1]), x[1])).map(lambda x: tuple(x[1])).collect()

with open(output_file, "w") as file:
    for community in output:
        file.write("\'" + community[0] + "\'")
        for i in range(len(community) - 1):
            file.write(", \'" + community[i + 1] + "\'")
        file.write("\n")

print(time.time() - start_time)