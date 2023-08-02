
import time
from collections import defaultdict

from pyspark import SparkContext
import sys

sc = SparkContext('local[*]', 'task2')

sc.setLogLevel("WARN")

filter_threshold = int(sys.argv[1])
#filter_threshold = 6
input_file = sys.argv[2]
#input_file = "../resource/asnlib/publicdata/ub_sample_data.csv"
betweenness_output_file = sys.argv[3]
#betweenness_output_file = "./betweenness2.txt"
modularity_output_file = sys.argv[4]
#modularity_output_file = "./modularity.txt"


start_time = time.time()

reviews_data = sc.textFile(input_file).filter(lambda x: "user_id" not in x).map(lambda line: line.split(",")).persist()
reviews_data = reviews_data.map(lambda x: (x[0], x[1]))
users = reviews_data.map(lambda x: x[0]).distinct()
user_business_data = reviews_data.groupByKey().mapValues(set)
user_pairs_rdd = user_business_data.cartesian(user_business_data).filter(lambda x: x[0][0] != x[1][0])
common_businesses_rdd = user_pairs_rdd.map(lambda x: (x[0][0], x[1][0], len(x[0][1].intersection(x[1][1]))))
filtered_common_businesses_rdd = common_businesses_rdd.filter(lambda x: x[2] >= filter_threshold)

edges_forward = filtered_common_businesses_rdd.map(lambda x: (x[0], x[1])).persist()
edges_reverse = filtered_common_businesses_rdd.map(lambda x: (x[1], x[0])).persist()
edges = sc.union([edges_forward, edges_reverse]).distinct()

vertices_src = filtered_common_businesses_rdd.map(lambda x: x[0]).persist()
vertices_dst = filtered_common_businesses_rdd.map(lambda x: x[1]).persist()
vertices = sc.union([vertices_src, vertices_dst]).distinct()
graph = dict(edges.groupByKey().mapValues(set).collect())


def bfs(graph, root):
    parent = defaultdict(list)
    depth = defaultdict(int)
    shortest_path = defaultdict(float)
    credit = defaultdict(float)
    queue = []
    visited = []

    parent[root] = []
    depth[root] = 0
    shortest_path[root] = 1.0
    visited.append(root)

    for child_node in graph[root]:
        parent[child_node] = [root]
        depth[child_node] = 1
        shortest_path[child_node] = 1.0
        queue.append(child_node)
        visited.append(child_node)

    while queue:
        node = queue.pop(0)
        credit[node] = 1.0
        num_paths = 0
        for parent_node in parent[node]:
            num_paths += shortest_path[parent_node]
        shortest_path[node] = num_paths
        for neighbour in graph[node]:
            if neighbour not in visited:
                parent[neighbour] = [node]
                depth[neighbour] = depth[node] + 1
                queue.append(neighbour)
                visited.append(neighbour)
            else:
                if depth[neighbour] == depth[node] + 1:
                    parent[neighbour].append(node)

    bfs_order = visited[::-1]
    for child_node in bfs_order[:-1]:
        for parent_node in parent[child_node]:
            contribution = credit[child_node] * (shortest_path[parent_node] / shortest_path[child_node])
            credit[parent_node] += contribution
            yield tuple(sorted([child_node, parent_node])), contribution


betweenness = vertices.flatMap(lambda x: bfs(graph, x)).reduceByKey(lambda x, y: x + y).map(
    lambda x: (x[0], round(x[1]/2, 5))).sortBy(lambda x: (-x[1], x[0])).collect()
with open(betweenness_output_file, "w") as file:
    file.write("(\'" + betweenness[0][0][0] + "\', \'" + betweenness[0][0][1] + "\')," + str(betweenness[0][1]))
    for edges, b in betweenness[1:]:
        file.write("\n(\'" + edges[0] + "\', \'" + edges[1] + "\')," + str(b))


def find_communities(graph, vertices):
    communities = []
    queue = []
    visited_nodes = []
    for vertex in vertices:
        if vertex not in visited_nodes:
            visited = [vertex]
            queue.append(vertex)
            while queue:
                node = queue.pop(0)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.append(neighbor)
                        queue.append(neighbor)
            visited.sort()
            visited_nodes = visited_nodes + visited
            communities.append(visited)
    return communities, max(len(sublist) for sublist in communities)


m = filtered_common_businesses_rdd.count()


def calculate_modularity(communities):
    modularity = 0
    for community in communities:
        for i in community:
            for j in community:
                actual_existence = 1 if j in graph[i] else 0
                expected_existence = (len(graph[i]) * len(graph[i])) / (2 * m)
                modularity += actual_existence - expected_existence
    return modularity / (2 * m)


def remove_edges(edges1, graph1):
    for edge in edges1:
        graph1[edge[0]].remove(edge[1])
        graph1[edge[1]].remove(edge[0])
    return graph1


modular_community = []
best_modularity = -1
all_vertices = vertices.collect()
all_communities = [all_vertices]
modularity = 1
while modularity > best_modularity - .05:
    modularity = calculate_modularity(all_communities)

    if modularity > best_modularity:
        best_modularity = modularity
        modular_community = all_communities

    edges_to_remove = vertices.flatMap(lambda x: bfs(graph, x)).reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[1]/2, [x[0]])).reduceByKey(lambda x, y: x + y).sortBy(lambda x: -x[0]).map(
        lambda x: x[1]).first()

    remove_edges(edges_to_remove, graph)

    all_communities, max_length = find_communities(graph, all_vertices)

final_communities = sc.parallelize(modular_community).sortBy(lambda x: (len(x), x)).collect()

with open(modularity_output_file, "w") as file:
    for community in final_communities:
        file.write("\'" + community[0] + "\'")
        for i in range(len(community) - 1):
            file.write(", \'" + community[i + 1] + "\'")
        file.write("\n")
print(time.time() - start_time)