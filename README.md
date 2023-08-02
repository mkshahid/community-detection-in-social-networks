# community-detection-in-social-networks

#Description
This project implements the Girvan-Newman algorithm using Spark RDD and Python/Scala libraries to perform efficient community detection in graphs. The project is part of a greater effort to understand and analyze social networks, specifically focusing on user-business interactions and preferences. The project uses a dataset of user reviews for businesses (ub_sample_data.csv) and constructs a social network graph where each node represents a user. Edges are established between users if their shared business reviews exceed a given threshold. From this graph, we aim to find communities of users with similar business tastes.

#Procedures
The project involves two main tasks:

##Task 1: Community Detection using GraphFrames
In the first task, we utilize the Spark GraphFrames library to detect communities in the constructed graph. Here, we apply the Label Propagation Algorithm (LPA), allowing for iterative community detection based on the edge structure of the graph.

##Task 2: Community Detection using Girvan-Newman algorithm
In the second task, we implement the Girvan-Newman algorithm using Spark RDD and standard Python or Scala libraries. This algorithm involves calculating the betweenness of each edge, and iteratively removing edges with the highest betweenness until the graph is divided into communities.

In both tasks, the communities are saved in a txt file and sorted by size in ascending order, and by the first user_id in lexicographical order within each community.
