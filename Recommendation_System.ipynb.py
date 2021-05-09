# Databricks notebook source
# MAGIC %sh 
# MAGIC /databricks/python3/bin/python3 -m pip install gensim==3.8.3
# MAGIC %sh 
# MAGIC /databricks/python3/bin/python3 -m pip install xgboost
# MAGIC %sh 
# MAGIC /databricks/python3/bin/python3 -m pip install networkx

# COMMAND ----------

import networkx as nx
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T

# COMMAND ----------

def alias_setup(probs):
    """
    compute utility lists for non-uniform sampling from discrete distributions.
    details: https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    :param probs: 某个概率分布
    """
    K = len(probs) # K为类型数目
    q = np.zeros(K) # 对应q数组：落在原类型的概率
    J = np.zeros(K, dtype=np.int) # 对应J数组：每一列第二层的类型
    
    #Sort the data into the outcomes with probabilities
    #that are larger and smaller than 1/K
    smaller = list() # 存储比1小的列
    larger = list() # 存储比1大的列
    
    for kk, prob in enumerate(probs):
        q[kk] = K * prob # 概率（每个类别概率乘以K，使得总和为K）
        if q[kk] < 1.0: # 然后分为两类：大于1的和小于1的
            smaller.append(kk)
        else:
            larger.append(kk)

    # 通过拼凑，将各个类别都凑为1
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0 #将大的分到小的上
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def get_alias_node(graph, node):
    """
    get the alias node setup lists for a given node.
    """
    # get the unnormalized probabilities with the first-order information
    unnormalized_probs = list()
    for nbr in graph.neighbors(node):
        unnormalized_probs.append(graph[node][nbr]["weight"])
    unnormalized_probs = np.array(unnormalized_probs)
    if len(unnormalized_probs) > 0:
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()
    else:
        normalized_probs = unnormalized_probs
        
    return alias_setup(normalized_probs)
    
def get_alias_edge(graph, src, dst, p=1, q=1):
    """
    get the alias edge setup lists for a given edge.
    """
    # get the unnormalized probabilities with the second-order information
    unnormalized_probs = list()
    for dst_nbr in graph.neighbors(dst):
        if dst_nbr == src: # distance is 0
            unnormalized_probs.append(graph[dst][dst_nbr]["weight"]/p)
        elif graph.has_edge(dst_nbr, src): # distance is 1
            unnormalized_probs.append(graph[dst][dst_nbr]["weight"])
        else: # distance is 2
            unnormalized_probs.append(graph[dst][dst_nbr]["weight"]/q)
    unnormalized_probs = np.array(unnormalized_probs)
    if len(unnormalized_probs) > 0:
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()
    else:
        normalized_probs = unnormalized_probs

    return alias_setup(normalized_probs)

def preprocess_transition_probs(graph, p=1, q=1):
    """
    preprocess transition probabilities for guiding the random walks.
    """
    alias_nodes = dict()
    for node in graph.nodes():
        alias_nodes[node] = get_alias_node(graph, node)

    alias_edges = dict()
    for edge in graph.edges():
        alias_edges[edge] = get_alias_edge(graph, edge[0], edge[1], p=p, q=q)

    return alias_nodes, alias_edges

# COMMAND ----------

def alias_draw(J, q):
    """
    draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


# helper function to generate the long random walk as desired
def fallback(walk, fetch_last_num=1):
    if len(walk) > fetch_last_num:
        walk.pop()
        fetched = []
        for i in range(fetch_last_num):
            fetched.append(walk[-1-i])
        return walk, fetched
    else:
        return [], [None for _ in range(fetch_last_num)]

# COMMAND ----------

def load_data(file_name):
    """
    read edges from an edge file
    """
    edges = list()
    df = pd.read_csv(file_name)
    #df = sqlContext.read.format('com.databricks.spark.csv').options(delimiter=',', header='true', inferschema='true').load(file_name)

    for idx, row in df.iterrows():
        user_id, friends = row["user_id"], eval(row["friends"])
        for friend in friends:
            # add each friend relation as an edge
            edges.append((user_id, friend))
    edges = sorted(edges)
    print("The number of edges is", len(edges))
    nodes = list(set(chain.from_iterable(edges)))
    print("The number of nodes is", len(nodes))
    return edges, nodes

def load_test_data(file_name):
    """
    read edges from an edge file
    """
    edges = list()
    scores = list()
    df = pd.read_csv(file_name)
    #df = sqlContext.read.format('com.databricks.spark.csv').options(delimiter=',', header='true', inferschema='true').load(file_name)
    for idx, row in df.iterrows():
        edges.append((row["src"], row["dst"]))
    edges = sorted(edges)
    print("The number of edges is", len(edges))
    nodes = list(set(chain.from_iterable(edges)))
    print("The number of nodes is", len(nodes))
    return edges

def generate_false_edges(true_edges, num_false_edges=5):
    """
    generate false edges given true edges
    """
    # chain.from_iterable 将所有edges的nodes合成一个iterable，set去重后转为list
    nodes = list(set(chain.from_iterable(true_edges)))
    N = len(nodes)
    true_edges = set(true_edges) # true_edges去重
    print("The number of nodes and edges before false edges generation:", N, len(true_edges)) #输出 nodes数量，edges数量
    false_edges = set()
    
    while len(false_edges) < num_false_edges:
        # randomly sample two different nodes and check whether the pair exisit or not
        src, dst = nodes[int(np.random.rand() * N)], nodes[int(np.random.rand() * N)]
        # 去掉 A->A 类的自指 edge && 不在 true_edges 中 && 与之前 false_edges 不重复
        if src != dst and (src, dst) not in true_edges and (src, dst) not in false_edges:
            false_edges.add((src, dst))
    false_edges = sorted(false_edges)
    print("Generated ", len(false_edges),'false edges...')
    return false_edges

def construct_graph_from_edges(edges):
    """
    generate a directed graph object given true edges
    DiGraph documentation: https://networkx.github.io/documentation/stable/reference/classes/digraph.html
    """
    # convert a list of edges {(u, v)} to a list of edges with weights {(u, v, w)}
    edge_weight = defaultdict(float)
    
    # edge_weight 由重复次数决定？
    for e in edges:
        edge_weight[e] += 1.0
    weighed_edge_list = list()
    for e in sorted(edge_weight.keys()):
        weighed_edge_list.append((e[0], e[1], edge_weight[e]))
        
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(weighed_edge_list)
    
    print("number of nodes:", graph.number_of_nodes())
    print("number of edges:", graph.number_of_edges())
    
    return graph
  
udf_construct_graph_from_edges = F.udf(construct_graph_from_edges, T.StringType())

# COMMAND ----------

def generate_second_order_random_walk(graph, alias_nodes, alias_edges, 
                                      walk_length=10, start_node=None, verbose=False, max_trails=10):
    """
    simulate a random walk starting from start node and considering the second order information.
    """
    if start_node == None:
        start_node = np.random.choice(graph.nodes())
    walk = [start_node]
    
    prev = None
    cur = start_node
    num_tried = 0

    ########## begin ##########
    while len(walk) < walk_length:
        cur_nbrs = list(graph.neighbors(cur))
        if len(cur_nbrs) > 0:
            if prev is None:
                # sample the next node based on alias_nodes
                prev, cur = cur, cur_nbrs[alias_draw(*alias_nodes[cur])]
            else:
                # sample the next node based on alias_edges
                prev, cur = cur, cur_nbrs[alias_draw(*alias_edges[(prev, cur)])]
            walk.append(cur)
        else:
            num_tried += 1
            if num_tried >= max_trails:
                break
            walk, (cur, prev) = fallback(walk, fetch_last_num=2)
            if len(walk) == 0:
                start_node = np.random.choice(graph.nodes())
                walk = [start_node]
                cur = start_node
                prev = None
    ########## end ##########
    #if verbose: 
    #    print(f'walk of lenght {len(walk)} generated with {num_tried} trails')
   # return walk

# COMMAND ----------

from pyspark.sql.functions import length
from pyspark.sql.functions import count

# COMMAND ----------

def build_node2vec(graph, alias_nodes, alias_edges, node_dim=10, num_walks=10, walk_length=10):
    """
    build a node2vec model
    """
    print("\nbuilding a node2vec model...", end="\t")
    st = time.time()
    np.random.seed(0)
    nodes = list(graph.nodes())
    walks = list()
    # generate random walks
    for walk_iter in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walks.append(generate_second_order_random_walk(graph, alias_nodes, alias_edges, walk_length=walk_length, start_node=node))
            
    walk_lens=[len(w) for w in walks]
    
    if len(walk_lens) > 0:
        avg_walk_len = sum(walk_lens) / len(walk_lens)
    else:
        avg_walk_len = 0.0    
    print("number of walks: %d\taverage walk length: %.4f" % (length(walks), avg_walk_len), end="\t")
    
    # train a skip-gram model for these walks
    model = Word2Vec(walks, vector_size=node_dim, window=3, min_count=0, sg=1, workers=os.cpu_count(), epochs=10)
    print("training time: %.4f" % (time.time()-st))
    
    return model
  
udf_build_node2vec = F.udf(build_node2vec, T.FloatType())

# COMMAND ----------

def get_cosine_sim(model, u, v):
    """
    get the cosine similarity between two nodes
    """
    try:
        u = model.wv[u]
        v = model.wv[v]
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    except:
        return 0.5

def get_auc_score(model, true_edges, false_edges):
    """
    get the auc score
    """
    y_true = [1] * len(true_edges) + [0] * len(false_edges)
    
    y_score = list()
    for e in true_edges:
        sim = get_cosine_sim(model, e[0], e[1])
        y_score.append(sim)
    for e in false_edges:
        sim = get_cosine_sim(model, e[0], e[1])
        y_score.append(sim)
    
    return roc_auc_score(y_true, y_score)

def write_pred(file_name, edges, scores):
    df = pd.DataFrame()
    df["src"] = [e[0] for e in edges]
    df["dst"] = [e[1] for e in edges]
    df["score"] = scores
    df.to_csv(file_name, index=False)
    

def write_valid_ans(file_name, edges, scores):
    df = pd.DataFrame()
    df["src"] = [e[0] for e in edges]
    df["dst"] = [e[1] for e in edges]
    df["score"] = scores
    df.to_csv(file_name, index=False)

# COMMAND ----------

import urllib
ACCESS_KEY = "AKIAVWMBO7IYZT4QM56I"
SECRET_KEY = "W5iaJpZ3GPbGrEJC4rIX2H/LYTadEHZI+baN/Vya"
encoded_secret_key = SECRET_KEY.replace("/", "%2F")
AWS_BUCKET_NAME = "comp4651-project-yzhangfe"
MOUNT_NAME = "s3"
#dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY, encoded_secret_key, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)
#display(dbutils.fs.ls("/mnt/%s" % MOUNT_NAME))
display(dbutils.fs.ls("/mnt/s3/data"))

# COMMAND ----------

DF = sqlContext.read.format('com.databricks.spark.csv').options(delimiter=',', header='true', inferschema='true').load("dbfs:/mnt/s3/data/train.csv")
display(DF)

# COMMAND ----------

import os.path
#filePath = 'dbfs:/FileStore/tables/shakespere.txt'
train_file = "/dbfs/mnt/s3/data/train.csv"
valid_file = "/dbfs/mnt/s3/data/valid.csv"
test_file = "/dbfs/mnt/s3/data/test.csv"

np.random.seed(0)
print("Load train edges...")
train_edges, train_nodes = load_data(train_file)
print("\nGenerate graph...")
graph = construct_graph_from_edges(train_edges)
print("\nLoad valid edges...")
valid_edges,valid_nodes = load_data(valid_file)
print("\nGenerate train_false edges...")
train_false_edges = generate_false_edges(train_edges, 100000)
print("\nGenerate valid_false edges...")
false_edges = generate_false_edges(train_edges+valid_edges, 40000-len(valid_edges))
print("\nLoad test edges...")
#test_edges = load_test_data(test_file)

# COMMAND ----------

print (graph)

# COMMAND ----------

np.random.seed(0)

node_dim = 10
num_walks = 10
walk_length = 10
p = 0.5
q = 0.5

print("node dim: %d,\tnum_walks: %d,\twalk_length: %d,\tp: %.2f,\tq: %.2f" % (
    node_dim, num_walks, walk_length, p, q), end="\t")

alias_nodes, alias_edges = preprocess_transition_probs(graph, p=p, q=q)
model = build_node2vec(graph, alias_nodes, alias_edges, node_dim, num_walks, walk_length)

# COMMAND ----------

len(model.wv.index_to_key)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train data

# COMMAND ----------

def get_embedding(node):
    try:
        node_emd = model.wv[node]
        return np.array(node_emd)
    except:
        return avg_vec

# COMMAND ----------

all_vec = np.zeros((8328,10))
i = -1

for node in model.wv.index_to_key:
    i += 1
    w_vec = model.wv[node]
    all_vec[i] = w_vec

avg_vec = np.mean(all_vec,axis=0)

# COMMAND ----------

em_data=np.zeros((200000,10))
i = -1
for (a,b) in train_edges:
    i += 1
    add_list = get_embedding(a)+get_embedding(b)
    em_data[i] = add_list
    
i = 99999
for (a,b) in train_false_edges:
    i += 1
    add_list = get_embedding(a)+get_embedding(b)
    em_data[i] = add_list

# COMMAND ----------

ytrain=[]
for i in range(100000):
    ytrain.append(1)
for i in range(100000):
    ytrain.append(0)
ytrain=np.array(ytrain)
ytrain.reshape((200000,-1))

# COMMAND ----------

print(em_data.shape)
print(ytrain.shape)
data = pd.DataFrame(em_data)
lable = pd.DataFrame(ytrain)

# COMMAND ----------

import xgboost as xgb

dtrain = xgb.DMatrix(data, label=lable)

xg_reg = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.5,
                max_depth = 20, alpha = 10, n_estimators = 100)
xg_reg.fit(data,lable)

# COMMAND ----------

em_valid_data=np.zeros((40000,10))
i=-1
for (a,b) in valid_edges:
    i+=1
    add_list = get_embedding(a)+get_embedding(b)
    em_valid_data[i] = add_list
    
i=19267
for (a,b) in false_edges:
    i+=1
    add_list = get_embedding(a)+get_embedding(b)
    em_valid_data[i] = add_list

# COMMAND ----------

yvalid=[]
for i in range(19268):
    yvalid.append(1)
for i in range(eval("40000-19268")):
    yvalid.append(0)
yvalid=np.array(yvalid)
yvalid.reshape((40000,-1))

# COMMAND ----------

print(em_valid_data.shape)
print(yvalid.shape)
em_valid_data = pd.DataFrame(em_valid_data)
valid_lable = pd.DataFrame(yvalid)

# COMMAND ----------

preds = xg_reg.predict(em_valid_data)
preds

# COMMAND ----------

auc_scores=roc_auc_score(yvalid, preds)
print (auc_scores)

# COMMAND ----------

import xgboost as xgb

# COMMAND ----------

df = pd.DataFrame(np.random.randn(3, 4),
                index=[10, 20, 40], columns=[5, 10, 20, 40])
for num_walk in [5, 10, 20, 40]:
    for walk_len in  [10, 20, 40]:
        np.random.seed(0)

        node_dim = 10
        num_walks = num_walk
        walk_length = walk_len
        p = 0.5
        q = 0.5

        print("node dim: %d,\tnum_walks: %d,\twalk_length: %d,\tp: %.2f,\tq: %.2f" % (
            node_dim, num_walks, walk_length, p, q), end="\t")

        alias_nodes, alias_edges = preprocess_transition_probs(graph, p=p, q=q)
        model = build_node2vec(graph, alias_nodes, alias_edges, 
                               node_dim=node_dim, num_walks=num_walks, walk_length=walk_length)

        def get_embedding(node):
            try:
                node_emd = model.wv[node]
                return np.array(node_emd)
            except:
                return avg_vec

        all_vec = np.zeros((8328,10))
        i = -1

        for node in model.wv.index_to_key:
            i += 1
            w_vec = model.wv[node]
            all_vec[i] = w_vec

        avg_vec = np.mean(all_vec,axis=0)

        em_data=np.zeros((200000,10))
        i = -1
        for (a,b) in train_edges:
            i += 1
            add_list = get_embedding(a)+get_embedding(b)
            em_data[i] = add_list

        i = 99999
        for (a,b) in train_false_edges:
            i += 1
            add_list = get_embedding(a)+get_embedding(b)
            em_data[i] = add_list

        ytrain=[]
        for i in range(100000):
            ytrain.append(1)
        for i in range(100000):
            ytrain.append(0)
        ytrain=np.array(ytrain)
        ytrain.reshape((200000,-1))

        data = pd.DataFrame(em_data)
        lable = pd.DataFrame(ytrain)

        dtrain = xgb.DMatrix(data, label=lable)
        xg_reg = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.5,
                        max_depth = 20, alpha = 10, n_estimators = 100)
        xg_reg.fit(data,lable)

        em_valid_data=np.zeros((40000,10))
        i=-1
        for (a,b) in valid_edges:
            i+=1
            add_list = get_embedding(a)+get_embedding(b)
            em_valid_data[i] = add_list

        i=19267
        for (a,b) in false_edges:
            i+=1
            add_list = get_embedding(a)+get_embedding(b)
            em_valid_data[i] = add_list

        yvalid=[]
        for i in range(19268):
            yvalid.append(1)
        for i in range(eval("40000-19268")):
            yvalid.append(0)
        yvalid=np.array(yvalid)
        yvalid.reshape((40000,-1))

        em_valid_data = pd.DataFrame(em_valid_data)
        valid_lable = pd.DataFrame(yvalid)

        preds = xg_reg.predict(em_valid_data)

        auc_scores = roc_auc_score(yvalid, preds)
        df.loc[walk_length,num_walks] = auc_scores
print (df)

# COMMAND ----------

print (df)

# COMMAND ----------

plt.figure(figsize=(3, 4))
node_dim = 10
# you should have an auc score dictionary here.
a = np.array( [ (0.964782  ,0.959141  ,0.960523 ,0.965694 ),
            (0.959553  ,0.961295  ,0.964317 ,0.964535 ),
           (0.959810,0.961637  ,0.964211  ,0.964695  )] )
plt.imshow(a, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.yticks(ticks=[0,1,2], labels=[40, 20, 10])
plt.ylabel("walk_length")
plt.xticks(ticks=[0,1,2,3], labels=[5, 10, 20, 40])
plt.xlabel("num_walks")
