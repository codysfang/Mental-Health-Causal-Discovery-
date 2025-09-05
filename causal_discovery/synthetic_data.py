import os
import pandas as pd
import numpy as np
os.sys.path.append('scm-identify/DGM')
os.sys.path.append('scm-identify/StructureLearning/RLCD')
os.sys.path.append('scm-identify')
from LinearSCM import LinearSCM
import networkx as nx
import matplotlib.pyplot as plt
from RLCD_alg import AdjToGraph
from utils.GraphDrawer import printGraph

def synthetic_set_graph(seed=123):
    dgm = LinearSCM(seed=seed)
    dgm.add_variable("L1", False)
    dgm.add_variable("L2", False)
    dgm.add_variable("L3", False)
    dgm.add_variable("L4", False)
    dgm.add_variable("L5", False)
    dgm.add_variable("L6", False)

    dgm.add_variable("X1", True)
    dgm.add_variable("X2", True)
    dgm.add_variable("X10", True)
    dgm.add_variable("X11", True)
    dgm.add_variable("X12", True)
    dgm.add_variable("X13", True)
    dgm.add_variable("X14", True)
    dgm.add_variable("X15", True)
    dgm.add_variable("X16", True)
    dgm.add_variable("X17", True)
    dgm.add_variable("X18", True)
    dgm.add_variable("X19", True)
    dgm.add_variable("X3", True)
    dgm.add_variable("X4", True)
    dgm.add_variable("X5", True)
    dgm.add_variable("X6", True)
    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)
    dgm.add_variable("X9", True)

    dgm.add_edge("L1", "L3")
    dgm.add_edge("L1", "L2")
    dgm.add_edge("L1", "L4")
    dgm.add_edge("L3", "L4")
    dgm.add_edge("L4", "L5")
    dgm.add_edge("L6", "L1")
    dgm.add_edge("L6", "L2")

    dgm.add_edge("L1", "X1")
    dgm.add_edge("L1", "X2")
    dgm.add_edge("L1", "X3")
    dgm.add_edge("X3", "X4")
    dgm.add_edge("X3", "X5")
    dgm.add_edge("X1", "X6")
    dgm.add_edge("X1", "X7")

    dgm.add_edge("L2", "X8")
    dgm.add_edge("L2", "X9")
    dgm.add_edge("L2", "X10")
    dgm.add_edge("L3", "X11")
    dgm.add_edge("L3", "X12")
    dgm.add_edge("L4", "X13")
    dgm.add_edge("L4", "X14")
    dgm.add_edge("L5", "X15")
    dgm.add_edge("L5", "X16")
    dgm.add_edge("L6", "X17")
    dgm.add_edge("L6", "X18")
    dgm.add_edge("L6", "X19")

    return dgm, "set_graph"

def synthetic_balanced_graph(seed=123):
    dgm = LinearSCM(seed=seed)
    
    dgm.add_variable("L1", False)
    dgm.add_variable("L2", False)
    dgm.add_variable("L3", False)
    dgm.add_variable("X1", True)
    dgm.add_variable("X10", True)
    dgm.add_variable("X2", True)
    dgm.add_variable("X3", True)
    dgm.add_variable("X4", True)
    dgm.add_variable("X5", True)
    dgm.add_variable("X6", True)
    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)
    dgm.add_variable("X9", True)

    dgm.add_edge("L1", "L2")
    dgm.add_edge("L1", "L3")

    dgm.add_edge("L2", "X1")
    dgm.add_edge("L2", "X2")
    dgm.add_edge("L2", "X3")
    dgm.add_edge("L2", "X4")
    dgm.add_edge("L2", "X5")
    dgm.add_edge("L3", "X6")
    dgm.add_edge("L3", "X7")
    dgm.add_edge("L3", "X8")
    dgm.add_edge("L3", "X9")
    dgm.add_edge("L3", "X10")

    return dgm, "balanced_graph"


def synthetic_unbalanced_graph(seed=123):
    dgm = LinearSCM(seed=seed)
    
    dgm.add_variable("L1", False)
    dgm.add_variable("L2", False)
    dgm.add_variable("L3", False)
    dgm.add_variable("X1", True)
    dgm.add_variable("X10", True)
    dgm.add_variable("X2", True)
    dgm.add_variable("X3", True)
    dgm.add_variable("X4", True)
    dgm.add_variable("X5", True)
    dgm.add_variable("X6", True)
    dgm.add_variable("X7", True)
    dgm.add_variable("X8", True)
    dgm.add_variable("X9", True)

    dgm.add_edge("L1", "L2")
    dgm.add_edge("L1", "L3")

    dgm.add_edge("L2", "X1")
    dgm.add_edge("L2", "X2")
    dgm.add_edge("L3", "X3")
    dgm.add_edge("L3", "X4")
    dgm.add_edge("L3", "X5")
    dgm.add_edge("L3", "X6")
    dgm.add_edge("L3", "X7")
    dgm.add_edge("L3", "X8")
    dgm.add_edge("L3", "X9")
    dgm.add_edge("L3", "X10")

    return dgm, "unbalanced_graph"

def saveplot(m, name):
    adj = np.asarray(m.F!=0, dtype=int)
    np.save(f"data/synthetic/graphs/{name}_adj.npy", adj)
    adj = m.get_causallearn_adj()

    dot_graph = AdjToGraph(adj, m.vars)
    printGraph(dot_graph, f'./data/synthetic/graphs/{name}_true.png')
    return

def discretize(df):
    for col in df:
        df[col] = pd.qcut(df[col], [0, 0.29, 0.48, 0.69, 0.88, 1], labels=False).astype(int)
    return

def generate_save_linear(m, name, n=10):
    X, V = m.generate_data(N=n, normalized=False)

    def get_corr(model):
        B = model.F.T
        I = np.eye(B.shape[0])
        variance = np.diag(model.omega)
        inv = np.linalg.inv(I - B)
        sigma = inv @ variance @ inv.T
        std = np.sqrt(np.diag(sigma))
        inv_sqrt = np.diag(1.0/std)

        return inv_sqrt @ sigma @ inv_sqrt

    np.save(f"data/synthetic/graphs/corr_{name}.npy", get_corr(m))

    discretize(X)
    X.to_csv(f"./data/synthetic/{name}_linear.csv", index=False)
    V.to_csv(f"./data/synthetic/V/{name}_linear_full.csv", index=False)
    return

def generate_save_additive(m, name, n=10):

    def generate_additive_data():
        adj = m.F!=0
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        df_x, df_v = pd.DataFrame(), pd.DataFrame()
        for i, node in enumerate(list(nx.topological_sort(G))):
            parents = list(G.predecessors(node))
            std = np.sqrt(m.omega[i])
            noise = np.random.normal(loc=0, scale=std, size=n)
            if not parents:
                df_v[node] = noise
            else:
                df_v[node] = np.exp(-1 * df_v[parents].sum(axis=1)) + noise

        df_v.columns = m.vars
        df_x = df_v.filter(regex="^X+").copy()

        return df_x, df_v

    
    X, V = generate_additive_data()
    discretize(X)
    X.to_csv(f"./data/synthetic/{name}_additive.csv", index=False)
    V.to_csv(f"./data/synthetic/V/{name}_additive_full.csv", index=False)
    return


def generate_save_non_diff(m, name, n=10):
    def generate_non_diff_data():
        adj = m.F!=0
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        df_x, df_v = pd.DataFrame(), pd.DataFrame()
        for i, node in enumerate(list(nx.topological_sort(G))):
            parents = list(G.predecessors(node))
            std = np.sqrt(m.omega[i])
            noise = np.random.normal(loc=0, scale=std, size=n)
            if not parents:
                df_v[node] = np.maximum(noise + 1, -2 * noise ** 2)
            else:
                df_v[node] = np.maximum(noise + df_v[parents].sum(axis=1) + 1, -2 * (noise + df_v[parents].sum(axis=1)) ** 2)

        df_v.columns = m.vars
        df_x = df_v.filter(regex="^X+").copy()

        return df_x, df_v

    
    X, V = generate_non_diff_data()
    discretize(X)
    X.to_csv(f"./data/synthetic/{name}_non_diff.csv", index=False)
    V.to_csv(f"./data/synthetic/V/{name}_non_diff_full.csv", index=False)
    return

if __name__ == "__main__":

    if not os.path.exists("./data/synthetic/graphs"):
        os.makedirs("./data/synthetic/graphs")
    
    if not os.path.exists("./data/synthetic/V"):
        os.makedirs("./data/synthetic/V")


    set_graph = synthetic_set_graph()
    balanced_graph = synthetic_balanced_graph()
    unbalanced_graph = synthetic_unbalanced_graph()
    for g in [set_graph, balanced_graph, unbalanced_graph]:
        name = g[1]
        graph = g[0]
        saveplot(graph, name)
        generate_save_linear(graph, name, n=10000)
        generate_save_additive(graph, name, n=10000)
        generate_save_non_diff(graph, name, n=10000)
