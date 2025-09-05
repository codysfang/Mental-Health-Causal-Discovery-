import sys
sys.path.append("latent-causal-models")
sys.path.append("scm-identify/StructureLearning/RLCD")
sys.path.append("scm-identify")
import numpy as np
from argparse import ArgumentParser
import os
from algorithm.utils import *
from algorithm.hierarchical_model import *
from algorithm.get_causal_graph import *
from evaluation.metrics import *
from RLCD_alg import AdjToGraph
from utils.GraphDrawer import printGraph


def save_graph(name, adj, vars, save_path):
    adj = adj * -1
    for i, row in enumerate(adj):
        for j, val in enumerate(row):
            if val == -1:
                adj[j, i] = 1
    dim = adj.shape[0]
    padded_vars = []
    for i in range(dim):
        if i < (dim - len(vars)):
            padded_vars.append(f"L{i}")
        else:
            padded_vars.append(vars[i - (dim - len(vars))])
    dot_graph = AdjToGraph(adj, padded_vars)
    print(dot_graph.dirEdges)
    printGraph(dot_graph, f'{save_path}{name}_VAE.png')
    return

def save_adjacency(matrix, vars, name, save_path):
    if matrix.is_cuda:
        matrix_converted = matrix.detach().cpu().numpy()
    else:
        matrix_converted = matrix.detach().numpy()
    np.save(f"{save_path}{name}_VAE.npy", matrix_converted)
    save_graph(name, matrix_converted, vars, save_path)
    return

def run_diff(read_path, save_path, name, subset=None, index=-1):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X = np.genfromtxt(f"{read_path}{name}.csv", dtype=int, delimiter=',', skip_header=1)[0:100]
    if subset is not None:
        X = X[subset]
    vars = np.genfromtxt(f"{read_path}{name}.csv", delimiter=',', dtype=str, max_rows=1)
    if name == "mental_health_categorical_sum":
        rate = 8e-03
        l_reg = 0.08
    elif name.startswith("set"):
        rate = 0.02
        l_reg = 1e-04
    else:
        rate = 0.02
        l_reg = 1e-04
    _, adj_matrix_pred = get_graph(X, lr=rate, l_reg=l_reg)
    if index > 0:
        save_name = f"{name}_{index}"
    else:
        save_name = name
    save_adjacency(adj_matrix_pred.data, vars, save_name, save_path=save_path)
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='mode')
    _ = subparser.add_parser('all')
    args = parser.parse_args()

    synthetic_data_path = './data/synthetic/'
    synthetic_results_path = './results/differentiable/synthetic/'
    
    if args.mode == "all":
        for graph in ["balanced", "unbalanced", "set"]:
            for func in ["linear", "additive", "non_diff"]:
                run_diff(synthetic_data_path, synthetic_results_path, f"{graph}_graph_{func}")
