import pandas as pd
import numpy as np
import sys, os
sys.path.append("./scm-identify")
from StructureLearning.RLCD.RLCD_alg import RLCD
from DGM.DataModel import DataModel
from utils.Chi2RankTest import Chi2RankTest
from PermutationTest import PermutationTest
from utils.GraphDrawer import printGraph
from utils.utils import set_random_seed
from argparse import ArgumentParser
import glob


def load_data(path):
    df=pd.read_csv(path)
    dgm = DataModel(df, df)
    return dgm 

def run_rlcd(dgm, alpha, name, plots_save_path, mask, test_method='chisq', stage_one_method="all", n=-1):
    rank_test_N_scaling = 1

    if test_method == 'permutation':
        df_x, df_v = dgm.generate_data(N=100, normalized=False)
        test = PermutationTest(df_x.to_numpy(), mask, clipping=True)
    else:
        df_x, df_v = dgm.generate_data(N=n, normalized=True)
        test = Chi2RankTest(df_x.to_numpy(), rank_test_N_scaling)

    input_parameters = {
        "ranktest_method": test,
        "citest_method": None,
        "stage1_method": stage_one_method,
        "alpha_dict": {0: alpha, 1: alpha, 2: alpha, 3: alpha},
        "stage1_ges_sparsity": 2,
        "stage1_partition_thres": 3
    }
    result_dotgraph, _, _, _ = RLCD(True, dgm.xvars, df_x, input_parameters)

    if not os.path.exists(plots_save_path):
        os.makedirs(plots_save_path)
    
    printGraph(result_dotgraph, f'{plots_save_path}/{name}__alpha{alpha}_{test_method}.png')
    return

def run_all(graph_prefix, test_type, alpha=0.05):
    synthetic_dir = "data/synthetic/"
    linear = load_data(synthetic_dir + f"{graph_prefix}_graph_linear.csv")
    additive = load_data(synthetic_dir + f"{graph_prefix}_graph_additive.csv")
    non_diff = load_data(synthetic_dir + f"{graph_prefix}_graph_non_diff.csv")
    
    results_dir = './results/rlcd/synthetic'
    mask = np.array([False for _ in linear.xvars])
    run_rlcd(linear, alpha, f"{graph_prefix}_graph_linear", results_dir, test_method=test_type, mask=mask)
    run_rlcd(additive, alpha, f"{graph_prefix}_graph_additive", results_dir, test_method=test_type, mask=mask)
    run_rlcd(non_diff, alpha, f"{graph_prefix}_graph_non_diff", results_dir, test_method=test_type, mask=mask)

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest='mode')
    _ = subparser.add_parser('all')
    _ = subparser.add_parser('0')
    _ = subparser.add_parser('1')
    _ = subparser.add_parser('2')
    parser.add_argument('-t', '--test_method')
    args = parser.parse_args()

    if not os.path.exists("./results/rlcd/synthetic"):
        os.makedirs("./results/rlcd/synthetic")

    set_random_seed(12345)

    if args.mode == '0' or args.mode == 'all': 
        run_all("set", args.test_method)
    if args.mode == '1' or args.mode == 'all': 
        run_all("balanced", args.test_method)
    if args.mode == '2' or args.mode == 'all': 
        run_all("unbalanced", args.test_method)