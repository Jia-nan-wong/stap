import argparse, sys, numpy as np
import copy
import os.path, random
from data_utils import *
from calculate import fits


def go(Experiment_Number, dataset, n_gram, discriminator_mode, ds_randomseed):
    input_size, num_classes = 3000, 101
    if dataset == "conn":
        X_data_for_mytrain, Y_data_for_mytrain = load_conn_dataset(input_size, num_classes, ds_randomseed, mode="total")
        X_data_for_mytest, Y_data_for_mytest = load_conn_dataset_mytest(input_size, num_classes, ds_randomseed, mode="total")
    elif dataset == "mconn":
        X_data_for_mytrain, Y_data_for_mytrain = load_mconn_dataset(input_size, num_classes, ds_randomseed, mode="total")
        X_data_for_mytest, Y_data_for_mytest = load_mconn_dataset_mytest(input_size, num_classes, ds_randomseed, mode="total")
    elif dataset == "brows":
        X_data_for_mytrain, Y_data_for_mytrain = load_brows_dataset(input_size, num_classes, ds_randomseed, mode="total")
        X_data_for_mytest, Y_data_for_mytest = load_brows_dataset_mytest(input_size, num_classes, ds_randomseed, mode="total")

    Unmonitor_data_for_mytrain = copy.deepcopy(X_data_for_mytrain[Y_data_for_mytrain == 0])
    Unmonitor_label_for_mytrain = copy.deepcopy(Y_data_for_mytrain[Y_data_for_mytrain == 0])
    Monitor_data_for_mytrain = copy.deepcopy(X_data_for_mytrain[Y_data_for_mytrain != 0])
    Monitor_label_for_mytrain = copy.deepcopy(Y_data_for_mytrain[Y_data_for_mytrain != 0])
    mytrain_data = [Unmonitor_data_for_mytrain, Unmonitor_label_for_mytrain, Monitor_data_for_mytrain,
                    Monitor_label_for_mytrain]

    Unmonitor_data_for_mytest = copy.deepcopy(X_data_for_mytest[Y_data_for_mytest == 0])
    Unmonitor_label_for_mytest = copy.deepcopy(Y_data_for_mytest[Y_data_for_mytest == 0])
    Monitor_data_for_mytest = copy.deepcopy(X_data_for_mytest[Y_data_for_mytest != 0])
    Monitor_label_for_mytest = copy.deepcopy(Y_data_for_mytest[Y_data_for_mytest != 0])
    mytest_data = [Unmonitor_data_for_mytest, Unmonitor_label_for_mytest, Monitor_data_for_mytest,
                   Monitor_label_for_mytest]

    fits(mytrain_data, mytest_data, dataset, Experiment_Number, n_gram, discriminator_mode, ds_randomseed)


dataset = "conn" # conn mconn brows
data_split = False
n_gram = 1 # 1 2 3 4 5
discriminator_mode = "nt" # nt ag
ds_randomseed = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data', default=dataset)
    parser.add_argument('-ds', '--data_split', action='store_true', default=data_split)
    parser.add_argument('-gram', '--gram', default=n_gram)
    parser.add_argument('-dm', '--discriminator_mode', default=discriminator_mode)
    parser.add_argument('-dataseed', '--ds_randomseed', default=ds_randomseed)

    args = parser.parse_args()
    dataset = args.data
    data_split = args.data_split
    ds_randomseed = int(args.ds_randomseed)
    discriminator_mode = str(args.discriminator_mode)
    n_gram = int(args.gram)

    if data_split:
        split_datasets(dataset, ds_randomseed)
        sys.exit(1)

    experiment_number = "12-%sgram-%s" %(n_gram, discriminator_mode)
    go(experiment_number, dataset, n_gram, discriminator_mode, ds_randomseed)
