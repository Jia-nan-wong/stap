import pickle, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os, time
from collections import defaultdict, Counter

def split_dataset_one_to_two(X, y, ds_randomseed):
    [X_train, X_, y_train, y_] = train_test_split(X, y,
                                    test_size=0.5,
                                    random_state=ds_randomseed,
                                    stratify=y)
    return X_train, y_train, X_, y_

def format_data(X, y, times, input_size):
    X = X[:, :input_size]
    x_format = []
    times = times[:, :input_size]
    if len(X) != len(times):
        print("Error")
    for index in range(len(X)):
        result = np.concatenate((times[np.newaxis, index], X[np.newaxis, index]), axis=0).T
        x_format.append(result)
    x_format = np.array(x_format)
    return x_format, y

def load_conn_dataset(input_size=3000, num_classes=100, ds_randomseed=0, mode=""):
    dataset_dir = './dataset/qcsd/'
    if mode == "total":
        datafile = dataset_dir + 'conn_for_mytrain_%s.npz' %ds_randomseed
    elif mode == "train":
        datafile = dataset_dir + 'conn_train_for_mytrain.npz'
    elif mode == "valid":
        datafile = dataset_dir + 'conn_valid_for_mytrain.npz'
    elif mode == "test":
        datafile = dataset_dir + 'conn_test_for_mytrain.npz'
    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']
        times = npzdata['time'] # interval time

    X, Y = format_data(data, labels, times, input_size)
    return X, Y

def load_conn_dataset_mytest(input_size=3000, num_classes=100, ds_randomseed=0, mode=""):
    dataset_dir = './dataset/qcsd/'
    if mode == "total":
        datafile = dataset_dir + 'conn_for_mytest_%s.npz' %ds_randomseed
    elif mode == "train":
        datafile = dataset_dir + 'conn_train_for_mytest.npz'
    elif mode == "valid":
        datafile = dataset_dir + 'conn_valid_for_mytest.npz'
    elif mode == "test":
        datafile = dataset_dir + 'conn_test_for_mytest.npz'
    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']
        times = npzdata['time']

    X, Y = format_data(data, labels, times, input_size)
    return X, Y

def load_mconn_dataset(input_size=3000, num_classes=100, ds_randomseed=0, mode=""):
    dataset_dir = './dataset/qcsd/'
    if mode == "total":
        datafile = dataset_dir + 'mconn_for_mytrain_%s.npz' %ds_randomseed
    elif mode == "train":
        datafile = dataset_dir + 'mconn_train_for_mytrain.npz'
    elif mode == "valid":
        datafile = dataset_dir + 'mconn_valid_for_mytrain.npz'
    elif mode == "test":
        datafile = dataset_dir + 'mconn_test_for_mytrain.npz'

    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']
        times = npzdata['time']
    X, Y = format_data(data, labels, times, input_size)
    return X, Y

def load_mconn_dataset_mytest(input_size=3000, num_classes=100, ds_randomseed=0, mode=""):
    dataset_dir = './dataset/qcsd/'
    if mode == "total":
        datafile = dataset_dir + 'mconn_for_mytest_%s.npz' %ds_randomseed
    elif mode == "train":
        datafile = dataset_dir + 'mconn_train_for_mytest.npz'
    elif mode == "valid":
        datafile = dataset_dir + 'mconn_valid_for_mytest.npz'
    elif mode == "test":
        datafile = dataset_dir + 'mconn_test_for_mytest.npz'
    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']
        times = npzdata['time']

    X, Y = format_data(data, labels, times, input_size)
    return X, Y

def load_brows_dataset(input_size=3000, num_classes=100, ds_randomseed=0, mode=""):
    dataset_dir = './dataset/qcsd/'
    if mode == "total":
        datafile = dataset_dir + 'brows_for_mytrain_%s.npz' %ds_randomseed
    elif mode == "train":
        datafile = dataset_dir + 'brows_train_for_mytrain.npz'
    elif mode == "valid":
        datafile = dataset_dir + 'brows_valid_for_mytrain.npz'
    elif mode == "test":
        datafile = dataset_dir + 'brows_test_for_mytrain.npz'
    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']
        times = npzdata['time']

    X, Y = format_data(data, labels, times, input_size)
    return X, Y

def load_brows_dataset_mytest(input_size=3000, num_classes=100, ds_randomseed=0, mode=""):
    dataset_dir = './dataset/qcsd/'
    if mode == "total":
        datafile = dataset_dir + 'brows_for_mytest_%s.npz' %ds_randomseed
    elif mode == "train":
        datafile = dataset_dir + 'brows_train_for_mytest.npz'
    elif mode == "valid":
        datafile = dataset_dir + 'brows_valid_for_mytest.npz'
    elif mode == "test":
        datafile = dataset_dir + 'brows_test_for_mytest.npz'
    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']
        times = npzdata['time'] # interval time

    X, Y = format_data(data, labels, times, input_size)
    return X, Y

def split_datasets(dataset, ds_randomseed):
    if dataset == "conn":
        dataset_dir = './dataset/qcsd/'
        # Load data
        datafile = dataset_dir + 'conn.npz'

        with np.load(datafile, allow_pickle=True) as npzdata:
            data = npzdata['data']
            labels = npzdata['labels']
            time = npzdata['interval']

        data_expanded = np.expand_dims(data, axis=1)
        time_expanded = np.expand_dims(time, axis=1)

        merged_array = np.concatenate((data_expanded, time_expanded), axis=1)
        y = labels.copy()
        x_for_mytrain, y_for_mytrain, x_for_mytest, y_for_mytest = split_dataset_one_to_two(merged_array, y, ds_randomseed)

        X_train, time_train = [], []
        X_train_for_mytest, time_train_for_mytest = [], []

        for i in x_for_mytrain:
            X_train.append(i[0])
            time_train.append(i[1])
        for i in x_for_mytest:
            X_train_for_mytest.append(i[0])
            time_train_for_mytest.append(i[1])

        X_train = np.array(X_train)
        time_train = np.array(time_train)

        np.savez(dataset_dir + 'conn_for_mytrain_%s.npz' %ds_randomseed, data=X_train, labels=y_for_mytrain, time=time_train)

        X_train_for_mytest = np.array(X_train_for_mytest)
        time_train_for_mytest = np.array(time_train_for_mytest)
        np.savez(dataset_dir + 'conn_for_mytest_%s.npz' %ds_randomseed, data=X_train_for_mytest,
                 labels=y_for_mytest, time=time_train_for_mytest)


    elif dataset == "mconn":
        dataset_dir = './dataset/qcsd/'
        # Load data
        datafile = dataset_dir + 'mconn.npz'

        with np.load(datafile, allow_pickle=True) as npzdata:
            data = npzdata['data']
            labels = npzdata['labels']
            time = npzdata['interval']

        data_expanded = np.expand_dims(data, axis=1)
        time_expanded = np.expand_dims(time, axis=1)
        merged_array = np.concatenate((data_expanded, time_expanded), axis=1)
        y = labels.copy()
        x_for_mytrain, y_for_mytrain, x_for_mytest, y_for_mytest = split_dataset_one_to_two(merged_array, y, ds_randomseed)

        X_train, time_train = [], []
        X_train_for_mytest, time_train_for_mytest = [], []

        for i in x_for_mytrain:
            X_train.append(i[0])
            time_train.append(i[1])
        for i in x_for_mytest:
            X_train_for_mytest.append(i[0])
            time_train_for_mytest.append(i[1])

        X_train  = np.array(X_train)
        time_train = np.array(time_train)
        np.savez(dataset_dir + 'mconn_for_mytrain_%s.npz' %ds_randomseed, data=X_train, labels=y_for_mytrain, time=time_train)
        X_train_for_mytest = np.array(X_train_for_mytest)
        time_train_for_mytest = np.array(time_train_for_mytest)
        np.savez(dataset_dir + 'mconn_for_mytest_%s.npz' %ds_randomseed, data=X_train_for_mytest,
                 labels=y_for_mytest, time=time_train_for_mytest)

    elif dataset == "brows":
        dataset_dir = './dataset/qcsd/'
        # Load data
        datafile = dataset_dir + 'brows.npz'
        with np.load(datafile, allow_pickle=True) as npzdata:
            data = npzdata['data']
            labels = npzdata['labels']
            time = npzdata['interval']

        data_expanded = np.expand_dims(data, axis=1)
        time_expanded = np.expand_dims(time, axis=1)
        merged_array = np.concatenate((data_expanded, time_expanded), axis=1)
        y = labels.copy()
        x_for_mytrain, y_for_mytrain, x_for_mytest, y_for_mytest = split_dataset_one_to_two(merged_array, y, ds_randomseed)

        X_train, time_train = [], []
        X_train_for_mytest, time_train_for_mytest = [], []

        for i in x_for_mytrain:
            X_train.append(i[0])
            time_train.append(i[1])
        for i in x_for_mytest:
            X_train_for_mytest.append(i[0])
            time_train_for_mytest.append(i[1])

        X_train  = np.array(X_train)
        time_train = np.array(time_train)
        np.savez(dataset_dir + 'brows_for_mytrain_%s.npz' %ds_randomseed, data=X_train, labels=y_for_mytrain, time=time_train)

        X_train_for_mytest = np.array(X_train_for_mytest)
        time_train_for_mytest = np.array(time_train_for_mytest)
        np.savez(dataset_dir + 'brows_for_mytest_%s.npz' %ds_randomseed, data=X_train_for_mytest, labels=y_for_mytest, time=time_train_for_mytest)


def cdf_get(Monitor_data_remain, Monitor_label_remain, label_name, max_n_gram):
    label_flow = Monitor_data_remain[Monitor_label_remain == label_name]
    s2c_length = [[int(item[1]) for item in sublist if item[1] < 0] for sublist in label_flow]  # 负值
    s2c_run_length = []
    s2c_pkt = []
    for flow in s2c_length:
        flow_run_length = []
        run_length_count = 0
        for pos in range(len(flow)):
            if flow[pos] <= -1200:
                run_length_count += 1
            else:
                flow_run_length.append(run_length_count)
                run_length_count = 0
                s2c_pkt.append(flow[pos])
        s2c_run_length.append(flow_run_length)

    total_rl = []
    for n_gram in range(1, max_n_gram + 1):
        run_length_feature = np.array(
            [flow[index:index + n_gram] for flow in s2c_run_length for index in range(len(flow) - n_gram + 1)])
        run_length_fre = []
        for src_run_length in list(set(map(tuple, run_length_feature[:, :-1]))):  # 使用tuple使其成为可哈希类型
            filtered = np.array([x for x in run_length_feature if tuple(x[:-1]) == src_run_length])
            counts = Counter(filtered[:, -1])
            sorted_counts = dict(sorted(counts.items()))
            total = sum(sorted_counts.values())
            cumulative_sum = 0
            cdf = {}
            for key, value in sorted_counts.items():
                cumulative_sum += value
                cdf[key] = cumulative_sum / total
            run_length_fre.append([src_run_length, cdf])
        total_rl.append(run_length_fre)

    value_counts = Counter(s2c_pkt)
    sorted_counts = dict(sorted(value_counts.items()))
    total_counts = sum(sorted_counts.values())
    cumulative_sum_counts = 0
    pkt_distribution = {}
    for key, count in sorted_counts.items():
        cumulative_sum_counts += count
        pkt_distribution[key] = cumulative_sum_counts / total_counts
    return pkt_distribution, total_rl