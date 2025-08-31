import copy
import os, random
import time
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import argparse, sys, numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter


def cdf_to_pdf(cdf_dict):
    sorted_keys = list(cdf_dict.keys())
    pdf_dict = {}

    for i in range(len(sorted_keys)):
        if i == 0:
            pdf_dict[sorted_keys[i]] = cdf_dict[sorted_keys[i]]
        else:
            pdf_dict[sorted_keys[i]] = cdf_dict[sorted_keys[i]] - cdf_dict[sorted_keys[i - 1]]
    return pdf_dict

def arg_cdf_run_length(gram, run_length_fre, his_run_length, weight_para):
    cur_gram = copy.deepcopy(gram)
    gram_knowledge = {}
    while cur_gram > 1:

        if len(his_run_length) < cur_gram-1:
            cur_gram -= 1
            continue
        search_key = tuple(his_run_length[-(cur_gram-1):])
        rl_cdf_feature = run_length_fre[cur_gram-1]

        for item in rl_cdf_feature:
            if item[0] == search_key:
                found_dict = item[1]
                pdf_distri = cdf_to_pdf(copy.deepcopy(found_dict))
                for key, value in pdf_distri.items():
                    if key>=20 or key<=2:
                        continue
                    if key not in gram_knowledge:
                        gram_knowledge[key] = [0 for _ in range(gram)]
                    gram_knowledge[key][cur_gram-1] = value
                break
        cur_gram -= 1

    rl_cdf_feature = run_length_fre[0][0][1] # gram1
    pdf_distri = cdf_to_pdf(copy.deepcopy(rl_cdf_feature))
    for key, value in pdf_distri.items():
        if key>=20 or key<=2:
            continue
        if key not in gram_knowledge:
            gram_knowledge[key] = [0 for _ in range(gram)]
        gram_knowledge[key][0] = value

    if len(gram_knowledge) == 0:
        return random.randint(1, 4), [random.random() for _ in range(gram)]

    result_gram_knowledge = {}
    for key, value in gram_knowledge.items():
        result = 0
        for idx in range(len(value)):
            result += value[idx]*weight_para[idx]
        result_gram_knowledge[key] = [value, result]

    max_key = max(result_gram_knowledge, key=lambda k: result_gram_knowledge[k][1])
    max_value, max_result = result_gram_knowledge[max_key]

    for idx in range(len(max_value)):
        max_value[idx] += random.random()

    for idx in range(len(weight_para)):
        weight_para[idx] = max_value[idx]/sum(max_value)

    return max_key, weight_para

def sampling(rl_cdf_feature):
    random_number = round(random.random(), 15)
    for key, value in rl_cdf_feature.items():
        if value >= random_number:
            if key < 20:
                return key
            return random.randint(1, 4)

def arg_cdf_pkt_length(pkt_distribution, last_tail_value):
    row_idx = min(int(last_tail_value/50), 29)
    tar_distribution = pkt_distribution[row_idx]

    i = 0
    while len(tar_distribution) == 0:
        tar_distribution = pkt_distribution[i]
        i += 1

    random_number = round(random.random(), 15)
    for key, value in tar_distribution.items():
        if value >= random_number:
            return key

def train_with_ag(x_data, y_label, saved_path, retrain=False):
    df_train = pd.DataFrame(x_data, columns=['%s' % i for i in range(x_data.shape[1])])
    df_train = df_train.iloc[:, :3000]
    df_train['Label'] = y_label
    if retrain is True or not os.path.exists(saved_path):
        hyperparameters = {
            'GBM': {},
            'FASTAI': {},
        }
        ag_predictor = TabularPredictor(label="Label", problem_type='multiclass', eval_metric='f1_weighted',
                                        path=saved_path).fit(df_train, hyperparameters=hyperparameters, time_limit=600)
    else:
        ag_predictor = TabularPredictor.load(saved_path)
    return ag_predictor

def test_with_ag(x_data, y_label, ag_predictor, need_knowledge=True):
    df_test = pd.DataFrame(x_data, columns=['%s' % i for i in range(x_data.shape[1])])
    df_test = df_test.iloc[:, :3000]

    y_pred = ag_predictor.predict(df_test)
    y_pred_prob = ag_predictor.predict_proba(df_test)

    top_20_prob_dict = {key: [] for key in range(101)}
    top_20_label_sequence = {key: [] for key in range(101)}
    if need_knowledge:
        zeros_column = pd.Series([0] * len(y_label), index=y_pred_prob.index)
        y_pred_prob.insert(0, 'zero_column', zeros_column)
        filtered_probabilities = []
        filtered_classes = []
        for i in range(len(y_label)):
            true_class = y_label.iloc[i] if isinstance(y_label, pd.Series) else y_label[i]
            filtered_prob = [y_pred_prob.iloc[i, j] for j in range(y_pred_prob.shape[1]) if j != true_class]
            filtered_class = [j for j in range(y_pred_prob.shape[1]) if j != true_class]
            filtered_probabilities.append(filtered_prob)
            filtered_classes.append(filtered_class)

        prob_dict = np.zeros((101, 100), dtype=float)
        label_sequence = [[] for _ in range(101)]
        for real_label, probs, classes in zip(y_label, filtered_probabilities, filtered_classes):
            prob_dict[real_label] += probs
            if len(label_sequence[real_label]) != 0:
                if label_sequence[real_label] != classes:
                    print("error!")
                    sys.exit(0)
            else:
                label_sequence[real_label] = classes
        top_20_prob_dict = {key: [] for key in range(101)}
        top_20_label_sequence = {key: [] for key in range(101)}
        for real_label in range(1, 101):
            if len(prob_dict[real_label]) > 0:
                top_20_indices = np.argsort(prob_dict[real_label])[-20:][::-1]
                top_20_probs = prob_dict[real_label][top_20_indices]
                top_20_classes_ = np.array(label_sequence[real_label])[top_20_indices]
                top_20_probs_normalized = top_20_probs / np.sum(top_20_probs)
                top_20_prob_dict[real_label] = top_20_probs_normalized.tolist()
                top_20_label_sequence[real_label] = top_20_classes_.tolist()
    return y_label, y_pred, [top_20_prob_dict, top_20_label_sequence]

def fresh_mtu(flow_length, pos, older_mtu):
    if pos <= 20:
        new_list = [i for i in flow_length[pos:pos+20] if i<0]
        if len(new_list) == 0:
            return older_mtu
        else:
            return max(new_list, key=abs)

    elif len(flow_length)-pos <= 20:
        new_list =[i for i in flow_length[-20:] if i<0]
        if len(new_list) == 0:
            return older_mtu
        else:
            return max(new_list, key=abs)
    else:
        new_list = [i for i in flow_length[pos-20:pos+20] if i<0]
        if len(new_list) == 0:
            return older_mtu
        else:
            return max(new_list, key=abs)

def mocking_sequence_generation(cls, probs):
    label_sequence_weight = []
    for prob in probs:
        random_value = random.uniform(0, 1)
        weight_reciprocal = 1 / prob
        label_sequence_weight.append(random_value ** weight_reciprocal)

    combined = list(zip(label_sequence_weight, cls))

    combined.sort(key=lambda x: x[0], reverse=True)
    label_sequence_weight, cls = zip(*combined)
    cls = list(cls)
    return cls

def mocking_target(y_real, mocking_label_knowledge):
    target_label = []
    top_20_prob_dict, top_20_label_sequence = mocking_label_knowledge[0], mocking_label_knowledge[1]
    mocking_sequence = [[] for i in range(len(top_20_prob_dict))]

    for each_label in y_real:
        if len(mocking_sequence[each_label]) == 0:
            mocking_sequence[each_label] = mocking_sequence_generation(top_20_label_sequence[each_label], top_20_prob_dict[each_label])
        target_label.append(mocking_sequence[each_label][0])
        mocking_sequence[each_label].pop(0)
    return target_label


def packet_manipulation(Monitor_data, target_labels, pkt_cdf, runlen_cdf, max_n_gram, overhead_tol):
    from tqdm import tqdm
    perturbed_data = []
    his_run_length = []
    weight_para = [random.random() for _ in range(max_n_gram)]

    for perturbing_flow, target_label in tqdm(zip(Monitor_data, target_labels), total=len(Monitor_data)):
        flow_mtu = max(perturbing_flow[:, 1], key=abs)
        flow_run_length_now, flow_run_length_expected = 0, random.randint(1, 4)
        overhead = sum(abs(perturbing_flow[:, 1]))
        last_s2c_len = 0

        perturbed_flow = []

        if True:
            pkt_dis_cdf = pkt_cdf[target_label - 1]
            runlen_dis_cdf = runlen_cdf[target_label - 1]
            for pkt_pos, pkt in enumerate(perturbing_flow):
                if pkt[1] > 0:
                    perturbed_flow.append([pkt[0],pkt[1]])

                    continue
                if pkt[1] == 0:
                    continue
                flow_mtu = fresh_mtu(perturbing_flow[:, 1], pkt_pos, flow_mtu)

                if pkt[1] > flow_mtu:
                    if flow_run_length_now < flow_run_length_expected:
                        perturbed_flow.append([pkt[0], flow_mtu])

                        flow_run_length_now += 1
                    else:
                        last_s2c_len = pkt[1]
                        perturbed_flow.append([pkt[0], pkt[1]])

                        flow_run_length_now = 0
                        his_run_length.append(flow_run_length_expected)
                        flow_run_length_expected, weight_para = arg_cdf_run_length(max_n_gram, runlen_dis_cdf, his_run_length, weight_para)
                elif pkt[1] <= flow_mtu:
                    if flow_run_length_now < flow_run_length_expected:
                        flow_run_length_now += 1
                        perturbed_flow.append([ pkt[0], pkt[1] ])
                    else:
                        inserted_packet_length = arg_cdf_pkt_length(pkt_dis_cdf, last_s2c_len)
                        perturbed_flow.append([pkt[0], inserted_packet_length])
                        last_s2c_len = perturbed_flow[-1][1]
                        perturbed_flow.append([ pkt[0], pkt[1]] )
                        flow_run_length_now = 1
                        his_run_length.append(flow_run_length_expected)
                        flow_run_length_expected, weight_para = arg_cdf_run_length(max_n_gram, runlen_dis_cdf, his_run_length, weight_para)

        while len(perturbed_flow) < 3000:
            perturbed_flow.append([0, 0])

        overhead2 = sum(abs(np.array(perturbed_flow)[:, 1]))
        overhead_tol.append([overhead, overhead2])
        perturbed_data.append(perturbed_flow[:3000])
    return perturbed_data


def fits(mytrain_data, mytest_data, dataset, Experiment_Number, max_n_gram, discriminator_mode, ds_randomseed, maximum_round=10):
    Monitor_data_for_mytrain, Monitor_label_for_mytrain = mytrain_data[2], mytrain_data[3]
    Unmonitor_data_for_mytest, Unmonitor_label_for_mytest, Monitor_data_for_mytest, Monitor_label_for_mytest = \
        mytest_data[0], mytest_data[1], mytest_data[2], mytest_data[3]

    Log_overhead, Log_acc, Log_seed = [], [], []
    if dataset == "conn":
        seed_list = [[56, 65], [75, 44], [75, 44], [75, 29], [75, 19], [75, 82], [75, 68], [75, 76], [75, 55], [75, 20]] # conn
    elif dataset == "mconn":
        seed_list = [[56, 65], [75, 10], [75, 63], [75, 9], [75, 99], [75, 12], [75, 63], [75, 29], [75, 95], [75, 73]]  # mconn
    elif dataset == "brows":
        seed_list = [[56, 65], [75, 44], [75, 44], [75, 29], [75, 19], [75, 82], [75, 68], [75, 76], [75, 55], [75, 20]] # conn

    for Round in range(maximum_round):
        seed, seed2 = seed_list[Round][0], seed_list[Round][1]
        if os.path.exists("%s_%s/%s/OpenWorld/test_%d.npz" %(dataset, ds_randomseed, Experiment_Number, Round)):
            print("Skip")
            continue
        elif Round > 0:
            print("Loading")
            with np.load('%s_%s/%s/Manipulations/train_mytrain_%d_manipulations.npz' %(dataset, ds_randomseed, Experiment_Number, (Round-1)), allow_pickle=True) as npzdata:
                Monitor_data_for_mytrain = npzdata['data']
            with np.load('%s_%s/%s/Manipulations/train_mytest_%d_manipulations.npz' %(dataset, ds_randomseed, Experiment_Number, (Round-1)), allow_pickle=True) as npzdata:
                Monitor_data_for_mytest = npzdata['data']
        else:
            print("Start from scratch")

        #### AppSniffer
        if discriminator_mode == "ag":
            if not os.path.exists("%s_%s/%s/ag" %(dataset, ds_randomseed, Experiment_Number)):
                os.makedirs("%s_%s/%s/ag" %(dataset, ds_randomseed, Experiment_Number))

            ag_predictor = train_with_ag(Monitor_data_for_mytrain[:, :, 1], Monitor_label_for_mytrain, "%s_%s/%s/ag/baseline_%d" %(dataset, ds_randomseed, Experiment_Number, Round), retrain=False)
            y_real, y_pred, mocking_label_knowledge = test_with_ag(Monitor_data_for_mytrain[:, :, 1], Monitor_label_for_mytrain, ag_predictor)
        elif discriminator_mode == "nt":
            from neutic_plugin.neutic_plugin import train_with_nt, test_with_nt
            if not os.path.exists("%s_%s/%s/nt" %(dataset, ds_randomseed, Experiment_Number)):
                os.makedirs("%s_%s/%s/nt" %(dataset, ds_randomseed, Experiment_Number))
            model_path = "%s_%s/%s/nt/baseline_%d.pkl" %(dataset, ds_randomseed, Experiment_Number, Round)
            train_with_nt(Monitor_data_for_mytrain[:, :, 1], Monitor_label_for_mytrain, model_path)
            y_real, y_pred, mocking_label_knowledge = test_with_nt(Monitor_data_for_mytrain[:, :, 1], Monitor_label_for_mytrain, model_path)
        else:
            print("error")
            sys.exit(0)

        tp = sum(1 for i, j in zip(y_pred, y_real) if int(j) != 0 and int(i) == int(j))
        total = sum(1 for j in y_real if int(j) != 0)
        print(f"Accuracy: {tp / total * 100:.2f}% ({tp}/{total}) {tp} {total}")
        target_label_for_mytrain = mocking_target(y_real, mocking_label_knowledge) ### select target

        if discriminator_mode == "ag":
            y_real, y_pred, _ = test_with_ag(Monitor_data_for_mytest[:, :, 1], Monitor_label_for_mytest, ag_predictor, need_knowledge=False)
        elif discriminator_mode == "nt":
            y_real, y_pred, _ = test_with_nt(Monitor_data_for_mytrain[:, :, 1], Monitor_label_for_mytrain, model_path, need_knowledge=False)

        tp = sum(1 for i, j in zip(y_pred, y_real) if int(j) != 0 and int(i) == int(j))
        total = sum(1 for j in y_real if int(j) != 0)
        print(f" Accuracy: {tp / total * 100:.2f}% ({tp}/{total}) {tp} {total}" )
        target_label_for_mytest = mocking_target(y_real, mocking_label_knowledge)

        pkt_cdf, runlen_cdf = [], []
        from tqdm import tqdm
        for label_name in tqdm(range(1, 101)):
            label_flow = Monitor_data_for_mytrain[Monitor_label_for_mytrain == label_name]
            s2c_length = [[int(item[1]) for item in sublist if item[1] < 0] for sublist in label_flow]
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
                if len(run_length_feature) == 0:
                    print(n_gram)
                    total_rl.append(run_length_fre)
                    continue

                for src_run_length in list(set(map(tuple, run_length_feature[:, :-1]))):
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
            runlen_cdf.append(total_rl)

            pkt_number = [{} for _ in range(30)]
            for idx in range(0, len(s2c_pkt)-1):
                row_idx = min(int(s2c_pkt[idx]/50), 29)
                if s2c_pkt[idx+1] not in pkt_number[row_idx]:
                    pkt_number[row_idx][s2c_pkt[idx+1]] = 0
                pkt_number[row_idx][s2c_pkt[idx + 1]] += 1

            from itertools import accumulate
            cdf_pkt_number = [{} for _ in range(30)]
            for i, row in enumerate(pkt_number):
                total_count = sum(row.values())
                if total_count > 0:
                    probabilities = {key: value / total_count for key, value in row.items()}
                    sorted_keys = sorted(probabilities.keys())
                    cdf_values = list(accumulate(probabilities[key] for key in sorted_keys))
                    cdf_pkt_number[i] = {key: cdf for key, cdf in zip(sorted_keys, cdf_values)}
            pkt_cdf.append(cdf_pkt_number)

        ### Packet Manipulation
        overhead_tol_mytrain = []
        perturbed_data_for_mytrain = packet_manipulation(Monitor_data_for_mytrain, target_label_for_mytrain, pkt_cdf, runlen_cdf, max_n_gram, overhead_tol_mytrain)
        perturbed_data_for_mytrain = np.array(perturbed_data_for_mytrain)
        overhead_tol_mytrain = np.array(overhead_tol_mytrain)
        print("Training Overhead", (sum(overhead_tol_mytrain[:, 1])-sum(overhead_tol_mytrain[:, 0]))/(sum(overhead_tol_mytrain[:, 0])))

        overhead_tol_mytest = []
        perturbed_data_for_mytest = packet_manipulation(Monitor_data_for_mytest, target_label_for_mytest, pkt_cdf, runlen_cdf, max_n_gram, overhead_tol_mytest)
        perturbed_data_for_mytest = np.array(perturbed_data_for_mytest)
        overhead_tol_mytest = np.array(overhead_tol_mytest) # 计算带宽开销
        print("Testing Overhead", (sum(overhead_tol_mytest[:, 1])-sum(overhead_tol_mytest[:, 0]))/(sum(overhead_tol_mytest[:, 0])))

        if not os.path.exists('%s_%s/%s/Manipulations' %(dataset, ds_randomseed, Experiment_Number)):
            os.makedirs('%s_%s/%s/Manipulations' %(dataset, ds_randomseed, Experiment_Number))
        np.savez('%s_%s/%s/Manipulations/train_mytrain_%d_manipulations' % (dataset, ds_randomseed, Experiment_Number, Round), data=perturbed_data_for_mytrain, labels=Monitor_label_for_mytrain)
        np.savez('%s_%s/%s/Manipulations/train_mytest_%d_manipulations' % (dataset, ds_randomseed, Experiment_Number, Round), data=perturbed_data_for_mytest, labels=Monitor_label_for_mytest)

        perturbed_data_remain, perturbed_data_train, perturbed_label_remain, perturbed_label_train = train_test_split(
            perturbed_data_for_mytest, Monitor_label_for_mytest, test_size=0.8, stratify=Monitor_label_for_mytest, random_state=seed)
        valid_data, test_data, valid_label, test_label = train_test_split(
            perturbed_data_remain, perturbed_label_remain, test_size=0.5, stratify=perturbed_label_remain, random_state=seed2)

        if discriminator_mode == "ag":
            ag_predictor2 = train_with_ag(perturbed_data_train[:, :, 1], perturbed_label_train, "%s_%s/%s/ag/retrain_%s" % (dataset, ds_randomseed, Experiment_Number, Round), retrain=False)
            y_test2, y_pred2, _ = test_with_ag(perturbed_data_remain[:, :, 1], perturbed_label_remain, ag_predictor2, need_knowledge=False)
        elif discriminator_mode == "nt":
            model_path = "%s_%s/%s/nt/retrain_%s" %(dataset, ds_randomseed, Experiment_Number, Round)
            train_with_nt(perturbed_data_train[:,:,1], perturbed_label_train, model_path, retrain=False)
            y_test2, y_pred2, _ = test_with_nt(perturbed_data_remain[:,:,1], perturbed_label_remain, model_path, need_knowledge=False)

        tp = sum(1 for i, j in zip(y_pred2, y_test2) if int(j) != 0 and int(i) == int(j))
        total = sum(1 for j in y_test2 if int(j) != 0)
        print(f"Accuracy: {tp / total * 100:.2f}% ({tp}/{total})")

        # Next Round
        Log_overhead.append((sum(overhead_tol_mytest[:, 1])-sum(overhead_tol_mytest[:, 0]))/(sum(overhead_tol_mytest[:, 0])))
        Log_acc.append(tp / total * 100)
        Log_seed.append([seed, seed2])

        file_path = "%s_%s/%s/log.txt" % (dataset, ds_randomseed, Experiment_Number)
        with open(file_path, 'a') as file:
            file.write("Round: " + str(Round) + "\n")
            file.write("Overhead: " + str(Log_overhead) + "\n")
            file.write("acc: " + str(Log_acc) + "\n")
            file.write("seed: " + str(Log_seed) + "\n\n")

        # ClosedWorld
        if not os.path.exists('%s_%s/%s/ClosedWorld' % (dataset, ds_randomseed, Experiment_Number)):
            os.makedirs('%s_%s/%s/ClosedWorld' % (dataset, ds_randomseed, Experiment_Number))
        np.savez('%s_%s/%s/ClosedWorld/train_%d' % (dataset, ds_randomseed, Experiment_Number, Round),
                 data=perturbed_data_train[:, :, 1], time=perturbed_data_train[:, :, 0],
                 labels=perturbed_label_train)
        np.savez('%s_%s/%s/ClosedWorld/valid_%d' % (dataset, ds_randomseed, Experiment_Number, Round),
                 data=valid_data[:, :, 1], time=valid_data[:, :, 0], labels=valid_label)
        np.savez('%s_%s/%s/ClosedWorld/test_%d' % (dataset, ds_randomseed, Experiment_Number, Round),
                 data=test_data[:, :, 1], time=test_data[:, :, 0], labels=test_label)

        # OpenWorld
        Openworld_data = np.concatenate([perturbed_data_for_mytest, Unmonitor_data_for_mytest], axis=0)
        Openworld_label = np.concatenate([Monitor_label_for_mytest, Unmonitor_label_for_mytest], axis=0)
        if not os.path.exists('%s_%s/%s/OpenWorld' % (dataset, ds_randomseed, Experiment_Number)):
            os.makedirs('%s_%s/%s/OpenWorld' % (dataset, ds_randomseed, Experiment_Number))
        Openworld_data_remain, Openworld_data_train, Openworld_label_remain, Openworld_label_train = train_test_split(
            Openworld_data, Openworld_label, test_size=0.8, stratify=Openworld_label, random_state=seed)
        Openworld_data_valid, Openworld_data_test, Openworld_label_valid, Openworld_label_test = train_test_split(
            Openworld_data_remain, Openworld_label_remain, test_size=0.5, stratify=Openworld_label_remain,
            random_state=seed2)
        np.savez('%s_%s/%s/OpenWorld/train_%d' % (dataset, ds_randomseed, Experiment_Number, Round),
                 data=Openworld_data_train[:, :, 1], time=Openworld_data_train[:, :, 0],
                 labels=Openworld_label_train)
        np.savez('%s_%s/%s/OpenWorld/valid_%d' % (dataset, ds_randomseed, Experiment_Number, Round),
                 data=Openworld_data_valid[:, :, 1], time=Openworld_data_valid[:, :, 0],
                 labels=Openworld_label_valid)
        np.savez('%s_%s/%s/OpenWorld/test_%d' % (dataset, ds_randomseed, Experiment_Number, Round),
                 data=Openworld_data_test[:, :, 1], time=Openworld_data_test[:, :, 0], labels=Openworld_label_test)
