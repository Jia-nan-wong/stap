import torch, os, sys
import torch.nn as nn
from neutic_plugin.neutic.Models import NeuTic
from tqdm import tqdm
from neutic_plugin.myDataLoader import CustomDatasetPlugin
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import random
import pandas as pd

n_src_vocab = [1500, 2, 2]
h = 2000
grained = 101
num_epochs = 10

train_batch_size = 2
valid_batch_size = test_batch_size = 1

def trans(X, Y):
    input_size = 2000
    data, label = [], []
    for x, y in zip(X, Y):
        x_length, x_win, x_flag = [abs(i) for i in x][:input_size], [], []
        length = len(np.where(x != 0)[0])
        for i in range(length):
            x_win.append(1)
            x_flag.append(1)
        while len(x_length) < input_size:
            x_length.append(0)
        x_length = x_length[:input_size]
        while len(x_win) < input_size:
            x_win.append(0)
        x_win = x_win[:input_size]
        while len(x_flag) < input_size:
            x_flag.append(0)
        x_flag = x_flag[:input_size]

        data.append([x_length, x_win, x_flag])
        label.append(y)
    return data, label
def TrainToNeuTic(length_data, length_label):
    train_data, valid_data, train_label, valid_label = train_test_split(
        length_data, length_label, test_size=0.1, stratify=length_label, random_state=random.randint(1, 100))
    train_data, train_label = trans(train_data, train_label)
    valid_data, valid_label = trans(valid_data, valid_label)
    return train_data, train_label, valid_data, valid_label
def TestToNeuTic(length_data, length_label):
    length_data, length_label = trans(length_data, length_label)
    return length_data, length_label


def train_with_nt(length_data, length_label, saved_path, retrain= False):
    import os
    if retrain is False and os.path.exists(saved_path):
        print("No Training！")
    else:
        train_data, train_label, valid_data, valid_label = TrainToNeuTic(length_data, length_label)

        train_dataloader = DataLoader(CustomDatasetPlugin(train_data, train_label), batch_size=train_batch_size, shuffle=True)
        valid_dataloader = DataLoader(CustomDatasetPlugin(valid_data, valid_label), batch_size=valid_batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuTic(n_src_vocab=n_src_vocab, h=h, grained=grained).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        best_val_loss = 99999999
        patience, max_patience = 0, 3
        for epoch in range(num_epochs):
            model.train()
            count = 0
            for batch_data, batch_labels in tqdm(train_dataloader, desc="Training Progress"):
                batch_seq = batch_data[:, 0, :h].tolist()
                batch_win = batch_data[:, 1, :h].tolist()
                batch_flag = batch_data[:, 2, :h].tolist()
                batch_label = batch_labels.tolist()
                src_seq = torch.tensor([batch_seq, batch_win, batch_flag]).to(device)
                src_pos = torch.tensor(h).to(device)
                labels = torch.tensor(batch_label).to(device)
                outputs = model(src_seq, src_pos)

                loss = criterion(outputs, labels)
                loss.backward()
                if count != 0 and (count % 8 == 0 or len(train_dataloader) - 1 == count):
                    optimizer.step()
                    optimizer.zero_grad()
                count += 1

            with torch.no_grad():
                model.eval()
                val_loss = 0
                for batch_data, batch_labels in valid_dataloader:
                    batch_seq = batch_data[:, 0, :h].tolist()
                    batch_win = batch_data[:, 1, :h].tolist()
                    batch_flag = batch_data[:, 2, :h].tolist()
                    batch_label = batch_labels.tolist()
                    val_seq = torch.tensor([batch_seq, batch_win, batch_flag]).to(device)
                    val_pos = torch.tensor(h).to(device)
                    val_labels = torch.tensor(batch_label).to(device)
                    outputs = model(val_seq, val_pos)
                    _, predicted = torch.max(outputs.data, dim=1)
                    val_loss += criterion(outputs, val_labels)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    torch.save(model, saved_path)
                else:
                    patience += 1
                    if patience >= max_patience:
                        break

def test_with_nt(length_data, length_label, saved_path, need_knowledge=True):
    test_data, test_label = TestToNeuTic(length_data, length_label)
    model = torch.load(saved_path)

    test_dataloader = DataLoader(CustomDatasetPlugin(test_data, test_label), batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_true_label, test_pre_label = [], []
    y_pred_prob_list = []
    with torch.no_grad():
        model.eval()
        for batch_data, batch_labels in tqdm(test_dataloader, desc="Testing eval Progress"):
            batch_seq = batch_data[:, 0, :h].tolist()
            batch_win = batch_data[:, 1, :h].tolist()
            batch_flag = batch_data[:, 2, :h].tolist()
            batch_label = batch_labels.tolist()

            test_seq = torch.tensor([batch_seq, batch_win, batch_flag]).to(device)
            test_pos = torch.tensor(h).to(device)

            outputs = model(test_seq, test_pos)
            y_pred_prob_list.append(outputs.data.cpu().numpy())

            topk_values, topk_indices = torch.topk(outputs.data, k=2, dim=1)

            predicted = topk_indices[:, 0]

            test_true_label += batch_label
            test_pre_label += predicted.tolist()

    y_label = pd.Series(test_true_label)
    y_pred_prob = pd.DataFrame(np.concatenate(y_pred_prob_list), columns=[f'class_{i}' for i in range(outputs.shape[1])])


    top_20_prob_dict = {key: [] for key in range(101)}
    top_20_label_sequence = {key: [] for key in range(101)}

    if need_knowledge:
        y_pred_prob.iloc[:, 0] = 0

        filtered_probabilities = []
        filtered_classes = []
        for i in range(len(test_true_label)):
            true_class = test_true_label[i]
            filtered_prob = [y_pred_prob.iloc[i, j] for j in range(y_pred_prob.shape[1]) if j != true_class]
            filtered_class = [j for j in range(y_pred_prob.shape[1]) if j != true_class]
            filtered_probabilities.append(filtered_prob)
            filtered_classes.append(filtered_class)

        prob_dict = np.zeros((101, 100), dtype=float)
        label_sequence = [[] for _ in range(101)]
        for real_label, probs, classes in zip(test_true_label, filtered_probabilities, filtered_classes):
            prob_dict[real_label] += probs
            if len(label_sequence[real_label]) != 0:
                if label_sequence[real_label] != classes:
                    print("error!")
                    sys.exit(0)
            else:
                label_sequence[real_label] = classes

        top_20_prob_dict = {key: [] for key in range(101)}
        top_20_label_sequence = {key: [] for key in range(101)}
        for real_label in range(1, 101):  # 假设类别范围是 0 到 100
            if len(prob_dict[real_label]) > 0:
                top_20_indices = np.argsort(prob_dict[real_label])[-20:][::-1]
                top_20_probs = prob_dict[real_label][top_20_indices]
                top_20_classes_ = np.array(label_sequence[real_label])[top_20_indices]
                top_20_probs_normalized = top_20_probs / np.sum(top_20_probs)
                top_20_prob_dict[real_label] = top_20_probs_normalized.tolist()
                top_20_label_sequence[real_label] = top_20_classes_.tolist()

    return test_true_label, test_pre_label, [top_20_prob_dict, top_20_label_sequence]