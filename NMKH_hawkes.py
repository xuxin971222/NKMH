import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


class Hawkesdata(Dataset):
    def __init__(self, features, labels, clk_num, neb_item, last_item):
        super(Hawkesdata, self).__init__()
        self.features = features
        self.len = len(self.features)
        self.labels = labels
        self.clk_num = clk_num
        self.neb_item = neb_item
        self.last_item = last_item

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        session = self.features
        labels = self.labels
        clk_num = self.clk_num
        neb_item = self.neb_item
        last_item = self.last_item

        return session, labels, clk_num, neb_item, last_item

# 578 128 572


class HawkesNet(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(HawkesNet, self).__init__()
        self.hidden_size = hidden_num
        self.embed_item = nn.Embedding(input_num, 128)
        self.embed_clk_num = nn.Embedding(150000, 128)
        self.GRU_layer = nn.GRU(128, hidden_size=hidden_num, batch_first=True, num_layers=1)
        self.linear = nn.Linear(hidden_num, output_num)
        self.linear1 = nn.Linear(hidden_num, 1)
        self.hidden = None
        self.a = nn.Parameter(torch.FloatTensor([0]))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.c = nn.Parameter(torch.FloatTensor([0]))
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, x, clk_num, neb_item, last_item):
        embed_session = self.embed_item(x)
        embed_clk_num = self.embed_clk_num(clk_num)
        embed_neb_item = self.embed_item(neb_item)
        embed_last_item = self.embed_item(last_item)
        neb = embed_neb_item.sum(dim=1)/72
        last = embed_last_item.sum(dim=1)/5

        a = self.a
        a = torch.tanh(a)
        b = self.b
        b = torch.sigmoid(b)
        c = self.c
        c = torch.sigmoid(c)
        clk_num = torch.sigmoid(embed_clk_num)
        intensity = clk_num + a
        neb_short = b * last + c * neb

        x, hidden = self.GRU_layer(embed_session)

        h = x[:, -1, :]
        h = h + neb_short
        h1 = F.softplus(h)
        h1 = h1 + intensity
        h1 = self.linear(h1)
        return h1


def loaddata():
    datas = open('neb.csv', 'r').readlines()
    new_neb = {}
    for data in datas:
        data = data.split('\n')[0].split('\t')
        user = int(data[0])
        nebs = data[1].split(';')

        user_neb = []
        for neb in nebs:
            neb = neb.split(',')
            temp_neb = []
            for i in neb:
                temp_neb.append(int(i))
            user_neb.append(temp_neb)
        new_neb[user] = user_neb
    print('-------------1---------------')# 读取邻居

    neb_x = {}
    n = 0
    for i in new_neb:
        if len(new_neb[i]) > 4:
            neb_x[i] = []
            for j in new_neb[i]:
                neb_x[i].append(j)
                n += 1
                if n > 4:
                    n = 0
                    break
    session_key = list(neb_x.keys())
    print('-------------2---------------')# 邻居5个

    session_dict = {}
    datas = open('session_dict.csv', 'r').readlines()
    for data in datas:
        data = data.split('\n')[0].split('\t')
        datas = data[0].split('  ')
        user = int(datas[0])
        nebs = list(eval(datas[1]))
        session_dict[user] = nebs
    print('-------------3---------------')# 读取所有session

    last_session = []
    for i in session_key:
        last_session.append(session_dict[i])
    print('-------------4---------------')# 最终session

    labels = []
    for pos, j in enumerate(last_session):
        if len(j) < 15:
            last_session[pos] = [0] * (15 - len(j)) + last_session[pos]
        labels.append(j[-1])
    print('-------------5---------------')# 补0后session以及label

    for pos, j in enumerate(last_session):
        last_session[pos] = j[0:14]

    neb_item_dict = {}
    neb_item_datas = open('neb_item_dict.csv', 'r').readlines()
    for data in neb_item_datas:
        data = data.split('\n')[0].split('\t')
        neb_item_datas = data[0].split('  ')
        user = int(neb_item_datas[0])
        nebs = list(eval(neb_item_datas[1]))
        neb_item_dict[user] = nebs
    print('-------------6---------------')
    neb_item = list(neb_item_dict.values())

    for pos, j in enumerate(neb_item):
        if len(j) < 72:
            neb_item[pos] = [0] * (72 - len(j)) + neb_item[pos]

    last_item_dict = {}
    last_item_datas = open('last_item_dict.csv', 'r').readlines()
    for data in last_item_datas:
        data = data.split('\n')[0].split('\t')
        last_item_datas = data[0].split('  ')
        user = int(last_item_datas[0])
        nebs = list(eval(last_item_datas[1]))
        last_item_dict[user] = nebs
    print('-------------7---------------')
    last_item = list(last_item_dict.values())

    clk = pd.read_csv('oth_new_clk_num.csv', header=None, names=['click'], usecols=[0],
                             dtype={0: np.int})
    clk_num = list(clk['click'])
    test_clk_num = clk_num[60000:]
    train_clk_num = clk_num[:60000]

    test_session_list = last_session[60000:]
    train_session_list = last_session[:60000]

    test_labels = labels[60000:]
    train_labels = labels[:60000]

    test_neb_item = neb_item[60000:]
    train_neb_item = neb_item[:60000]

    test_last_item = last_item[60000:]
    train_last_item = last_item[:60000]
    print('____数据集划分完成____')

    print('____数据处理完成____')
    return train_session_list, train_labels, test_session_list, test_labels, train_clk_num, test_clk_num, test_neb_item, train_neb_item, test_last_item, train_last_item


def hit(labels, indices):
    hit_number = 0
    for label, index_list in zip(labels, indices):
        if label.data in index_list:
            hit_number += 1
    return hit_number


def ndcg(labels, indices):
    user_ndcg = 0
    for label, index_list in zip(labels, indices):
        if label in index_list:
           index_list = index_list.tolist()
           index = index_list.index(label)
           user_ndcg += np.reciprocal(np.log2(index + 2))
    return user_ndcg


def metrics(model, test_loader, top_k):
    HR = 0
    NDCG = 0
    N = 0
    for session, labels, clk_num, neb_item, last_item in test_loader:
        session = session.cuda()
        labels = labels.cuda()
        clk_num = clk_num.cuda()
        neb_item = neb_item.cuda()
        last_item = last_item.cuda()
        predictions = model(session, clk_num, neb_item, last_item)
        _, indices = torch.topk(predictions, top_k)
        # print(indices)
        # recommends = torch.take(session, indices).cpu().numpy().tolist()
        HR += hit(labels, indices)
        NDCG += ndcg(labels, indices)
        N += predictions.shape[0]
    HR = HR / N
    NDCG = NDCG / N
    return HR, NDCG


if __name__ == '__main__':
    train_session_list, train_labels, test_session_list, test_labels, train_clk_num, test_clk_num, test_neb_item, train_neb_item, test_last_item, train_last_item = loaddata()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.benchmark = True

    train_session_tensor = torch.LongTensor(train_session_list)
    test_session_tensor = torch.LongTensor(test_session_list)
    train_labels_tensor = torch.LongTensor(train_labels)
    test_labels_tensor = torch.LongTensor(test_labels)
    train_clk_num_tensor = torch.LongTensor(train_clk_num)
    test_clk_num_tensor = torch.LongTensor(test_clk_num)

    train_neb_item_tensor = torch.LongTensor(train_neb_item)
    test_neb_item_tensor = torch.LongTensor(test_neb_item)
    train_last_item_tensor = torch.LongTensor(train_last_item)
    test_last_item_tensor = torch.LongTensor(test_last_item)

    train_data = torch.utils.data.TensorDataset(train_session_tensor, train_labels_tensor, train_clk_num_tensor, train_neb_item_tensor, train_last_item_tensor)
    test_data = torch.utils.data.TensorDataset(test_session_tensor, test_labels_tensor, test_clk_num_tensor, test_neb_item_tensor, test_last_item_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=2)
    print('____数据加载完成____')
    print("%20s%20s%20s%20s" % ('epoch', 'HitRatio', 'NDCG', 'epoch_loss'))
    model = HawkesNet(52740, 128, 52740).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_function = nn.BCEWithLogitsLoss()
    loss_function = nn.CrossEntropyLoss()
    best_HR, best_NDCG = 0.0, 0.0
    for epoch in range(1000):
        epoch_loss = 0
        model.train()
        for session, labels, clk_num, neb_item, last_item in train_loader:
            session = session.cuda()
            labels = labels.cuda()
            clk_num = clk_num.cuda()
            neb_item = neb_item.cuda()
            last_item = last_item.cuda()
            model.zero_grad()
            prediction = model(session, clk_num, neb_item, last_item)
            loss = loss_function(prediction, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += np.array(loss.cpu().detach())
        model.eval()
        HR, NDCG = metrics(model, test_loader, top_k=10)
        if HR > best_HR:
            best_HR = HR
        if NDCG > best_NDCG:
            best_NDCG = NDCG
        print("%20d%20.6f%20.6f%20.6f" % (epoch, best_HR, best_NDCG, epoch_loss))
