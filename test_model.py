import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F


from util.data import *
from util.preprocess import *



def test_model(model, dataloader):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    att_weight_list = []
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        with torch.no_grad():
            predicted, att_weight_, edge_index_  = model(x, edge_index)
            predicted = predicted.float().to(device)
            
            
            loss = loss_func(predicted, y)
            

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        # 예를 들어 heads=1이면, squeeze 해서 사용
        att_weight_ = att_weight_.squeeze(-1)  # (num_edges, heads)

        # 하나의 (source, target, attention) 튜플 만들기
        edge_source = edge_index_[0]  # source nodes
        edge_target = edge_index_[1]  # target nodes
        attention_value = att_weight_[:, 0]  # head=0이라고 가정할 때
        
        # (num_edges, 3) 형태로 만들기
        edge_attention = torch.stack([edge_source, edge_target, attention_value], dim=1)  # (num_edges, 3)

        # 그리고 이걸 리스트에 저장
        att_weight_list.append(edge_attention.cpu())
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))


    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)

    all_edge_attention = torch.cat(att_weight_list, dim=0)  # (전체 edge 수, 3)

    df = pd.DataFrame(all_edge_attention.numpy(), columns=['source', 'target', 'attention'])

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list], df




