# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train_model import train_model
from test_model  import test_model
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())


        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)


        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # generator = torch.Generator()
        # generator.manual_seed(train_config['seed'])

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        test_len = int(len(test_dataset))
        adj_use_len = int(test_len * 0.5)
        indices = torch.arange(test_len)

        adj_sub_indices = indices[:adj_use_len]
        adj_subset = Subset(test_dataset, adj_sub_indices)

        test_sub_indices = indices[:]
        test_subset = Subset(test_dataset, test_sub_indices)

        self.adj_dataloader = DataLoader(adj_subset, batch_size=train_config['batch'],
                            shuffle=False)    

        self.test_dataloader = DataLoader(test_subset, batch_size=train_config['batch'],
                            shuffle=False)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk'],
                model_type=train_config['model_type']
            ).to(self.device)


    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train_model(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result, attention = test_model(best_model, self.test_dataloader)
        _, self.val_result, _ = test_model(best_model, self.val_dataloader)
        _, self.adj_result, _ = test_model(best_model, self.adj_dataloader)

        id_to_sensor_name = {i: name for i, name in enumerate(self.feature_map)}
        source_sensor_names = [id_to_sensor_name[int(s) % len(self.feature_map)] for s in attention['source']]
        target_sensor_names = [id_to_sensor_name[int(t) % len(self.feature_map)] for t in attention['target']]

        attention['source'] = source_sensor_names
        attention['target'] = target_sensor_names

        attention.to_csv(f'./csv/{self.env_config["save_path"]}/attention_result.csv', index=False)
        
        self.get_score(self.test_result, self.val_result, self.adj_result, self.train_config['slide_win'])


    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader


    def get_top_anomalies_as_dataframe(self, anomaly_score, columns, labels, detection, top_n=10):
        """
        :param anomaly_score: Anomaly scores, type: Pandas DataFrame (rows: timestamps, columns: sensors)
        :param top_n: Number of top anomalies to extract, default is 5
        :return: Pandas DataFrame with timestamp as index and top anomalies in 1~N columns
        """
        anomaly_score.columns = columns

        results = []

        for i, row in anomaly_score.iterrows():
            top_indices = row.sort_values(ascending=False).index[:top_n]
            top_scores = row.sort_values(ascending=False).iloc[:top_n]

            result_dict = {str(j + 1): f"{col}: {round(score, 3)}" for j, (col, score) in enumerate(zip(top_indices, top_scores))}
            result_dict["timestamp"] = i
            results.append(result_dict)

        result_df = pd.DataFrame(results).set_index("timestamp")

        additional_data = pd.DataFrame()
        if labels is not None:
            additional_data['ground truth label'] = labels
        if detection is not None:
            additional_data['model prediction'] = detection

        result_df = pd.concat([additional_data, result_df], axis=1)

        return result_df


    def plot_anomaly_scores_with_threshold(self, max_anomaly_scores, threshold, save_path=None):
        time_steps = np.arange(len(max_anomaly_scores))

        plt.figure(figsize=(15, 5))
        plt.bar(time_steps, max_anomaly_scores, label='Total Anomaly Score', width=2.0)
        # plt.plot(max_anomaly_scores, label='Total Anomaly Score', linewidth=0.5)
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
        plt.xlabel('Time Step')
        plt.ylabel('Anomaly Score')
        plt.legend(loc='upper right')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


    def get_score(self, test_result, val_result, adj_result, dilation_window):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)
        np_adj_result = np.array(adj_result)

        test_labels = np_test_result[2, :, 0].tolist()
        adj_labels = np_adj_result[2, :, 0].tolist()
        # np.save('./test_labels.npy', test_labels)
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)
        ano_scores = np.array(test_scores).T
        anomaly_df = pd.DataFrame(ano_scores)

        # np.save('./test_scores.npy', test_scores)
        
        if self.env_config['report'] == 'val':
            top1_adj_info = get_best_performance_data(test_scores, adj_labels, dilation_window, topk=1)
            threshold = top1_adj_info[-1]
        
        if self.env_config['report'] == 'val':
            top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, dilation_window, topk=1, threshold=threshold)
        elif self.env_config['report'] == 'origin':
            top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, dilation_window, topk=1)
        elif self.env_config['report'] == 'best':
            top1_best_info = get_best_performance_data(test_scores, test_labels, dilation_window, topk=1) 

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info
        elif self.env_config['report'] == 'origin':
            info = top1_val_info
            threshold = top1_val_info[-1]

        print(f'F1 score: {info[0]}')
        print(f'Precision: {info[1]}')
        print(f'Recall: {info[2]}')
        print(f'Accuracy: {info[3]}')
        print(f'AUC: {info[4]}\n')

        _, result_csv_path = self.get_save_path()
        with open(result_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['F1', 'Precision', 'Recall', 'Accuracy', 'AUC'])
            writer.writerow([info[0], info[1], info[2], info[3], info[4]])

        fig_path = f'./fig/{self.env_config["save_path"]}/{self.train_config["model_type"]}/total_anomaly.png'
        Path(os.path.dirname(fig_path)).mkdir(parents=True, exist_ok=True)
        
        max_anomaly_scores = np.max(test_scores, axis=0)
        self.plot_anomaly_scores_with_threshold(max_anomaly_scores, threshold, save_path=fig_path)

        detection = [1 if score > threshold else 0 for score in max_anomaly_scores]
        detection = detection[train_config['slide_win']:]

        labels = test_labels[train_config['slide_win']:]

        rca_csv = self.get_top_anomalies_as_dataframe(anomaly_df, self.feature_map, labels, detection, top_n=3)
        csv_save_dir = f'./csv/{self.env_config["save_path"]}'
        os.makedirs(csv_save_dir, exist_ok=True)
        anomaly_df.to_csv(f'./csv/{self.env_config["save_path"]}/anomaly_score.csv', index=True, index_label="timestamp")
        rca_csv.to_csv(f'./csv/{self.env_config["save_path"]}/test_result.csv', index=True, index_label="timestamp")

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-lr', help='learning rate', type = float, default=0.001)
    parser.add_argument('-loss_func', help='loss function', type = str, default='mse')
    parser.add_argument('-early_stop_win', help='early stopping', type = int, default=15)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-model_type', help='model type [GDN/STGDN]', type = str, default='GDN')
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'loss_func' : args.loss_func,
        'lr': args.lr,
        'early_stop_win': args.early_stop_win,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'model_type': args.model_type,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()





