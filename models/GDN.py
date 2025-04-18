import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100, model_type="GDN"):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, node_num, inter_dim=inter_dim, heads=heads, concat=False, model_type="GDN")
        
        if model_type == "GDN":
            self.bn = nn.BatchNorm1d(out_channel)
        else:
            self.bn = nn.BatchNorm1d(out_channel*2)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, temporal_x=None, time_edge_index=None, model_type=None, node_num=0, all_feature=0, batch_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True, spatial=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        if model_type == 'STGDN':
            if temporal_x is not None and time_edge_index is not None:
                # temporal_x.shape = (batch_num * all_feature, node_num)
                temporal_out = self.gnn(
                    temporal_x, time_edge_index,
                    embedding=None,
                    return_attention_weights=False,
                    spatial=False
                )

                # Combine: z_i^(t) = ReLU(spatial + temporal)
                # temporal_out을 여기서 time_window의 길이 만큼 묶고, 총 (batch_num, d) 차원만큼 나와야함
                # 그리고 이걸 out에 더할때는 배치 단위로 동일하게 여러개 만들어서 차원 맞추어서 더하기
                # out.shape은 torch.Size([1632, 64]) -> 1632 = 51(node_num)x32(batch_size)
                # temporal_out.shape은 torch.Size([160, 64]) -> 160 = 5(window_size)x32(batch_size)
                node_num = temporal_x.size(-1)
                all_feature = int(time_edge_index.size(-1)/temporal_x.size(0))
                batch_num = int(temporal_out.size(0)/all_feature)
                d = temporal_out.size(-1)

                temporal_out = temporal_out.view(batch_num, all_feature, d).mean(dim=1)
                
                temporal_out = temporal_out.unsqueeze(1).repeat(1, node_num, 1).view(batch_num * node_num, d)
                
                out = torch.cat([out, temporal_out], dim=-1)

                out = F.relu(out)
            else:
                out = F.relu(out)
        else:
            if model_type != 'GDN':
                raise ValueError("Choose model_type between [GDN/STGDN]")

        out = self.bn(out)
        
        return self.relu(out)


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20, model_type='GDN'):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]


        if model_type == "GDN":
            embed_dim = dim
        else:
            embed_dim = dim*2

        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)


        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1, node_num=node_num, model_type=model_type) for i in range(edge_set_num)
        ])


        self.node_embedding = None
        self.topk = topk
        self.model_type = model_type

        self.learned_graph = None

        if model_type == "GDN":
            self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)
        else:
            self.out_layer = OutLayer(2*dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        
        x = x.view(-1, all_feature).contiguous()
        
        temporal_x = data.clone().detach()
        temporal_x = temporal_x.permute(0, 2, 1)
        temporal_x = temporal_x.reshape(-1, node_num)

        source = torch.arange(all_feature).repeat_interleave(all_feature)
        target = torch.arange(all_feature).repeat(all_feature)
        gated_edge_index = torch.stack([source, target], dim=0)
        gated_edge_index = gated_edge_index.to(device)

        time_edge_index = get_batch_edge_index(gated_edge_index, batch_num, all_feature).to(device)

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings, temporal_x=temporal_x, time_edge_index=time_edge_index, model_type=self.model_type)

            
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)


        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
   

        return out
        