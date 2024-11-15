#!/usr/bin/env
# coding:utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn
from models.layers.gcn_conv_input_mat import GCNConv
from models.layers.Attention import AttentionLayer
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.glob import MaxPooling,AvgPooling
import warnings
warnings.filterwarnings("ignore")

#=========================
import numpy as np
device = 'cpu'

import scipy.sparse as spp
import dgl

def default_loader(cpath, pid, embed_data):
    cmap_data = np.load(cpath)
    nodenum = len(str(cmap_data['seq']))
    cmap = cmap_data['contact']
    g_embed = torch.tensor(embed_data[pid][:nodenum]).float().to(device)

    adj = spp.coo_matrix(cmap)
    G = dgl.DGLGraph(adj).to(device)
    G = G.to(torch.device('cuda'))
    G.ndata['feat'] = g_embed

    if nodenum > 1000:
        textembed = embed_data[pid][:1000]
    elif nodenum < 1000:
        textembed = np.concatenate((embed_data[pid], np.zeros((1000 - nodenum, 1024))))

    textembed = torch.tensor(textembed).float().to(device)
    return G, textembed
#=========================


#9.18xiugai
class MyGATModel(nn.Module):
    def __init__(self, args):
        super(MyGATModel, self).__init__()
        self.embedding_size = args['emb_dim']
        self.output_dim = args['output_dim']

        # 定义GATConv层
        self.gat1 = GATConv(in_feats=self.embedding_size, out_feats=self.embedding_size, num_heads=3)
        self.gat2 = GATConv(in_feats=self.embedding_size * 3, out_feats=self.embedding_size * 3, num_heads=3)
        self.gat3 = GATConv(in_feats=self.embedding_size * 9, out_feats=self.embedding_size * 9, num_heads=1)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.embedding_size * 9, self.output_dim)

        self.maxpooling = MaxPooling()

    def forward(self, G):
        # 使用GATConv层进行图卷积操作
        g = self.relu(self.gat1(G))
        g = g.reshape(-1, self.embedding_size * 3)
        g = self.relu(self.gat2(G, g))
        g = g.reshape(-1, self.embedding_size * 9)
        g = self.relu(self.gat3(G, g))
        g = g.reshape(-1, self.embedding_size * 9)

        # 使用全局池化操作
        g_maxpooling = self.maxpooling(G, g)

        # 将结果传递给线性层
        output = self.fc(g_maxpooling)

        return output


class SelfAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout, l=0):
        super(SelfAttentionModule, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        # self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.local_linear1 = nn.Linear(l*64, 512)
        self.dropout = nn.Dropout(0.5)
        # self.local_linear1 = nn.Linear(64, 1024)

    def forward(self, protein_ft):
        '''
        :param protein_ft: batch * len * in_channel
        :return:
        '''
        # 应用自注意力机制
        batch_size = protein_ft.size()[0]
        attn_output, _ = self.self_attn(protein_ft, protein_ft, protein_ft)
        attn_output = attn_output.transpose(1, 2)  # 为了进行 Conv1D，转置输出的维度

        # 应用卷积操作
        conv_ft = self.conv(attn_output)
        conv_ft = self.dropout(conv_ft)
        # conv_ft = F.relu(conv_ft)

        # 应用最大池化
        pooled_ft = self.pool(conv_ft).view(batch_size, -1)

        # 展平
        batch_size = pooled_ft.size(0)
        flattened_ft = pooled_ft.view(batch_size, -1)
        # attn_output = self.dropout(attn_output)
        local_pair_ft = self.local_linear1(flattened_ft)
        return local_pair_ft

class CNNmodule(nn.Module):
    def __init__(self, in_channel, kernel_width, l=0):
        super(CNNmodule, self).__init__()
        self.kernel_width = kernel_width
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=1, kernel_size=2, stride=2, padding=0)
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        # self.out_linear = nn.Linear(l*64, 512)
        self.dropout = nn.Dropout(0.5)


    def forward(self, protein_ft):
        '''
        :param protein_ft: batch*len*amino_dim
        :return:
        '''
        # print("Original protein_ft shape:", protein_ft.shape)

        # batch_size = protein_ft.size()[0]
        # protein_ft = protein_ft.transpose(1, 2)

        conv_ft = self.conv(protein_ft)
        conv_ft = self.dropout(conv_ft)
        # conv_ft = self.pool(conv_ft).view(batch_size, -1)
        # conv_ft = self.out_linear(conv_ft)
        return conv_ft


class DeepAAIKmerPssmEmbeddingCls(nn.Module):
    def __init__(self, **param_dict):
        super(DeepAAIKmerPssmEmbeddingCls, self).__init__()
        self.amino_ft_dim = param_dict['amino_type_num'],
        self.param_dict = param_dict
        self.kmer_dim = param_dict['kmer_dim']
        self.h_dim = param_dict['h_dim']
        self.dropout = param_dict['dropout_num']
        self.add_bn = param_dict['add_bn']
        self.add_res = param_dict['add_res']
        self.amino_embedding_dim = param_dict['amino_embedding_dim']
        # self.kernel_cfg = param_dict['kernel_cfg']
        # self.channel_cfg = param_dict['channel_cfg']
        # self.dilation_cfg = param_dict['dilation_cfg']

        self.antibody_kmer_linear = nn.Linear(param_dict['kmer_dim'], self.h_dim)
        self.virus_kmer_linear = nn.Linear(param_dict['kmer_dim'], self.h_dim)

        self.antibody_pssm_linear = nn.Linear(param_dict['pssm_antibody_dim'], self.h_dim)
        self.virus_pssm_linear = nn.Linear(param_dict['pssm_virus_dim'], self.h_dim)

        self.share_linear = nn.Linear(self.h_dim, self.h_dim)
        self.share_gcn1 = GCNConv(self.h_dim, self.h_dim)
        self.share_gcn2 = GCNConv(self.h_dim, self.h_dim)

        self.antibody_adj_trans = nn.Linear(self.h_dim, self.h_dim)
        self.virus_adj_trans = nn.Linear(self.h_dim, self.h_dim)

        self.cross_scale_merge = nn.Parameter(
            torch.ones(1)
        )

        # self.amino_embedding_layer = nn.Embedding(param_dict['amino_type_num'], self.amino_embedding_dim)
        # self.channel_cfg.insert(0, self.amino_embedding_dim)
        # self.local_linear = nn.Linear(self.channel_cfg[-1] * 2, self.h_dim)
        self.global_linear = nn.Linear(self.h_dim * 2, self.h_dim)
        self.pred_linear = nn.Linear(self.h_dim, 1)

        self.activation = nn.ELU()
        for m in self.modules():
            self.weights_init(m)
            
        self.max_antibody_len = param_dict['max_antibody_len']
        self.max_virus_len = param_dict['max_virus_len']

        # 创建 GAT 模型实例,9.18
        args_antibody = {'emb_dim': 344, 'output_dim': 128}
        args_virus = {'emb_dim': 912, 'output_dim': 128}
        self.gat_antibody = MyGATModel(args_antibody)  # 创建 antibody 的 GAT 模型
        self.gat_virus = MyGATModel(args_virus)  # 创建 virus 的 GAT 模型
            
        # self.cnnmodule = CNNmodule(in_channel=344, kernel_width=self.amino_ft_dim, l=self.max_antibody_len)
        # self.cnnmodule2 = CNNmodule(in_channel=22, kernel_width=self.amino_ft_dim, l=self.max_virus_len)
        self.cnn_module = CNNmodule(in_channel=1, kernel_width=2, l=1024)  # 根据您的需要设置 kernel_width 和 l

        # Create SelfAttentionModules
        self.self_attn_module = SelfAttentionModule(d_model=344, num_heads=4, dropout=0.5, l=self.max_antibody_len)
        self.self_attn_module2 = SelfAttentionModule(d_model=912, num_heads=2, dropout=0.5, l=self.max_virus_len)

        self.local_linear1 = nn.Linear(1024, 512)
        self.local_linear2 = nn.Linear(512, 512)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)


    def forward(self, **ft_dict):
        '''
        :param ft_dict:
                ft_dict = {
                'antibody_graph_node_ft': FloatTensor  node_num * kmer_dim
                'virus_graph_node_ft': FloatTensor  node_num * kmer_dim,
                'antibody_amino_ft': LongTensor  batch * max_antibody_len * 1
                'virus_amino_ft': LongTensor  batch * max_virus_len * 1,
                'antibody_idx': LongTensor  batch
                'virus_idx': LongTensor  batch
            }
        :return:
        '''
        device = ft_dict['antibody_graph_node_kmer_ft'].device
        antibody_graph_node_num = ft_dict['antibody_graph_node_kmer_ft'].size()[0]
        virus_graph_node_num = ft_dict['virus_graph_node_kmer_ft'].size()[0]
        #创建了两个全零张量 antibody_res_mat 和 virus_res_mat，它们的大小分别是根据输入特征的维度和指定的 self.h_dim 来确定的。
        antibody_res_mat = torch.zeros(antibody_graph_node_num, self.h_dim).to(device)
        virus_res_mat = torch.zeros(virus_graph_node_num, self.h_dim).to(device)

        antibody_node_kmer_ft = self.antibody_kmer_linear(ft_dict['antibody_graph_node_kmer_ft'])
        antibody_node_pssm_ft = self.antibody_pssm_linear(ft_dict['antibody_graph_node_pssm_ft'])
        antibody_node_ft = torch.cat([antibody_node_kmer_ft, antibody_node_pssm_ft], dim=-1)
        antibody_node_ft = self.activation(antibody_node_ft)
        antibody_node_ft = F.dropout(antibody_node_ft, p=self.dropout, training=self.training)

        # antibody_node_ft = self.local_linear1(antibody_node_ft)

        antibody_node_ft = antibody_node_ft.unsqueeze(1)
        antibody_node_ft = self.cnn_module(antibody_node_ft)
        antibody_node_ft = antibody_node_ft.squeeze(1)

        # share
        antibody_node_ft = self.share_linear(antibody_node_ft)
        antibody_res_mat = antibody_res_mat + antibody_node_ft
        antibody_node_ft = self.activation(antibody_node_ft)
        antibody_node_ft = F.dropout(antibody_node_ft, p=self.dropout, training=self.training)

        # generate antibody adj,线性变换、tanh 激活函数和归一化
        antibody_trans_ft = self.antibody_adj_trans(antibody_node_ft)
        antibody_trans_ft = torch.tanh(antibody_trans_ft)
        w = torch.norm(antibody_trans_ft, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        #得到的 antibody_adj 是一个代表邻接矩阵的张量。
        antibody_adj = torch.mm(antibody_trans_ft, antibody_trans_ft.t()) / w_mat

        # 进行 GCN（图卷积神经网络）操作
        antibody_node_ft = self.share_gcn1(antibody_node_ft, antibody_adj)
        antibody_res_mat = antibody_res_mat + antibody_node_ft

        antibody_node_ft = self.activation(antibody_res_mat)  # add
        antibody_node_ft = F.dropout(antibody_node_ft, p=self.dropout, training=self.training)
        antibody_node_ft = self.share_gcn2(antibody_node_ft, antibody_adj)
        antibody_res_mat = antibody_res_mat + antibody_node_ft

        # 在GCN之后应用注意力模块
        attention_layer = AttentionLayer(in_features=self.h_dim).to(device)
        antibody_node_ft_with_attention = attention_layer(antibody_node_ft)
        antibody_res_mat = antibody_res_mat + antibody_node_ft_with_attention

        # virus
        virus_node_kmer_ft = self.virus_kmer_linear(ft_dict['virus_graph_node_kmer_ft'])
        virus_node_pssm_ft = self.virus_pssm_linear(ft_dict['virus_graph_node_pssm_ft'])
        virus_node_ft = torch.cat([virus_node_kmer_ft, virus_node_pssm_ft], dim=-1)
        virus_node_ft = self.activation(virus_node_ft)
        virus_node_ft = F.dropout(virus_node_ft, p=self.dropout, training=self.training)

        # virus_node_ft = self.local_linear1(virus_node_ft)

        virus_node_ft = virus_node_ft.unsqueeze(1)
        virus_node_ft = self.cnn_module(virus_node_ft)
        virus_node_ft = virus_node_ft.squeeze(1)

        # share
        virus_node_ft = self.share_linear(virus_node_ft)
        virus_res_mat = virus_res_mat + virus_node_ft
        virus_node_ft = self.activation(virus_node_ft)
        virus_node_ft = F.dropout(virus_node_ft, p=self.dropout, training=self.training)

        # generate antibody adj
        virus_trans_ft = self.virus_adj_trans(virus_node_ft)
        virus_trans_ft = torch.tanh(virus_trans_ft)
        w = torch.norm(virus_trans_ft, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        virus_adj = torch.mm(virus_trans_ft, virus_trans_ft.t()) / w_mat
        # virus_adj = eye_adj

        virus_node_ft = self.share_gcn1(virus_node_ft, virus_adj)
        virus_res_mat = virus_res_mat + virus_node_ft

        virus_node_ft = self.activation(virus_res_mat)  # add
        virus_node_ft = F.dropout(virus_node_ft, p=self.dropout, training=self.training)
        virus_node_ft = self.share_gcn2(virus_node_ft, virus_adj)
        virus_res_mat = virus_res_mat + virus_node_ft

        # 在GCN之后应用注意力模块,
        attention_layer = AttentionLayer(in_features=self.h_dim).to(device)
        virus_node_ft_with_attention = attention_layer(virus_node_ft)
        virus_res_mat = virus_res_mat + virus_node_ft_with_attention

        antibody_res_mat = self.activation(antibody_res_mat)
        virus_res_mat = self.activation(virus_res_mat)

        antibody_res_mat = antibody_res_mat[ft_dict['antibody_idx']]
        virus_res_mat = virus_res_mat[ft_dict['virus_idx']]

        # cross
        global_pair_ft = torch.cat([virus_res_mat, antibody_res_mat], dim=1)
        global_pair_ft = self.activation(global_pair_ft)
        global_pair_ft = F.dropout(global_pair_ft, p=self.dropout, training=self.training)
        global_pair_ft = self.global_linear(global_pair_ft)

        batch_size = ft_dict['virus_amino_ft'].size()[0]
        # antibody_ft = self.cnnmodule(ft_dict['antibody_amino_ft']).view(batch_size, -1)
        # virus_ft = self.cnnmodule2(ft_dict['virus_amino_ft']).view(batch_size, -1)
        antibody_ft = self.self_attn_module(ft_dict['antibody_amino_ft']).reshape(batch_size, -1)
        virus_ft = self.self_attn_module2(ft_dict['virus_amino_ft']).reshape(batch_size, -1)

        # 使用 GAT 模型处理 antibody_amino_ft 和 virus_amino_ft,9.18
        antibody_amino_ft = ft_dict['antibody_amino_ft']
        virus_amino_ft = ft_dict['virus_amino_ft']

        # =======
        # temp = default_loader(antibody_amino_ft[0],0,)
        # =======

        # antibody_ft = self.gat_antibody(antibody_amino_ft)
        # virus_ft = self.gat_virus(virus_amino_ft)

        local_pair_ft = torch.cat([virus_ft, antibody_ft], dim=-1).view(batch_size, -1)
        local_pair_ft = self.activation(local_pair_ft)
        local_pair_ft = self.local_linear1(local_pair_ft)
        local_pair_ft = self.activation(local_pair_ft)
        local_pair_ft = self.local_linear2(local_pair_ft)

        pair_ft = global_pair_ft + local_pair_ft + (global_pair_ft * local_pair_ft) * self.cross_scale_merge
        # pair_ft = local_pair_ft
        # pair_ft = global_pair_ft
        pair_ft = self.activation(pair_ft)
        pair_ft = F.dropout(pair_ft, p=self.dropout, training=self.training)

        pred = self.pred_linear(pair_ft)
        pred = torch.sigmoid(pred)

        return pred, antibody_adj, virus_adj
