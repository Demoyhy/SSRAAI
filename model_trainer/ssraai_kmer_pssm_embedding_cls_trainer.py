#!/usr/bin/env
# coding:utf-8
import sys
import os
# sys.path.append(os.getcwd())
sys.path.append('/data0/yanghy/workplace/DeepAAI-main')
import torch.nn.functional as F
import torch.nn as nn
from dataset.abs_dataset_cls import AbsDataset
from metrics.evaluate import evaluate_classification
from models.ssraai_kmer_pssm_embedding_cls import DeepAAIKmerPssmEmbeddingCls
import numpy as np
import random
import torch
import math
import os.path as osp
from utils.index_map import get_map_index_for_sub_arr
import warnings
warnings.filterwarnings("ignore")


current_path = osp.dirname(osp.realpath(__file__))
# save_model_data_dir = 'save_model_data'


class Trainer(object):
    def __init__(self, **param_dict):
        self.param_dict = param_dict
        self.epoch_results = []  # 添加这一行定义epoch_results列表
        self.setup_seed(self.param_dict['seed'])
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.dataset = AbsDataset(max_antibody_len=self.param_dict['max_antibody_len'],
                                  max_virus_len=self.param_dict['max_virus_len'],
                                  train_split_param=self.param_dict['hot_data_split'],
                                  label_type=self.param_dict['label_type'],
                                  kmer_min_df=self.param_dict['kmer_min_df'],
                                  reprocess=False)
        self.param_dict.update(self.dataset.dataset_info)
        self.dataset.to_tensor(self.device)
        #获取当前文件的文件名，并去掉扩展名，保存到self.file_name属性中。
        self.file_name = __file__.split('/')[-1].replace('.py', '')
        #根据文件名、种子和批次大小创建训练器信息，保存到self.trainer_info属性中。
        self.trainer_info = '{}_seed={}_batch={}'.format(self.file_name, self.param_dict['seed'], self.param_dict['batch_size'])
        self.loss_op = nn.BCELoss()
        self.build_model()

    def build_model(self):
        self.model = DeepAAIKmerPssmEmbeddingCls(**self.param_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'])
        self.best_res = None
        self.min_dif = -1e10
    
    #用于设置随机种子。
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def iteration(self, epoch, pair_idx, antibody_graph_node_idx, virus_graph_node_idx, is_training=True, shuffle=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        #获取pair_idx的第一个维度的大小，即样本对的数量。
        pair_num = pair_idx.shape[0]
        if shuffle is True:
            range_idx = np.random.permutation(pair_num)
        else:
            range_idx = np.arange(0, pair_num)
        
        #用于保存每个批次的预测结果、真实标签和损失值。
        all_pred = []
        all_label = []
        all_loss = []

        #这些行从self.dataset中获取抗体和病毒的特征，并将它们存储在对应的变量中。
        #antibody_graph_node_kmer_ft和virus_graph_node_kmer_ft表示抗体和病毒的k-mer特征。
        antibody_graph_node_kmer_ft = self.dataset.protein_ft_dict['antibody_kmer_whole'][antibody_graph_node_idx]
        virus_graph_node_kmer_ft = self.dataset.protein_ft_dict['virus_kmer_whole'][virus_graph_node_idx]
        #表示抗体和病毒的PSSM特征。
        antibody_graph_node_pssm_ft = self.dataset.protein_ft_dict['antibody_pssm'][antibody_graph_node_idx]
        virus_graph_node_pssm_ft = self.dataset.protein_ft_dict['virus_pssm'][virus_graph_node_idx]
        #是通过调用get_map_index_for_sub_arr函数获取的抗体和病毒的索引映射数组。
        antibody_graph_map_arr = get_map_index_for_sub_arr(
            antibody_graph_node_idx, np.arange(0, 254))
        virus_graph_map_arr = get_map_index_for_sub_arr(
            virus_graph_node_idx, np.arange(0, 940))

        for i in range(math.ceil(pair_num/self.param_dict['batch_size'])):
            right_bound = min((i + 1)*self.param_dict['batch_size'], pair_num + 1)
            batch_idx = range_idx[i * self.param_dict['batch_size']: right_bound]
            shuffled_batch_idx = pair_idx[batch_idx]

            batch_antibody_idx = self.dataset.antibody_index_in_pair[shuffled_batch_idx]
            batch_virus_idx = self.dataset.virus_index_in_pair[shuffled_batch_idx]
            batch_label = self.dataset.all_label_mat[shuffled_batch_idx]
            batch_tensor_label = torch.FloatTensor(batch_label).to(self.device)

            # batch_antibody_amino_ft = self.dataset.protein_ft_dict['antibody_one_hot'][batch_antibody_idx]
            batch_antibody_amino_ft = self.dataset.protein_ft_dict['antibody_contact_map'][batch_antibody_idx]
            # batch_virus_amino_ft = self.dataset.protein_ft_dict['virus_one_hot'][batch_virus_idx]
            batch_virus_amino_ft = self.dataset.protein_ft_dict['virus_contact_map'][batch_virus_idx]
            # index_remap:  raw_index -> graph_index
            # batch_antibody_idx -> batch_antibody_node_idx_in_graph
            # batch_virus_idx -> batch_virus_node_idx_in_graph
            batch_antibody_node_idx_in_graph = antibody_graph_map_arr[batch_antibody_idx]
            batch_virus_node_idx_in_graph = virus_graph_map_arr[batch_virus_idx]

            #构建一个字典ft_dict，包含了所有的特征和索引，用于传递给模型进行预测。
            ft_dict = {
                'antibody_graph_node_kmer_ft': antibody_graph_node_kmer_ft,
                'virus_graph_node_kmer_ft': virus_graph_node_kmer_ft,
                'antibody_graph_node_pssm_ft': antibody_graph_node_pssm_ft,
                'virus_graph_node_pssm_ft': virus_graph_node_pssm_ft,
                'antibody_amino_ft': batch_antibody_amino_ft,
                'virus_amino_ft': batch_virus_amino_ft,
                'antibody_idx': batch_antibody_node_idx_in_graph,
                'virus_idx': batch_virus_node_idx_in_graph
            }

            #调用模型的__call__方法，传递ft_dict字典作为关键字参数，进行预测。预测结果存储在pred中。
            pred, antibody_adj, virus_adj = self.model(**ft_dict)
            #使用pred.view(-1)将预测结果转换为一维张量。
            pred = pred.view(-1)
            
            #计算预测结果pred和批次标签batch_tensor_label之间的损失c_loss
            if is_training:
                c_loss = self.loss_op(pred, batch_tensor_label)
                param_l2_loss = 0
                param_l1_loss = 0
                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_l2_loss += torch.norm(param, p=2)
                        param_l1_loss += torch.norm(param, p=1)

                param_l2_loss = self.param_dict['param_l2_coef'] * param_l2_loss
                # param_l1_loss = self.param_dict['param_l1_coef'] * param_l1_loss
                adj_l1_loss = self.param_dict['adj_loss_coef'] * torch.norm(virus_adj) + \
                              self.param_dict['adj_loss_coef'] * torch.norm(antibody_adj)
                loss = c_loss + adj_l1_loss + param_l2_loss
                # print('c_loss = ', c_loss.detach().to('cpu').numpy(),
                #       'adj_l1_loss = ', adj_l1_loss.detach().to('cpu').numpy(),
                #       'param_l2_loss = ', param_l2_loss.detach().to('cpu').numpy())
                
                #将损失值添加到all_loss列表中。
                all_loss.append(loss.detach().to('cpu').item())
                #执行反向传播和参数更新步骤，清零梯度。
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            #将预测结果pred从Tensor类型转换为NumPy数组，并存储在all_pred列表中
            pred = pred.detach().to('cpu').numpy()
            all_pred = np.hstack([all_pred, pred])
            #将批次标签batch_label转换为NumPy数组，并将其添加到all_label列表中。
            all_label = np.hstack([all_label, batch_label]).astype(np.long)
        # print(all_loss)

        return all_pred, all_label, all_loss

    def print_res(self, res_list, epoch):
        train_acc, seen_valid_acc, seen_test_acc, unseen_test_acc, \
        train_p, seen_valid_p, seen_test_p, unseen_test_p, \
        train_r, seen_valid_r, seen_test_r, unseen_test_r, \
        train_f1, seen_valid_f1, seen_test_f1, unseen_test_f1, \
        train_auc, seen_valid_auc, seen_test_auc, unseen_test_auc, \
        train_mcc, seen_valid_mcc, seen_test_mcc, unseen_test_mcc = res_list

        #这行代码构造了要打印的结果字符串msg_log，使用了格式化字符串的方式将结果值插入到相应的位置。字符串中包含了轮数epoch以及各个结果的名称和对应的值。
        msg_log = 'Epoch: {:03d}, ' \
                  'ACC: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'P: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'R: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'F1: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'AUC: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f}, ' \
                  'MCC: Train {:.4f}, Val: {:.4f}, Test Seen: {:.4f}, Test Unseen: {:.4f},  ' \
            .format(epoch,
                    train_acc, seen_valid_acc, seen_test_acc, unseen_test_acc,
                    train_p, seen_valid_p, seen_test_p, unseen_test_p, \
                    train_r, seen_valid_r, seen_test_r, unseen_test_r, \
                    train_f1, seen_valid_f1, seen_test_f1, unseen_test_f1, \
                    train_auc, seen_valid_auc, seen_test_auc, unseen_test_auc, \
                    train_mcc, seen_valid_mcc, seen_test_mcc, unseen_test_mcc
                    )
        print(msg_log)

    def train(self, display=True):
        for epoch in range(1, self.param_dict['epoch_num'] + 1):
            train_pred, train_label, train_loss = self.iteration(epoch, self.dataset.train_index,
                           antibody_graph_node_idx=self.dataset.known_antibody_idx,
                           virus_graph_node_idx=self.dataset.known_virus_idx,
                           is_training=True, shuffle=True)
            #对训练结果进行评估，计算训练集的准确率、精确度、召回率、F1 值、AUC 值和 Matthews 相关系数（MCC）
            train_acc, train_p, train_r, train_f1, train_auc, train_mcc = \
                evaluate_classification(predict_proba=train_pred, label=train_label)

            valid_pred, valid_label, valid_loss = self.iteration(epoch, self.dataset.valid_seen_index,
                           antibody_graph_node_idx=self.dataset.known_antibody_idx,
                           virus_graph_node_idx=self.dataset.known_virus_idx,
                           is_training=False, shuffle=False)
            seen_valid_acc, seen_valid_p, seen_valid_r, seen_valid_f1, seen_valid_auc, seen_valid_mcc = \
                evaluate_classification(predict_proba=valid_pred, label=valid_label)

            seen_test_pred, seen_test_label, seen_test_loss = self.iteration(epoch, self.dataset.test_seen_index,
                           antibody_graph_node_idx=self.dataset.known_antibody_idx,
                           virus_graph_node_idx=self.dataset.known_virus_idx,
                           is_training=False, shuffle=False)
            seen_test_acc, seen_test_p, seen_test_r, seen_test_f1, seen_test_auc, seen_test_mcc = \
                evaluate_classification(predict_proba=seen_test_pred, label=seen_test_label)

            unseen_test_pred, unseen_test_label, unseen_test_loss = self.iteration(epoch, self.dataset.test_unseen_index,
                           antibody_graph_node_idx=np.hstack(
                               (self.dataset.known_antibody_idx, self.dataset.unknown_antibody_idx)),
                           virus_graph_node_idx=self.dataset.known_virus_idx,
                           is_training=False, shuffle=False)
            unseen_test_acc, unseen_test_p, unseen_test_r, unseen_test_f1, unseen_test_auc, unseen_test_mcc = \
                evaluate_classification(predict_proba=unseen_test_pred, label=unseen_test_label)

            res_list = [
                train_acc, seen_valid_acc, seen_test_acc, unseen_test_acc,
                train_p, seen_valid_p, seen_test_p, unseen_test_p,
                train_r, seen_valid_r, seen_test_r, unseen_test_r,
                train_f1, seen_valid_f1, seen_test_f1, unseen_test_f1,
                train_auc, seen_valid_auc, seen_test_auc, unseen_test_auc,
                train_mcc, seen_valid_mcc, seen_test_mcc, unseen_test_mcc
            ]
            # 保存当前epoch的结果到epoch_results列表
            self.epoch_results.append((epoch, res_list))

            if seen_valid_acc > self.min_dif:
                self.min_dif = seen_valid_acc
                self.best_res = res_list
                self.best_epoch = epoch
                
                #save model，将完整的模型 self.model 保存
                save_model_data_dir = '/data0/yanghy/workplace/DeepAAI-main/c_1/'
                # save_complete_model_path = osp.join(current_path,save_model_data_dir ,self.trainer_info + '_complete.pkl')
                save_complete_model_path = osp.join(save_model_data_dir,self.trainer_info + '_complete.pkl')
                torch.save(self.model, save_complete_model_path)
                #将模型的参数（state_dict）保存
                same_model_param_path = osp.join(current_path,save_model_data_dir,self.trainer_info + '_param.pkl')
                torch.save(self.model.state_dict(), same_model_param_path)

            #调用 print_res 方法打印当前的评估指标和 epoch
            if display:
                self.print_res(res_list, epoch)

            #如果当前的 epoch 是 50 的倍数且大于 0，则打印最佳结果 best_res 和最佳 epoch best_epoch。
            if epoch % 50 == 0 and epoch > 0:
                print('Best res')
                self.print_res(self.best_res, self.best_epoch)

            # 训练完成后将结果保存到文件中
            save_path = "/data0/yanghy/workplace/DeepAAI-main/doc/evaluation_results/evaluation_results6.txt"
            with open(save_path, "w") as f:
                f.write(
                    "Epoch\tACC_Train\tACC_Val\tACC_Test_Seen\tACC_Test_Unseen\t"
                    "P_Train\tP_Val\tP_Test_Seen\tP_Test_Unseen\t"
                    "R_Train\tR_Val\tR_Test_Seen\tR_Test_Unseen\t"
                    "F1_Train\tF1_Val\tF1_Test_Seen\tF1_Test_Unseen\t"
                    "AUC_Train\tAUC_Val\tAUC_Test_Seen\tAUC_Test_Unseen\t"
                    "MCC_Train\tMCC_Val\tMCC_Test_Seen\tMCC_Test_Unseen\n")
                for epoch, res_list in enumerate(self.epoch_results, 1):
                    f.write(f"{epoch}\t")
                    f.write("\t".join(map(str, res_list)))
                    f.write("\n")

    def evaluate_model(self):
        # load param
        model_file_path = osp.join(current_path, '../save_model_param_pred', 'deep_aai_k+p+e',
                                   'deep_aai_kmer_pssm_embedding_cls_seed={}_param.pkl'.format(self.param_dict['seed']))
        #torch.load函数加载模型参数文件
        self.model.load_state_dict(torch.load(model_file_path))
        print('load_param ', model_file_path)

        #在测试集上进行评估。首先调用iteration函数进行预测和计算损失。
        seen_test_pred, seen_test_label, seen_test_loss = self.iteration(
            0, self.dataset.test_seen_index,
            antibody_graph_node_idx=self.dataset.known_antibody_idx,
            virus_graph_node_idx=self.dataset.known_virus_idx,
            is_training=False, shuffle=False)
        seen_test_acc, seen_test_p, seen_test_r, seen_test_f1, seen_test_auc, seen_test_mcc = \
            evaluate_classification(predict_proba=seen_test_pred, label=seen_test_label)

        #对未知类别的测试集样本进行评估。
        unseen_test_pred, unseen_test_label, unseen_test_loss = self.iteration(
            0, self.dataset.test_unseen_index,
            antibody_graph_node_idx=np.hstack((self.dataset.known_antibody_idx, self.dataset.unknown_antibody_idx)),
            virus_graph_node_idx=self.dataset.known_virus_idx,
            is_training=False, shuffle=False)
        unseen_test_acc, unseen_test_p, unseen_test_r, unseen_test_f1, unseen_test_auc, unseen_test_mcc = \
            evaluate_classification(predict_proba=unseen_test_pred, label=unseen_test_label)

        #将评估结果以格式化字符串的形式保存在log_str变量中
        log_str = \
            'Evaluate Result:  ACC      \tP      \tR      \tF1     \tAUC    \tMCC  \n' \
            'Seen Test:        {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}  \n'  \
            'Unseen Test:      {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(
            seen_test_acc, seen_test_p, seen_test_r, seen_test_f1, seen_test_auc, seen_test_mcc,
            unseen_test_acc, unseen_test_p, unseen_test_r, unseen_test_f1, unseen_test_auc, unseen_test_mcc
            )
        print(log_str)


if __name__ == '__main__':
    for seed in range(5):
        print('seed = ', seed)
        param_dict = {
            'hot_data_split': [0.9, 0.05, 0.05],
            'seed': seed,
            'kmer_min_df': 0.1,
            'label_type': 'label_10',
            'batch_size': 32,
            'epoch_num': 100,
            'h_dim': 512,
            'dropout_num': 0.4,
            'lr': 5e-5,
            'amino_embedding_dim': 7,
            'adj_loss_coef': 1e-4,
            'param_l2_coef': 5e-4,
            'add_res': True,
            'add_bn': False,
            'max_antibody_len': 344,
            'max_virus_len': 912,
        }
        trainer = Trainer(**param_dict)
        trainer.train()
        # trainer.evaluate_model()