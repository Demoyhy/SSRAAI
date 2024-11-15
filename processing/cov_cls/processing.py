import os.path as osp
import numpy as np
import pandas as pd
import json

try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys
import os
import Bio.PDB

from map import calc_contact_map

from dataset_tools import get_padding_ft_dict, get_index_in_target_list, get_all_protein_ac_feature
from dataset_split import train_test_split
from k_mer_utils import k_mer_ft_generate, KmerTranslator

current_path = osp.dirname(osp.realpath(__file__))
corpus_dir = 'corpus/cov_cls'
processed_dir = 'corpus/processed_mat'
pssm_dir = 'pssm'

antibody_pssm_file = osp.join(current_path, pssm_dir, "anti_medp_pssm_840.json")
virus_pssm_file = osp.join(current_path, pssm_dir, "virus_medp_pssm_420.json")
data_file = osp.join(current_path, 'dataset_cov_cls6.xlsx')

dataset_name = 'abs_dataset_cov_cls'
max_antibody_len = 436
max_virus_len = 1288
kmer_min_df = 0.1
dataset_param_str = '{}_antibody={}_virus={}_kmer_min_df={}'.format(
    dataset_name, max_antibody_len, max_virus_len, kmer_min_df)
protein_ft_save_path = osp.join(current_path, processed_dir, dataset_param_str + '__protein_ft_dict2.pkl')

amino_one_hot_ft_pad_dict, amino_pssm_ft_pad_dict, amino_physicochemical_ft_pad_dict, amino_map_idx = get_padding_ft_dict()

protein_ft_dict = {}


def processing():
    data_df = pd.read_excel(data_file)

    split = data_df['split'].to_numpy()
    train_index = np.where(split == 'seen')[0]
    test_unseen_index = np.where(split == 'unseen')[0]
    all_label_mat = data_df['label'].to_numpy().astype(np.long)

    raw_all_antibody_seq_list = data_df['antibody_seq'].to_list()
    raw_all_virus_seq_list = data_df['virus_seq'].to_list()

    raw_all_antibody_set = list(sorted(set(raw_all_antibody_seq_list)))
    raw_all_virus_set = list(sorted(set(raw_all_virus_seq_list)))

    raw_all_antibody_set_len = np.array(
        list(map(lambda x: len(x), raw_all_antibody_set)))
    raw_all_virus_set_len = np.array(
        list(map(lambda x: len(x), raw_all_virus_set)))

    antibody_index_in_pair = get_index_in_target_list(raw_all_antibody_seq_list, raw_all_antibody_set)
    virus_index_in_pair = get_index_in_target_list(raw_all_virus_seq_list, raw_all_virus_set)

    known_antibody_idx = np.unique(antibody_index_in_pair)
    # unknown_antibody_idx = np.unique(antibody_index_in_pair[test_unseen_index])
    known_virus_idx = np.unique(virus_index_in_pair)

    # one-hot
    # protein_ft_dict['antibody_one_hot'] = protein_seq_list_to_ft_mat(
    #     raw_all_antibody_set, max_antibody_len, ft_type='amino_one_hot')
    # protein_ft_dict['virus_one_hot'] = protein_seq_list_to_ft_mat(
    #     raw_all_virus_set, max_virus_len, ft_type='amino_one_hot')

    # # pssm
    # protein_ft_dict['antibody_pssm'], protein_ft_dict['virus_pssm'] = load_pssm_ft_mat()

    # amino_num
    protein_ft_dict['antibody_amino_num'] = protein_seq_list_to_ft_mat(
        raw_all_antibody_set, max_antibody_len, ft_type='amino_num')
    protein_ft_dict['virus_amino_num'] = protein_seq_list_to_ft_mat(
        raw_all_virus_set, max_virus_len, ft_type='amino_num')

    # k-mer-whole
    kmer_translator = KmerTranslator(trans_type='std', min_df=kmer_min_df, name=dataset_param_str)
    protein_ft = kmer_translator.fit_transform(raw_all_antibody_set + raw_all_virus_set)
    # kmer_translator.save()
    protein_ft_dict['antibody_kmer_whole'] = protein_ft[0: len(raw_all_antibody_set)]
    protein_ft_dict['virus_kmer_whole'] = protein_ft[len(raw_all_antibody_set):]

    protein_ft_dict['antibody_contact_map'] = process_antibody_files(input_folder_antibody, raw_all_antibody_set)
    protein_ft_dict['virus_contact_map'] = process_virus_files(input_folder_virus, raw_all_virus_set)

    # save
    np.save(osp.join(current_path, corpus_dir, 'train_index'), train_index)
    np.save(osp.join(current_path, corpus_dir, 'test_unseen_index'), test_unseen_index)
    np.save(osp.join(current_path, corpus_dir, 'all_label_mat'), all_label_mat)
    np.save(osp.join(current_path, corpus_dir, 'antibody_index_in_pair'), antibody_index_in_pair)
    np.save(osp.join(current_path, corpus_dir, 'virus_index_in_pair'), virus_index_in_pair)
    np.save(osp.join(current_path, corpus_dir, 'known_antibody_idx'), known_antibody_idx)
    # np.save(osp.join(current_path, corpus_dir, 'unknown_antibody_idx'), unknown_antibody_idx)
    np.save(osp.join(current_path, corpus_dir, 'known_virus_idx'), known_virus_idx)
    np.save(osp.join(current_path, corpus_dir, 'raw_all_antibody_set_len'), raw_all_antibody_set_len)
    np.save(osp.join(current_path, corpus_dir, 'raw_all_virus_set_len'), raw_all_virus_set_len)


input_folder_antibody = 'D:/workplace/SSRAAI-main/Data/cov_antibody_seq/results'
input_folder_virus = 'D:/workplace/SSRAAI-main/Data/cov_virus_seq/results'


def process_antibody_files(input_folder_antibody, antibody_set):
    call_count = 0
    antibody_contact_map_list = []  # 存储所有抗体的接触图的列表
    data_df = pd.read_excel(data_file)
    for antibody_seq in antibody_set:
        # 找到对应的行
        row = data_df[data_df['antibody_seq'] == antibody_seq].iloc[0]
        # 获取对应的 pdb 文件名
        pdb_file_name = str(int(row['antibody_pdb'])) + '.pdb'
        # 构建 PDB 文件的路径
        pdb_path = os.path.join(input_folder_antibody, pdb_file_name)
        # 调用 calc_contact_map() 函数计算接触图
        contact_map = calc_contact_map(pdb_path)
        # 将接触图维度调整为 (max_antibody_len, max_antibody_len)
        contact_map = contact_map[:max_antibody_len, :max_antibody_len]
        # 如果接触图维度小于 max_antibody_len，则进行填充
        if contact_map.shape[0] < max_antibody_len:
            pad_width = max_antibody_len - contact_map.shape[0]
            contact_map = np.pad(contact_map, ((0, pad_width), (0, pad_width)), mode='constant', constant_values=0)
        antibody_contact_map_list.append(contact_map)
        call_count += 1  # 增加调用次数计数器的值
        print(f" Call count: {call_count}")
    # 将列表转换为 NumPy 数组
    antibody_contact_map_array = np.array(antibody_contact_map_list)

    return antibody_contact_map_array


def process_virus_files(input_folder_virus, virus_set):
    virus_contact_map_list = []  # 存储所有抗体的接触图的列表
    data_df = pd.read_excel(data_file)
    for virus_seq in virus_set:
        # 找到对应的行
        row = data_df[data_df['virus_seq'] == virus_seq].iloc[0]
        # 获取对应的 pdb 文件名
        pdb_file_name = str(int(row['virus_pdb'])) + '.pdb'
        # 构建 PDB 文件的路径
        pdb_path = os.path.join(input_folder_virus, pdb_file_name)
        # 调用 calc_contact_map() 函数计算接触图
        contact_map = calc_contact_map(pdb_path)
        # 将接触图维度调整为 (max_virus_len, max_virus_len)
        contact_map = contact_map[:max_virus_len, :max_virus_len]
        # 如果接触图维度小于 max_virus_len，则进行填充
        if contact_map.shape[0] < max_virus_len:
            pad_width = max_virus_len - contact_map.shape[0]
            contact_map = np.pad(contact_map, ((0, pad_width), (0, pad_width)), mode='constant', constant_values=0)
        virus_contact_map_list.append(contact_map)
    # 将列表转换为 NumPy 数组
    virus_contact_map_array = np.array(virus_contact_map_list)
    return virus_contact_map_array


def save_data(protein_ft_dict):
    with open(protein_ft_save_path, 'wb') as f:
        if sys.version_info > (3, 0):
            pickle.dump(protein_ft_dict, f, protocol=4)
        else:
            pickle.dump(protein_ft_dict, f, protocol=4)


def protein_seq_list_to_ft_mat(protein_seq_list, max_sql_len, ft_type='amino_one_hot'):
    ft_mat = []
    for protein_seq in protein_seq_list:
        protein_ft = []
        for idx in range(max_sql_len):
            if idx < len(protein_seq):
                amino_name = protein_seq[idx]
            else:
                amino_name = 'pad'

            if 'amino_one_hot' == ft_type:
                amino_ft = amino_one_hot_ft_pad_dict[amino_name]
            elif 'phych' == ft_type:
                amino_ft = amino_physicochemical_ft_pad_dict[amino_name]
            elif 'amino_num' == ft_type:
                amino_ft = amino_map_idx[amino_name]
            else:
                exit('error ft_type')
            protein_ft.append(amino_ft)

        protein_ft = np.array(protein_ft)
        ft_mat.append(protein_ft)

    ft_mat = np.array(ft_mat).astype(np.float32)
    return ft_mat


# def load_pssm_ft_mat():
#     with open(antibody_pssm_file, 'r') as f:
#         anti_pssm_dict = json.load(f)

#     with open(virus_pssm_file, 'r') as f:
#         virus_pssm_dict = json.load(f)

#     anti_pssm_mat = np.array(list(map(lambda name: anti_pssm_dict[name], raw_all_antibody_name)))
#     virus_pssm_mat = np.array(list(map(lambda name: virus_pssm_dict[name], raw_all_virus_name)))
#     return anti_pssm_mat, virus_pssm_mat

if __name__ == '__main__':
    processing()
    save_data(protein_ft_dict)
