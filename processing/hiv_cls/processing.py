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

#current_path：获取当前文件的所在目录的绝对路径。
current_path = osp.dirname(osp.realpath(__file__))
#corpus_dir、processed_dir和pssm_dir：定义了三个目录的相对路径。
corpus_dir = 'corpus/cls'
processed_dir = 'corpus/processed_mat'
pssm_dir = 'pssm'

#antibody_pssm_file和virus_pssm_file：根据当前路径和pssm_dir拼接得到了两个PSSM文件的路径。
antibody_pssm_file = osp.join(current_path, pssm_dir, "anti_medp_pssm_840.json")
virus_pssm_file = osp.join(current_path, pssm_dir, "virus_medp_pssm_420.json")
#data_file：根据当前路径拼接得到了名为dataset_hiv_cls.xlsx的数据文件的路径。
data_file = osp.join(current_path, 'dataset_hiv_cls6.xlsx')

dataset_name = 'abs_dataset_cls'
max_antibody_len= 344
#max_virus_len = 740
max_virus_len = 912
kmer_min_df = 0.1


#dataset_param_str：根据数据集参数拼接得到了一个字符串，用于在后续的文件保存路径中使用。
dataset_param_str = '{}_antibody={}_virus={}_kmer_min_df={}'.format(dataset_name, max_antibody_len, max_virus_len, kmer_min_df)
# protein_ft_save_path：根据当前路径、processed_dir和dataset_param_str拼接得到了保存蛋白质特征字典的文件路径。
protein_ft_save_path = osp.join(current_path, processed_dir, dataset_param_str+'__protein_ft_dict.pkl')

#调用了一个名为get_padding_ft_dict()的函数，返回了一些填充特征字典和索引映射。
amino_one_hot_ft_pad_dict, amino_pssm_ft_pad_dict, amino_physicochemical_ft_pad_dict, amino_map_idx = get_padding_ft_dict()

protein_ft_dict = {}

def processing():
    data_df = pd.read_excel(data_file)
    #从data_df中提取split列的数据，并将其转换为NumPy数组，存储在split中。
    split = data_df['split'].to_numpy()
    #通过np.where()函数找到split数组中值为'seen'的索引，存储在train_index中。
    train_index = np.where(split == 'seen')[0]
    test_unseen_index = np.where(split == 'unseen')[0]
    #取label列的数据，并将其转换为NumPy数组，同时将数据类型转换为np.long
    all_label_mat = data_df['label'].to_numpy().astype(np.long)

    raw_all_antibody_seq_list = data_df['antibody_seq'].to_list()
    raw_all_virus_seq_list = data_df['virus_seq'].to_list()

    raw_all_antibody_set = list(sorted(set(raw_all_antibody_seq_list)))
    raw_all_virus_set = list(sorted(set(raw_all_virus_seq_list)))

    #创建一个NumPy数组，其中的每个元素是raw_all_antibody_set中对应序列的长度。
    raw_all_antibody_set_len = np.array(list(map(lambda x: len(x), raw_all_antibody_set)))
    raw_all_virus_set_len = np.array(list(map(lambda x: len(x), raw_all_virus_set)))

    #返回一个表示raw_all_antibody_seq_list中每个元素在raw_all_antibody_set中的索引的数组，存储在antibody_index_in_pair中。
    antibody_index_in_pair = get_index_in_target_list(raw_all_antibody_seq_list, raw_all_antibody_set)
    virus_index_in_pair = get_index_in_target_list(raw_all_virus_seq_list, raw_all_virus_set)

    #从antibody_index_in_pair中提取train_index中对应的元素，并去除重复的元素
    known_antibody_idx = np.unique(antibody_index_in_pair[train_index])
    unknown_antibody_idx = np.unique(antibody_index_in_pair[test_unseen_index])
    known_virus_idx = np.unique(virus_index_in_pair[train_index])



    # one-hot
    #使用protein_seq_list_to_ft_mat()函数计算蛋白质的One-Hot编码特征，并将结果存储在protein_ft_dict['antibody_one_hot']和protein_ft_dict['virus_one_hot']中。
    # protein_ft_dict['antibody_one_hot'] = protein_seq_list_to_ft_mat(raw_all_antibody_set, max_antibody_len, ft_type='amino_one_hot')
    # protein_ft_dict['virus_one_hot'] = protein_seq_list_to_ft_mat(raw_all_virus_set, max_virus_len, ft_type='amino_one_hot')

    # # pssm
    # protein_ft_dict['antibody_pssm'], protein_ft_dict['virus_pssm'] = load_pssm_ft_mat()

    # amino_num,使用protein_seq_list_to_ft_mat()函数计算蛋白质的氨基酸数值编码特征，
    protein_ft_dict['antibody_amino_num'] = protein_seq_list_to_ft_mat(raw_all_antibody_set, max_antibody_len, ft_type='amino_num')
    protein_ft_dict['virus_amino_num'] = protein_seq_list_to_ft_mat(raw_all_virus_set, max_virus_len, ft_type='amino_num')

    # k-mer-whole,创建一个KmerTranslator对象，并使用其fit_transform()方法将raw_all_antibody_set和raw_all_virus_set作为输入，进行k-mer特征的转换
    kmer_translator = KmerTranslator(trans_type='std', min_df=kmer_min_df, name=dataset_param_str)
    protein_ft = kmer_translator.fit_transform(raw_all_antibody_set + raw_all_virus_set)
    # kmer_translator.save()
    protein_ft_dict['antibody_kmer_whole'] = protein_ft[0: len(raw_all_antibody_set)]
    protein_ft_dict['virus_kmer_whole'] = protein_ft[len(raw_all_antibody_set):]

    protein_ft_dict['antibody_contact_map'] = process_antibody_files(input_folder_antibody, raw_all_antibody_set)
    protein_ft_dict['virus_contact_map'] = process_virus_files(input_folder_virus, raw_all_virus_set)


    # save,使用np.save()函数将一系列数组和变量保存为Numpy的二进制文件。
    np.save(osp.join(current_path, corpus_dir, 'train_index'), train_index)
    np.save(osp.join(current_path, corpus_dir, 'test_unseen_index'), test_unseen_index)
    np.save(osp.join(current_path, corpus_dir, 'all_label_mat'), all_label_mat)
    np.save(osp.join(current_path, corpus_dir, 'antibody_index_in_pair'), antibody_index_in_pair)
    np.save(osp.join(current_path, corpus_dir, 'virus_index_in_pair'), virus_index_in_pair)
    np.save(osp.join(current_path, corpus_dir, 'known_antibody_idx'), known_antibody_idx)
    np.save(osp.join(current_path, corpus_dir, 'unknown_antibody_idx'), unknown_antibody_idx)
    np.save(osp.join(current_path, corpus_dir, 'known_virus_idx'), known_virus_idx)
    np.save(osp.join(current_path, corpus_dir, 'raw_all_antibody_set_len'), raw_all_antibody_set_len)
    np.save(osp.join(current_path, corpus_dir, 'raw_all_virus_set_len'), raw_all_virus_set_len)

input_folder_antibody = 'D:/workplace/SSRAAI-main\Data/antibody_seq/result'
input_folder_virus = 'D:/workplace/SSRAAI-main/Data/virus_seq/esult'

def process_antibody_files(input_folder_antibody,antibody_set):
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
        # 将列表转换为 NumPy 数组
    antibody_contact_map_array = np.array(antibody_contact_map_list)

    return antibody_contact_map_array

def process_virus_files(input_folder_virus,virus_set):
    call_count = 0
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
        # 将接触图维度调整为 (max_antibody_len, max_antibody_len)
        contact_map = contact_map[:max_virus_len, :max_virus_len]
        # 如果接触图维度小于 max_antibody_len，则进行填充
        if contact_map.shape[0] < max_virus_len:
            pad_width = max_virus_len - contact_map.shape[0]
            contact_map = np.pad(contact_map, ((0, pad_width), (0, pad_width)), mode='constant', constant_values=0)
        virus_contact_map_list.append(contact_map)
        # 将列表转换为 NumPy 数组
        call_count += 1  # 增加调用次数计数器的值
        print(f" Call count: {call_count}")
    virus_contact_map_array = np.array(virus_contact_map_list)

    return virus_contact_map_array

def save_data(protein_ft_dict):

    #使用pickle.dump()函数将protein_ft_dict保存到protein_ft_save_path路径下的文件中。
    with open(protein_ft_save_path, 'wb') as f:
        if sys.version_info > (3, 0):
            pickle.dump(protein_ft_dict, f, protocol=4)
        else:
            pickle.dump(protein_ft_dict, f, protocol=4)

#用于将蛋白质序列列表转换为特征矩阵。
def protein_seq_list_to_ft_mat(protein_seq_list, max_sql_len, ft_type='amino_one_hot'):
    ft_mat = []
    #遍历protein_seq_list中的每个蛋白质序列。
    for protein_seq in protein_seq_list:
        protein_ft = []
        for idx in range(max_sql_len):
            if idx < len(protein_seq):
                amino_name = protein_seq[idx]
            else:
                amino_name = 'pad'
            #根据ft_type的值，选择相应的特征转换方法。
            if 'amino_one_hot' == ft_type:
                amino_ft = amino_one_hot_ft_pad_dict[amino_name]
            elif 'phych' == ft_type:
                amino_ft = amino_physicochemical_ft_pad_dict[amino_name]
            elif 'amino_num' == ft_type:
                amino_ft = amino_map_idx[amino_name]
            # elif 'amino_contact_map' == ft_type:
            #     amino_ft = amino_contact_map_ft_pad_dict[amino_name]   #这个是后加上去得
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
    # process_antibody_files(input_folder_antibody)
    # process_virus_files(input_folder_virus)
    save_data(protein_ft_dict)


