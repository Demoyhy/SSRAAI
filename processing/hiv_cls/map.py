import Bio.PDB
import numpy as np
import os
from Bio.PDB.Residue import Residue


def get_center_atom(residue):
    if residue.has_id('CA'):
        c_atom = 'CA'
    elif residue.has_id('N'):
        c_atom = 'N'
    elif residue.has_id('C'):
        c_atom = 'C'
    elif residue.has_id('O'):
        c_atom = 'O'
    elif residue.has_id('CB'):
        c_atom = 'CB'
    elif residue.has_id('CD'):
        c_atom = 'CD'
    else:
        c_atom = 'CG'
    return c_atom


aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
}

def create_new_residue(residue, new_id):
    new_residue = Bio.PDB.Residue.Residue.__new__(Bio.PDB.Residue.Residue)
    new_residue.__dict__.update(residue.__dict__)
    new_residue.id = new_id
    for atom in residue:
        new_atom = atom.copy()
        new_residue.add(new_atom)

    return new_residue



#用于计算两个残基之间的Cα距离。
def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""

    #调用get_center_atom函数获取两个残基的中心原子名称
    c_atom1 = get_center_atom(residue_one)
    c_atom2 = get_center_atom(residue_two)
    #它计算两个中心原子的坐标差向量（diff_vector）
    diff_vector  = residue_one[c_atom1].coord - residue_two[c_atom2].coord
    #并使用欧几里德距离公式计算该向量的长度。最后，函数返回Cα距离。
    return np.sqrt(np.sum(diff_vector * diff_vector))

#用于计算两条链（chain_one和chain_two）之间的Cα距离矩阵
def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    residue_len = 0
    #函数首先计算链中有效的氨基酸残基数
    for row, residue_one in enumerate(chain_one):
        hetfield = residue_one.get_id()[0]
        hetname = residue_one.get_resname()
        if hetfield == " " and hetname in aa_codes.keys():
            residue_len = residue_len + 1
    #函数初始化一个全零浮点数矩阵answer，用于存储距离值。
    answer = np.zeros((residue_len, residue_len), float)
    x = -1
    for residue_one in chain_one:
        y = -1
        hetfield1 = residue_one.get_id()[0]
        hetname1 = residue_one.get_resname()
        if hetfield1 == ' ' and hetname1 in aa_codes.keys():
            x = x + 1
            for residue_two in chain_two:
                hetfield2 = residue_two.get_id()[0]
                hetname2 = residue_two.get_resname()
                if hetfield2 == ' ' and hetname2 in aa_codes.keys():
                    y = y + 1
                    #根据x和y的值，将calc_residue_dist函数计算的Cα距离存储在矩阵answer的相应位置上。
                    answer[x, y] = calc_residue_dist(residue_one, residue_two)
    #循环结束后，函数将距离矩阵对角线上的元素设置为100，并返回距离矩阵。
    for i in range(residue_len):
        answer[i,i] = 100
    return answer


# def calc_contact_map(pdb_path,chain_id):
    # pdb_path = data_root_path + pdb_id + '.pdb'
    # structure = Bio.PDB.PDBParser().get_structure("structure", pdb_path)
    # model = structure[0]
    # dist_matrix = calc_dist_matrix(model[chain_id], model[chain_id])
    # contact_map = (dist_matrix < 8.0).astype(int)
    # #print('contact map shape:',contact_map.shape)
    # return contact_map

def calc_contact_map(pdb_path):
    structure = Bio.PDB.PDBParser().get_structure("structure", pdb_path)
    model = structure[0]
    dist_matrix = calc_dist_matrix(model.get_residues(), model.get_residues())
    contact_map = (dist_matrix < 8.0).astype(int)
    return contact_map



def batch_process_pdb_files(input_folder, output_folder):
    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        # 检查文件扩展名是否为pdb
        if file_name.endswith('.pdb'):
            # 构建PDB文件的完整路径
            pdb_path = os.path.join(input_folder, file_name)

            # 从文件名中提取PDB ID，如果需要的话
            pdb_id = os.path.splitext(file_name)[0]

            # 解析PDB文件，获取所有链的ID
            structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_path)
            chain_ids = [chain.get_id() for chain in structure.get_chains()]

            # 遍历每个链，计算并保存接触图
            for chain_id in chain_ids:
                # 定义输出文件路径，使用PDB ID和链ID作为文件名
                output_file = os.path.join(output_folder, f'{pdb_id}_{chain_id}.npz')

                # 调用calc_contact_map()函数计算接触图
                contact_map = calc_contact_map(pdb_path, chain_id)
#
#                 # 保存接触图为NPZ文件
#                 np.savez(output_file,contact=contact_map)

#
#
# input_folder = 'D:/workplace/demo/TAGPPI-main/pdb_data/'
# output_folder = 'D:/workplace/demo/TAGPPI-main/pdb_data/'
# batch_process_pdb_files(input_folder, output_folder)







# # for example
# sequence = 'MQIVMFDRQSIFIHGMKISLQQRIPGVSIQGASQADELWQKL'
# # data_root_path = 'your_data_root_path'
# data_root_path = 'D:/workplace/demo/TAGPPI-main/pdb_data/'
# pdb_id = 'my_antibody'
# contact_map = calc_contact_map(pdb_id,'L')
# #声明了一个变量 contact_file，并赋值为保存接触图的文件路径和文件名。
# contact_file = 'my_pdb1.npz'
# np.savez(contact_file,seq = sequence, contact = contact_map)

# contact = np.load(contact_file)
# print(contact.files)
# print(contact['seq'])
# print(contact['contact'])
