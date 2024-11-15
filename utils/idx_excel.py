import os
import pandas as pd

# 读取Excel文件
df = pd.read_excel('/data0/yanghy/workplace/DeepAAI-main/processing/cov_cls/dataset_cov_cls.xlsx')

# 获取fasta文件夹路径
fasta_folder = '/data0/yanghy/workplace/igfold/cov_antibody_seq/results/'  # 将路径替换为存放fasta文件的文件夹的路径

# 创建一个字典，将序列与fasta文件名对应起来
fasta_dict = {}

# 初始化计数器
iteration_count = 0

# 遍历fasta文件夹中的所有fasta文件
for file_name in os.listdir(fasta_folder):
    iteration_count += 1
    merged_sequence = ''
    if file_name.endswith('.fasta'):
        fasta_file = os.path.join(fasta_folder, file_name)
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith(">H") and not line.startswith(">L"):
                    merged_sequence += line.strip()
                    merged_seq = merged_sequence[:30]
            # sequence = ''.join([line.strip() for line in lines[1:]])  # 连接重链和轻链序列
            if merged_seq in df['antibody_seq'].str[:30].values:
                seq_number = int(file_name.split('.')[0])
                fasta_dict[merged_seq] = seq_number
                print(f"Iteration: {iteration_count}")
print(fasta_dict)
# 创建一个新的列antibody_pdb，根据antibody_seq列的值来映射fasta文件名
df['antibody_pdb'] = df['antibody_seq'].str[:30].map(fasta_dict)

# 保存修改后的Excel文件
df.to_excel('/data0/yanghy/workplace/DeepAAI-main/processing/cov_cls/dataset_hiv_cls6.xlsx', index=False)



