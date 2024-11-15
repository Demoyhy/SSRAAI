import pickle
import numpy as np


# 加载.pkl文件
with open('/data0/yanghy/workplace/DeepAAI-main/dataset/corpus/processed_mat/new_pkl_file1.pkl', 'rb') as f:
    data = pickle.load(f)
# # 确定需要转换的键名
# target_key = 'antibody_contact_map'
#
# # 检查键是否存在以及对应值是否为列表类型
# if target_key in data_dict and isinstance(data_dict[target_key], list):
#     # 将列表值转换为ndarray
#     data_dict[target_key] = np.array(data_dict[target_key])
#
# # 将修改后的字典保存回原来的pickle文件
# with open('/data0/yanghy/workplace/DeepAAI-main/dataset/corpus/processed_mat/new_pkl_file.pkl', 'wb') as file:
#     pickle.dump(data_dict, file)

# 处理数据
# ...
# 打印数据
print(data)
keys = list(data.keys())
print("字典的键：")
for key in keys:
    print(key)

# # 新建一个字典，用于存储处理后的数据
# new_data = {}

for key, value in data.items():
    key_type = type(key).__name__
    value_type = type(value).__name__
    value_size = len(value)
    # print(f"键 '{key}' 的类型是：{key_type}，值 '{value}' 的类型是：{value_type}")
    print(f"键 '{key}' 的类型是：{key_type}，值的类型是：{value_type},值的大小是：{value_size}")

#     if key == 'antibody_contact_map':
#         flat_array = np.vstack(value)
#         # 将处理后的数组保存到新的字典中
#         new_data[key] = flat_array
#     else:
#         # 对于其他键，直接保存到新的字典中
#         new_data[key] = value
#
# # 将修改后的字典保存回原来的pickle文件
# with open('/data0/yanghy/workplace/DeepAAI-main/dataset/corpus/new_pkl_file.pkl', 'wb') as file:
#     pickle.dump(data, file)

# value = data['virus_pssm']
# print(value)
