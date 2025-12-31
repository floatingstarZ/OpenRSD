import os
from EXP_CONFIG.CONFIGS.base import Project_root
root = Project_root + '/M_configs/'
# CNData
# DOTA
cfg_dir = 'G02_Baselines/Data12_ShipImageNet'
# OD030_Oriented_Cascade_RCNN
# OD000_Baseline
# OD031_Part_Aware
# OD032_RelationNetwork
# OD033_Oriented_Cascade_RCNN_part2
# OD034_KMeans_Memory_Attention
root += cfg_dir

def check_list(s, word_list):
    for word in word_list:
        if word in s:
            return True
    return False
# Types: Model_cfg, Table_cfg
generate_type = 'Model_cfg'
key_word = 'G02'

for file in sorted(list(os.listdir(root))):
    fp = root + '/' + file
    if os.path.isdir(fp):
        continue
    if key_word not in file:
        continue
    epoch = 12
    gpu_num = 2
    if 'RTMDet' in file:
        epoch = 36
    if 'ViT_L' in file:
        gpu_num = 4

    if generate_type == 'Model_cfg':
        file_name = os.path.splitext(file)[0]
        tmp = f'    gen_dict("{file_name}",\n' \
              f'             root + "{cfg_dir}/" + "{file_name}.py",\n' \
              f'             epoch={epoch}, gpu_num={gpu_num}),'
        print(tmp)


