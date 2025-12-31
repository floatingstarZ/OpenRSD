from ctlib.os import *
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ctlib.dota import load_dota
# https://github.com/DIUx-xView/xView1_baseline

np.random.seed(2024)

# ------ 从定义的Label Space获得的可行的标签
Data9_FMoW = {
    'airport': 'transportation_area',  # 机场是与交通相关的区域。
    'airport_hangar': 'building',  # 机场机库属于建筑物类别。
    'airport_terminal': 'building',  # 机场航站楼属于建筑物类别。
    'amusement_park': 'public_area',  # 游乐园是公共区域的一部分。
    # 'aquaculture': 'agricultural_land',  # 水产养殖场属于农业用地。
    'archaeological_site': 'public_area',  # 考古遗址属于公共区域。
    # 'barn': 'building',  # 谷仓是一种建筑物。
    'border_checkpoint': 'transportation_area',  # 边境检查站是与交通相关的区域。
    'burial_site': 'public_area',  # 埋葬地点属于公共区域。
    'car_dealership': 'communication_facility',  # 汽车经销商属于商业设施。
    'construction_site': 'industrial_facility',  # 建筑工地是与工业相关的区域。
    # 'crop_field': 'agricultural_land',  # 农田属于农业用地。
    'dam': 'water_area',  # 水坝是与水相关的区域。
    # 'debris_or_rubble': 'public_area',  # 碎片或瓦砾属于公共区域的一部分，可能由于灾害。
    'educational_institution': 'public_area',  # 教育机构（如学校）属于公共区域。
    'electric_substation': 'energy_facility',  # 电力变电站属于能源设施。
    'factory_or_powerplant': 'industrial_facility',  # 工厂或电厂属于工业设施。
    # 'fire_station': 'public_area',  # 消防站属于公共服务设施。
    # 'flooded_road': 'transportation_area',  # 被淹没的道路属于交通区域。
    'fountain': 'water_area',  # 喷泉属于水域。
    'gas_station': 'transportation_area',  # 加油站是与交通相关的区域。
    'golf_course': 'sports_facility',  # 高尔夫球场属于体育设施。
    'ground_transportation_station': 'transportation_area',  # 地面交通站点（如公交车站、火车站）属于交通区域。
    'helipad': 'transportation_area',  # 停机坪是与交通相关的区域。
    # 'hospital': 'public_area',  # 医院属于公共服务设施。
    # 'impoverished_settlement': 'public_area',  # 贫困定居点属于公共区域的一部分。
    'interchange': 'transportation_area',  # 交通交汇处属于交通区域。
    'lake_or_pond': 'water_area',  # 湖泊或池塘属于水利结构区域。
    'lighthouse': 'water_area',  # 灯塔属于与水相关的区域。
    'military_facility': 'public_area',  # 军事设施通常被视为公共区域的一部分。
    # 'multi-unit_residential': 'building',  # 多单元住宅属于建筑物类别。
    'nuclear_powerplant': 'energy_facility',  # 核电厂属于能源设施。
    # 'office_building': 'communication_facility',  # 写字楼属于商业设施。
    'oil_or_gas_facility': 'industrial_facility',  # 石油或天然气设施属于工业设施。
    'park': 'public_area',  # 公园属于公共区域。
    'parking_lot_or_garage': 'transportation_area',  # 停车场或车库是与交通相关的区域。
    # 'place_of_worship': 'public_area',  # 礼拜场所（如教堂、寺庙）属于公共区域。
    # 'police_station': 'public_area',  # 警察局属于公共服务设施。
    'port': 'water_area',  # 港口是与交通相关的区域，划分为水域。
    # 'prison': 'public_area',  # 监狱属于公共区域的一部分。
    'race_track': 'sports_facility',  # 赛道属于体育设施。
    'railway_bridge': 'transportation_area',  # 铁路桥属于交通区域。
    'recreational_facility': 'public_area',  # 娱乐设施属于公共区域。
    'road_bridge': 'transportation_area',  # 公路桥属于交通区域。
    'runway': 'transportation_area',  # 跑道属于与交通相关的区域。
    'shipyard': 'industrial_facility',  # 造船厂属于工业设施。
    'shopping_mall': 'public_area',  # 购物中心属于公共区域。
    # 'single-unit_residential': 'building',  # 单户住宅属于建筑物类别。
    'smokestack': 'industrial_facility',  # 烟囱属于工业设施。
    'solar_farm': 'energy_facility',  # 太阳能农场属于能源设施。
    'space_facility': 'public_area',  # 太空设施通常被视为公共区域的一部分。
    'stadium': 'sports_facility'  # 体育场属于体育设施。
}

valid_names = list(Data9_FMoW.keys())
print('Valid Names:', valid_names)

for part_id in [0, 1, 2, 3, 4, 5]:
    data_root = '/data/space2/huangziyue/FMoW'
    img_dir = f'{data_root}/FMoW_Part{part_id}/images'
    ann_dir = f'{data_root}/FMoW_Part{part_id}/Step1_Trans_HBB2OBB'
    embed_dir = f'{data_root}/FMoW_Part{part_id}/Step4_Extract_DINOv2_Embeds_8_3_GT'

    # 去掉无意义
    ann_files = sorted(list(os.listdir(ann_dir)))
    valid_ann_files = []
    for ann_file in tqdm(ann_files):
        try:
            ann_pth = ann_dir + '/' + ann_file
            with open(ann_pth) as f:
                lines = f.readlines()
                lines = [l.strip().split(' ') for l in lines]
            gt_polys = []
            gt_names = []
            for l in lines:
                poly = [float(coord) for coord in l[:8]]
                poly = np.array(poly)
                gt_polys.append(poly)
                gt_names.append(l[8])
        except:
            print(f'Error in {ann_file}')
            continue
        if gt_names[0] not in valid_names: # 只有一个物体
            # print(f'Pass invalid: {ann_file}')
            continue
        valid_ann_files.append(ann_file)
    valid_ann_files = sorted(valid_ann_files)

    # 20%测试, 80%训练
    ann_files = np.random.permutation(np.array(valid_ann_files))
    n_test = int(0.3 * len(ann_files))

    test_files = sorted(ann_files[:n_test].tolist())
    train_files = sorted(ann_files[n_test:].tolist())

    for files, split in zip([train_files, test_files], ['train', 'test']):
        split_data_root = f'{data_root}/{split}'
        mkdir(split_data_root)
        split_img_dir = f'{split_data_root}/images'
        split_ann_dir = f'{split_data_root}/labelTxt'
        # split_embed_dir = f'{split_data_root}/vis_embeds'
        mkdir(split_ann_dir)
        mkdir(split_img_dir)
        # mkdir(split_embed_dir)
        file_split = [Path(f).stem for f in files]
        split_file_pth = f'{split_data_root}/split.txt'
        with open(split_file_pth, 'wt+') as f:
            for file in file_split:
                f.write(f'{file}\n')
        for file in tqdm(file_split):
            src_ann_pth = f'{ann_dir}/{file}.txt'
            tgt_ann_pth = f'{split_ann_dir}/{file}.txt'

            src_img_pth = f'{img_dir}/{file}.png'
            tgt_img_pth = f'{split_img_dir}/{file}.png'

            # src_embed_pth = f'{embed_dir}/{file}.pkl'
            # tgt_embed_pth = f'{split_embed_dir}/{file}.pkl'

            if not os.path.exists(src_ann_pth):
                print(f'Ann File {src_ann_pth} not exist')
                continue
            if not os.path.exists(src_img_pth):
                print(f'Image File {src_img_pth} not exist')
                continue
            if os.path.exists(tgt_ann_pth) and os.path.exists(tgt_img_pth):
                continue
            # if not os.path.exists(src_embed_pth):
            #     print(f'Embed File {src_embed_pth} not exist')
            #     continue
            os.system(f'cp {src_ann_pth} {tgt_ann_pth}')
            os.system(f'cp {src_img_pth} {tgt_img_pth}')
            # os.system(f'cp {src_embed_pth} {tgt_embed_pth}')
