import os
import matplotlib as mpl
import cv2
# mpl.use('Qt5Agg')
from xml.etree import ElementTree as et
import json
from ctlib.coco import COCOTmp

from ctlib.os import *
import os
import torch
import matplotlib.pyplot as plt
import random
from matplotlib import colors  # 注意！为了调整“色盘”，需要导入colors
from sklearn.manifold import TSNE
from tqdm import tqdm
import colorsys
from pathlib import Path
from ctlib.os import *
from tqdm import tqdm
from pathlib import Path
import time

import numpy as np
import torch
import random
import torch.nn as nn
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
from ctlib.vis import plot_embedding_2d
################################### 文本部分 ###########################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from PIL import Image
import torch, open_clip
from ctlib.vis import *
from ctlib.rbox import *
from ctlib.os import *
from copy import deepcopy
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import clip
from SkyScript_open_clip.factory import create_model_and_transforms, \
    create_model_from_pretrained, create_model
from ctlib.transform import to_array

##############################################################
data_root = '/data/space2/huangziyue/xView_800_600'
out_fig_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support_Text_Sims.png'
out_support_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

feat_root = f'{data_root}/Step5_1_Prepare_Classification_feats'
train_split_pth = feat_root + '/train_split.txt'
val_split_pth = feat_root + '/val_split.txt'
train_gt_dir = feat_root + '/train_gt'
val_gt_dir = feat_root + '/val_gt'
cat_file_pth = feat_root + '/categories.txt'

classification_ckpt_pth = ('/data/space2/huangziyue/mmdet_results/'
                           'F3_DETR_Classification_U03_Data6_Xview_DINOv2_VitL_gt_MLP/epoch_100.pth')
###############################################################

ckpt_pth = ('/data/space2/huangziyue/mmdet_checkpoints/'
            'SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt')
model_name = 'ViT-L-14'

model, _, preprocess = create_model_and_transforms(model_name,
                                                   ckpt_pth)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.cuda().eval()
"""
Content requirements: Given some category names, you need to generate phrases in English that indicate that these
objects are in a remote sensing image, such as: "A ship in the aerial image.",
"An aerial image contain the ship". "A ship", and so on.
Each category generates 3 different phrases, you can add some descriptive information, but not too long,
you can imagine the object in different scenes, such as weather, background conditions, etc.,
but do not change the category of the object, the more diverse the better.
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories Has: "classes = ['Aircraft_Hangar', 'Barge', 'Building', 'Bus', 'Cargo_Container_Car',
           'Cargo_Truck', 'Cement_Mixer', 'Construction_Site',
           'Container_Crane', 'Container_Ship', 'Crane_Truck', 'Damaged_Building',
           'Dump_Truck', 'Engineering_Vehicle', 'Excavator', 'Facility', 'Ferry',
           'Fishing_Vessel', 'Fixed-wing_Aircraft', 'Flat_Car', 'Front_loader_Bulldozer',
           'Ground_Grader', 'Haul_Truck', 'Helicopter', 'Helipad', 'Hut_Tent', 'Locomotive',
           'Maritime_Vessel', 'Mobile_Crane', 'Motorboat', 'Oil_Tanker', 'Passenger_Cargo_Plane',
           'Passenger_Car', 'Passenger_Vehicle', 'Pickup_Truck', 'Pylon', 'Railway_Vehicle',
           'Reach_Stacker', 'Sailboat', 'Scraper_Tractor', 'Shed', 'Shipping_Container',
           'Shipping_container_lot', 'Small_Aircraft', 'Small_Car', 'Storage_Tank',
           'Straddle_Carrier', 'Tank_car', 'Tower', 'Tower_crane', 'Trailer', 'Truck',
           'Truck_Tractor', 'Truck_Tractor_w__Box_Trailer', 'Truck_Tractor_w__Flatbed_Trailer',
           'Truck_Tractor_w__Liquid_Tank', 'Tugboat', 'Utility_Truck', 'Vehicle_Lot', 'Yacht']"

Format requirements: Output a dictionary in the following format: {class1=<$'phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""
classes = ['Aircraft_Hangar', 'Barge', 'Building', 'Bus', 'Cargo_Container_Car',
           'Cargo_Truck', 'Cement_Mixer', 'Construction_Site',
           'Container_Crane', 'Container_Ship', 'Crane_Truck', 'Damaged_Building',
           'Dump_Truck', 'Engineering_Vehicle', 'Excavator', 'Facility', 'Ferry',
           'Fishing_Vessel', 'Fixed-wing_Aircraft', 'Flat_Car', 'Front_loader_Bulldozer',
           'Ground_Grader', 'Haul_Truck', 'Helicopter', 'Helipad', 'Hut_Tent', 'Locomotive',
           'Maritime_Vessel', 'Mobile_Crane', 'Motorboat', 'Oil_Tanker', 'Passenger_Cargo_Plane',
           'Passenger_Car', 'Passenger_Vehicle', 'Pickup_Truck', 'Pylon', 'Railway_Vehicle',
           'Reach_Stacker', 'Sailboat', 'Scraper_Tractor', 'Shed', 'Shipping_Container',
           'Shipping_container_lot', 'Small_Aircraft', 'Small_Car', 'Storage_Tank',
           'Straddle_Carrier', 'Tank_car', 'Tower', 'Tower_crane', 'Trailer', 'Truck',
           'Truck_Tractor', 'Truck_Tractor_w__Box_Trailer', 'Truck_Tractor_w__Flatbed_Trailer',
           'Truck_Tractor_w__Liquid_Tank', 'Tugboat', 'Utility_Truck', 'Vehicle_Lot', 'Yacht']

phrases = {
    'Aircraft_Hangar': ['An aerial image shows an aircraft hangar.',
                        'A hangar visible from above houses several planes.',
                        'The image captures an aircraft hangar near the runway.'],
    'Barge': ['A barge floating on the river in the satellite photo.',
              'In the image, a barge is seen moored at the dock.',
              'The aerial view includes a barge moving along the waterway.'],
    'Building': ['A building stands tall in the urban aerial view.',
                 'The satellite image highlights a building in the city center.',
                 'A prominent building is captured from above.'],
    'Bus': ['A bus can be seen in the image, parked near the station.',
            'The aerial photo shows a bus driving on the road.',
            'In the image, a bus is visible next to a bus stop.'],
    'Cargo_Container_Car': ['A cargo container car is visible in the rail yard.',
                            'The image shows a cargo container car on the tracks.',
                            'A cargo container car appears in the aerial view.'],
    'Cargo_Truck': ['A cargo truck is seen driving along the highway.',
                    'In the image, a cargo truck is parked near a warehouse.',
                    'A cargo truck appears in the industrial area.'],
    'Cement_Mixer': ['A cement mixer is visible at the construction site.',
                     'The aerial image captures a cement mixer near a building.',
                     'In the photo, a cement mixer is seen mixing concrete.'],
    'Construction_Site': ['A construction site is bustling with activity in the image.',
                          'The image shows a construction site with heavy machinery.',
                          'An aerial view of a construction site under development.'],
    'Container_Crane': ['A container crane is lifting cargo at the port.',
                        'The image captures a container crane in operation.',
                        'A container crane is seen in the dockyard from above.'],
    'Container_Ship': ['A container ship is docked at the port in the image.',
                       'The aerial photo shows a container ship sailing into the harbor.',
                       'In the image, a container ship is loaded with cargo.'],
    'Crane_Truck': ['A crane truck is parked at the construction site.',
                    'The image shows a crane truck lifting materials.',
                    'A crane truck is seen in the industrial area.'],
    'Damaged_Building': ['A damaged building is visible in the image, with a collapsed roof.',
                         'The aerial photo captures a building with significant structural damage.',
                         'In the image, a damaged building is seen after a disaster.'],
    'Dump_Truck': ['A dump truck is seen dumping materials at the site.',
                   'The image captures a dump truck on a dirt road.',
                   'A dump truck is visible at the construction site.'],
    'Engineering_Vehicle': ['An engineering vehicle is operating at the site in the image.',
                            'The photo shows an engineering vehicle clearing debris.',
                            'An engineering vehicle appears in the construction area.'],
    'Excavator': ['An excavator is digging at the construction site.',
                  'The aerial image shows an excavator moving earth.',
                  'In the photo, an excavator is seen at work.'],
    'Facility': ['A large facility is visible in the industrial complex.',
                 'The image captures a facility with several buildings.',
                 'A facility is seen from above, surrounded by infrastructure.'],
    'Ferry': ['A ferry is crossing the water in the satellite image.',
              'The image shows a ferry docked at the terminal.',
              'In the photo, a ferry is visible transporting vehicles and passengers.'],
    'Fishing_Vessel': ['A fishing vessel is seen near the coastline.',
                       'The aerial image captures a fishing vessel at sea.',
                       'In the image, a fishing vessel is docked at the harbor.'],
    'Fixed-wing_Aircraft': ['A fixed-wing aircraft is seen on the runway.',
                            'The image shows a fixed-wing aircraft preparing for takeoff.',
                            'A fixed-wing aircraft is captured in flight.'],
    'Flat_Car': ['A flat car is seen carrying cargo on the train tracks.',
                 'The aerial image captures a flat car in the rail yard.',
                 'In the photo, a flat car is visible with a load.'],
    'Front_loader_Bulldozer': ['A front loader bulldozer is seen moving dirt at the site.',
                               'The image shows a front loader bulldozer clearing debris.',
                               'In the aerial photo, a front loader bulldozer is in operation.'],
    'Ground_Grader': ['A ground grader is seen leveling the soil.',
                      'The image captures a ground grader at work on a road.',
                      'In the photo, a ground grader is smoothing a construction site.'],
    'Haul_Truck': ['A haul truck is visible transporting materials.',
                   'The aerial image shows a haul truck at the mining site.',
                   'In the photo, a haul truck is seen on a dirt road.'],
    'Helicopter': ['A helicopter is flying over the area in the image.',
                   'The aerial photo captures a helicopter hovering above.',
                   'In the image, a helicopter is seen near a helipad.'],
    'Helipad': ['A helipad is visible in the image, marked with a large H.',
                'The aerial photo shows a helipad next to a building.',
                'In the image, a helipad is seen with a helicopter parked.'],
    'Hut_Tent': ['A hut tent is seen in the makeshift camp.',
                 'The image captures a hut tent in a remote area.',
                 'In the photo, a hut tent is visible among other structures.'],
    'Locomotive': ['A locomotive is pulling a train on the tracks.',
                   'The image shows a locomotive moving through the rail yard.',
                   'In the photo, a locomotive is visible at the station.'],
    'Maritime_Vessel': ['A maritime vessel is sailing through the waterway.',
                        'The image captures a maritime vessel near the coast.',
                        'In the aerial view, a maritime vessel is seen navigating the sea.'],
    'Mobile_Crane': ['A mobile crane is seen lifting heavy equipment.',
                     'The image shows a mobile crane at the construction site.',
                     'In the photo, a mobile crane is visible next to a building.'],
    'Motorboat': ['A motorboat is speeding across the lake in the image.',
                  'The aerial photo captures a motorboat near the shore.',
                  'In the image, a motorboat is seen docked at a pier.'],
    'Oil_Tanker': ['An oil tanker is anchored at the port in the image.',
                   'The aerial photo shows an oil tanker navigating the sea.',
                   'In the image, an oil tanker is visible near the refinery.'],
    'Passenger_Cargo_Plane': ['A passenger cargo plane is seen on the tarmac.',
                              'The image captures a passenger cargo plane loading cargo.',
                              'In the photo, a passenger cargo plane is preparing for departure.'],
    'Passenger_Car': ['A passenger car is parked along the street in the image.',
                      'The aerial photo shows a passenger car driving through the city.',
                      'In the image, a passenger car is visible in a parking lot.'],
    'Passenger_Vehicle': ['A passenger vehicle is seen on the highway.',
                          'The image shows a passenger vehicle parked in a driveway.',
                          'In the photo, a passenger vehicle is visible near a shopping center.'],
    'Pickup_Truck': ['A pickup truck is parked near the construction site.',
                     'The image captures a pickup truck driving off-road.',
                     'In the photo, a pickup truck is seen in a rural area.'],
    'Pylon': ['A pylon stands tall, carrying power lines in the image.',
              'The aerial photo shows a pylon in an open field.',
              'In the image, a pylon is visible next to the road.'],
    'Railway_Vehicle': ['A railway vehicle is seen on the tracks in the image.',
                        'The image captures a railway vehicle moving through the station.',
                        'In the photo, a railway vehicle is visible in the yard.'],
    'Reach_Stacker': ['A reach stacker is moving containers in the port.',
                      'The image shows a reach stacker handling cargo.',
                      'In the photo, a reach stacker is seen at the container terminal.'],
    'Sailboat': ['A sailboat is visible on the calm waters.',
                 'The image captures a sailboat anchored near the shore.',
                 'In the photo, a sailboat is seen sailing under clear skies.'],
    'Scraper_Tractor': ['A scraper tractor is leveling the ground at the site.',
                        'The image shows a scraper tractor in a construction zone.',
                        'In the aerial photo, a scraper tractor is seen working.'],
    'Shed': ['A shed is visible in the backyard in the image.',
             'The image captures a shed in a rural area.',
             'In the photo, a shed is seen next to a house.'],
    'Shipping_Container': ['A shipping container is visible at the dockyard.',
                           'The aerial photo shows a shipping container on a truck.',
                           'In the image, a shipping container is stacked with others.'],
    'Shipping_container_lot': ['A shipping container lot is seen at the port.',
                               'The image captures a shipping container lot filled with containers.',
                               'In the photo, a shipping container lot is visible from above.'],
    'Small_Aircraft': ['A small aircraft is seen on the airfield.',
                       'The image shows a small aircraft parked near a hangar.',
                       'In the photo, a small aircraft is preparing for takeoff.'],
    'Small_Car': ['A small car is parked along the narrow street.',
                  'The aerial photo captures a small car driving through the neighborhood.',
                  'In the image, a small car is visible in a parking space.'],
    'Storage_Tank': ['A storage tank is visible at the industrial site.',
                     'The image shows a storage tank near the refinery.',
                     'In the photo, a storage tank is seen next to other tanks.'],
    'Straddle_Carrier': ['A straddle carrier is moving containers at the port.',
                         'The aerial photo shows a straddle carrier in operation.',
                         'In the image, a straddle carrier is visible lifting cargo.'],
    'Tank_car': ['A tank car is seen on the railway tracks.',
                 'The image captures a tank car in the rail yard.',
                 'In the photo, a tank car is visible transporting liquids.'],
    'Tower': ['A tower is standing tall in the image.',
              'The aerial photo shows a tower with communication equipment.',
              'In the image, a tower is visible in the landscape.'],
    'Tower_crane': ['A tower crane is visible at the construction site.',
                    'The image captures a tower crane lifting materials.',
                    'In the photo, a tower crane is seen next to a high-rise building.'],
    'Trailer': ['A trailer is parked at the rest area.',
               'The image shows a trailer hitched to a truck.',
               'In the photo, a trailer is seen near a construction site.'],
    'Truck': ['A truck is driving along the highway in the image.',
              'The aerial photo captures a truck parked at a rest stop.',
              'In the image, a truck is seen delivering goods.'],
    'Truck_Tractor': ['A truck tractor is seen hauling a trailer.',
                      'The image shows a truck tractor parked at the depot.',
                      'In the photo, a truck tractor is visible at the loading dock.'],
    'Truck_Tractor_w__Box_Trailer': ['A truck tractor with a box trailer is seen on the highway.',
                                     'The image captures a truck tractor with a box trailer parked at a warehouse.',
                                     'In the photo, a truck tractor with a box trailer is visible in the industrial area.'],
    'Truck_Tractor_w__Flatbed_Trailer': ['A truck tractor with a flatbed trailer is transporting machinery.',
                                         'The image shows a truck tractor with a flatbed trailer on the road.',
                                         'In the photo, a truck tractor with a flatbed trailer is seen in a parking lot.'],
    'Truck_Tractor_w__Liquid_Tank': ['A truck tractor with a liquid tank is seen at the refinery.',
                                     'The aerial photo shows a truck tractor with a liquid tank on the highway.',
                                     'In the image, a truck tractor with a liquid tank is visible at the gas station.'],
    'Tugboat': ['A tugboat is guiding a ship in the harbor.',
                'The image captures a tugboat near the port.',
                'In the photo, a tugboat is seen assisting a larger vessel.'],
    'Utility_Truck': ['A utility truck is parked next to the building.',
                      'The image shows a utility truck on a service call.',
                      'In the photo, a utility truck is visible near the power lines.'],
    'Vehicle_Lot': ['A vehicle lot is filled with cars in the image.',
                    'The aerial photo captures a vehicle lot next to a dealership.',
                    'In the image, a vehicle lot is seen with rows of parked cars.'],
    'Yacht': ['A yacht is sailing on the open sea in the image.',
              'The image shows a yacht anchored near the marina.',
              'In the photo, a yacht is visible cruising along the coast.']
}



row_lables = []
support_data = dict()
for class_name, texts in phrases.items():
    row_lables.extend([class_name,] * len(texts))
    text = tokenizer(texts)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.cuda()).detach().cpu()
    text_embeds = to_array(text_features)
    text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)
    support_data[class_name] = dict(
        texts=texts,
        text_embeds=text_embeds)

################################### 视觉部分 ###########################
class head(nn.Module):
    def __init__(self):
        super(head, self).__init__()
        self.fc = nn.Linear(1024, num_classes)
    def forward(self, x):
        return self.fc(x)
class cls_model(torch.nn.Module):
    def __init__(self):
        super(cls_model, self).__init__()
        self.neck = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.head = head()
    def forward(self, x):
        x = self.neck(x)
        cls_score = self.head(x)
        return cls_score


num_classes = len(classes)

classifier = cls_model()
classifier.load_state_dict(torch.load(classification_ckpt_pth)['state_dict'])
classifier.eval()

max_support_shot = 50
min_score_thr = 0.5
# 好像是按照排序后的CLASSES来获得ID的
for cls_id, cls_name in enumerate(sorted(classes)):
    cls_feats = []
    cls_dir = f'{val_gt_dir}/{cls_name}'
    for feat_file in os.listdir(f'{val_gt_dir}/{cls_name}'):
        feat_pth = f'{cls_dir}/{feat_file}'
        feat = pklload(feat_pth, msg=False)
        cls_feats.append(feat)

    #### ---- Add train if not enough feats
    if len(cls_feats) < 50:
        cls_dir = f'{train_gt_dir}/{cls_name}'
        for feat_file in os.listdir(f'{train_gt_dir}/{cls_name}'):
            feat_pth = f'{cls_dir}/{feat_file}'
            feat = pklload(feat_pth, msg=False)
            cls_feats.append(feat)
    cls_feats = np.stack(cls_feats)
    cls_feats = torch.Tensor(cls_feats)
    cls_scores = classifier(cls_feats).softmax(dim=1)[:, cls_id]
    sort_idx = torch.argsort(-cls_scores)

    selected_idx = sort_idx[:max_support_shot]
    confidence = cls_scores[selected_idx]
    embeds = cls_feats[selected_idx]

    pos_idx = confidence >= min_score_thr
    if torch.sum(pos_idx) != 0:
        confidence = confidence[pos_idx]
        embeds = embeds[pos_idx]
    ######## Padding
    if len(embeds) != max_support_shot:
        embeds_pad = torch.cat([embeds for i in range(max_support_shot)])
        embeds = embeds_pad[:max_support_shot]
        confidence_pad = torch.cat([confidence for i in range(max_support_shot)])
        confidence = confidence_pad[:max_support_shot]

    if cls_name not in support_data.keys():
        raise Exception(f'{cls_name} not in visual embeds')
    support_data[cls_name]['visual_embeds'] = embeds.detach().cpu().numpy()
    support_data[cls_name]['confidence_scores'] = confidence.detach().cpu().numpy()
pklsave(support_data, out_support_pth)

############################# 可视化分析 ##################################

all_text_features = np.concatenate([info['text_embeds'] for cls_name, info in support_data.items()])
text_sims = np.matmul(all_text_features, all_text_features.T)
text_sims = np.around(text_sims, 2)
print(text_sims)
print('Min', np.min(text_sims))
print('Max-Min', np.max(text_sims) - np.min(text_sims))
fig, ax = plt.subplots(figsize=(128,128))
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.set(font_scale=1.25)
hm = sns.heatmap(text_sims,
                 cbar=False,
                 annot=True, # 注入数字
                 # square=True, # 单元格为正方形
                 fmt='.2f',   # 字符串格式代码
                  annot_kws={'size': 10}, # 当annot为True时，ax.text的关键字参数，即注入数字的字体大小
                  yticklabels=row_lables,  # 列标签
                  xticklabels=row_lables   # 行标签
                  )
plt.savefig(out_fig_pth)
plt.close()







