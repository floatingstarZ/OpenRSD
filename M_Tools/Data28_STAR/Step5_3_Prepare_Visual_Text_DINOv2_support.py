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
data_root = '/data/space2/huangziyue/STAR_800_200/val'
out_fig_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support_Text_Sims.png'
out_support_pth = f'{data_root}/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl'

feat_root = f'{data_root}/Step5_1_Prepare_Classification_feats'
train_split_pth = feat_root + '/train_split.txt'
val_split_pth = feat_root + '/val_split.txt'
train_gt_dir = feat_root + '/train_gt'
val_gt_dir = feat_root + '/val_gt'
cat_file_pth = feat_root + '/categories.txt'

classification_ckpt_pth = ('/data/space2/huangziyue/mmdet_results/'
                           'F3_DETR_Classification_U03_Data28_STAR_DINOv2_VitL_gt_MLP/epoch_50.pth')
###############################################################

ckpt_pth = ('/data/space2/huangziyue/mmdet_checkpoints/'
            'SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt')
model_name = 'ViT-L-14'

model, _, preprocess = create_model_and_transforms(model_name,
                                                   ckpt_pth)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.cuda().eval()
"""

Content requirements: Given a category dict C and a basic template list T, 
you need to generate phrases in English that indicate that these objects are in a remote sensing image. 
First, you need to complete categories into a complete category name based on categories and their parent classes. 
Each category generates 15 different phrases, the more diverse the better, and the template in T must be all used. 
The generated phrase must explicitly include this category and be free of grammatical errors.
The generated phrases need to clearly reflect the differences between the categories.

Categories list C: "classes = ['airplane', 'apron', 'arch_dam', 'basketball_court', 
'boarding_bridge', 'boat', 'breakwater', 'bridge', 'car', 'car_parking',
 'cement_concrete_pavement', 'chimney', 'coal_yard', 'containment_vessel', 
 'cooling_tower', 'crane', 'dock', 'engineering_vehicle', 'flood_dam', 'foundation_pit', 
 'gas_station', 'genset', 'goods_yard', 'gravity_dam', 'ground_track_field', 'intersection',
  'lattice_tower', 'roundabout', 'runway', 'ship', 'smoke', 'soccer_ball_field', 'storehouse',
   'substation', 'tank', 'taxiway', 'tennis_court', 'terminal', 'toll_gate', 
   'tower_crane', 'truck', 'truck_parking', 'unfinished_building', 'vapor', 'wind_mill']"

where the keys are classes, and the values are the parent classes.
                    
base_templates = [
    'A satellite photo of a {class}',
    'An aerial image of a {class}',
    '{class name (completed)}',
    '{class}',
    'A satellite photo of a {class}, a type of {parent_class}',
    'A satellite photo of a {class}, {description of the class}',
    'An overhead view capturing a {class}',
    'A high-resolution satellite image of a {class}',
    '{description of the class}',
]
Format requirements: Output a dictionary in the following format: 
{class1=['phrase 1','phrase 2',...],
class2=['Phrase 1',' Phrase 2',...]},
The output dictionary must be able to be compiled in python.
Output forms need to be diverse, and there should be no consistent rhetoric.

"""
classes = ['airplane', 'apron', 'arch_dam',
           'basketball_court', 'boarding_bridge', 'boat', 'breakwater',
           'bridge', 'car', 'car_parking', 'cement_concrete_pavement',
           'chimney', 'coal_yard', 'containment_vessel', 'cooling_tower',
           'crane', 'dock', 'engineering_vehicle', 'flood_dam', 'foundation_pit',
           'gas_station', 'genset', 'goods_yard', 'gravity_dam', 'ground_track_field',
           'intersection', 'lattice_tower', 'roundabout', 'runway', 'ship',
           'smoke', 'soccer_ball_field', 'storehouse', 'substation', 'tank',
           'taxiway', 'tennis_court', 'terminal', 'toll_gate', 'tower_crane',
           'truck', 'truck_parking', 'unfinished_building', 'vapor', 'wind_mill']


phrases = {
    'airplane': [
        'A satellite photo of an airplane',
        'An aerial image of an airplane',
        'Airplane, a type of aircraft',
        'Airplane',
        'A satellite photo of an airplane, a type of aircraft',
        'A satellite photo of an airplane, a vehicle designed for air travel',
        'An overhead view capturing an airplane',
        'A high-resolution satellite image of an airplane',
        'A vehicle designed for air travel',
        'A satellite image showing an airplane on a runway',
        'An aerial view of an airplane parked at an airport',
        'A satellite photo of an airplane flying at high altitude',
        'An airplane captured in a remote sensing image',
        'A satellite image of an airplane taxiing on the tarmac',
        'An aerial photograph of an airplane in flight'
    ],
    'apron': [
        'A satellite photo of an apron',
        'An aerial image of an apron',
        'Apron, a type of airport infrastructure',
        'Apron',
        'A satellite photo of an apron, a type of airport infrastructure',
        'A satellite photo of an apron, a paved area for aircraft parking',
        'An overhead view capturing an apron',
        'A high-resolution satellite image of an apron',
        'A paved area for aircraft parking',
        'A satellite image showing an apron at an airport',
        'An aerial view of an apron with parked airplanes',
        'A satellite photo of an apron adjacent to a terminal',
        'An apron captured in a remote sensing image',
        'A satellite image of an apron with ground service vehicles',
        'An aerial photograph of an apron at a busy airport'
    ],
    'arch_dam': [
        'A satellite photo of an arch dam',
        'An aerial image of an arch dam',
        'Arch Dam, a type of dam',
        'Arch Dam',
        'A satellite photo of an arch dam, a type of dam',
        'A satellite photo of an arch dam, a curved structure for water retention',
        'An overhead view capturing an arch dam',
        'A high-resolution satellite image of an arch dam',
        'A curved structure for water retention',
        'A satellite image showing an arch dam in a mountainous region',
        'An aerial view of an arch dam with a reservoir',
        'A satellite photo of an arch dam spanning a river',
        'An arch dam captured in a remote sensing image',
        'A satellite image of an arch dam with water flowing through',
        'An aerial photograph of an arch dam in a remote area'
    ],
    'basketball_court': [
        'A satellite photo of a basketball court',
        'An aerial image of a basketball court',
        'Basketball Court, a type of sports facility',
        'Basketball Court',
        'A satellite photo of a basketball court, a type of sports facility',
        'A satellite photo of a basketball court, a rectangular area for playing basketball',
        'An overhead view capturing a basketball court',
        'A high-resolution satellite image of a basketball court',
        'A rectangular area for playing basketball',
        'A satellite image showing a basketball court in a park',
        'An aerial view of a basketball court with players',
        'A satellite photo of a basketball court in an urban area',
        'A basketball court captured in a remote sensing image',
        'A satellite image of a basketball court surrounded by trees',
        'An aerial photograph of a basketball court at a school'
    ],
    'boarding_bridge': [
        'A satellite photo of a boarding bridge',
        'An aerial image of a boarding bridge',
        'Boarding Bridge, a type of airport infrastructure',
        'Boarding Bridge',
        'A satellite photo of a boarding bridge, a type of airport infrastructure',
        'A satellite photo of a boarding bridge, a movable connector for passengers',
        'An overhead view capturing a boarding bridge',
        'A high-resolution satellite image of a boarding bridge',
        'A movable connector for passengers',
        'A satellite image showing a boarding bridge at an airport terminal',
        'An aerial view of a boarding bridge connected to an airplane',
        'A satellite photo of a boarding bridge in use',
        'A boarding bridge captured in a remote sensing image',
        'A satellite image of a boarding bridge with passengers boarding',
        'An aerial photograph of a boarding bridge at a busy airport'
    ],
    'boat': [
        'A satellite photo of a boat',
        'An aerial image of a boat',
        'Boat, a type of watercraft',
        'Boat',
        'A satellite photo of a boat, a type of watercraft',
        'A satellite photo of a boat, a small vessel for traveling on water',
        'An overhead view capturing a boat',
        'A high-resolution satellite image of a boat',
        'A small vessel for traveling on water',
        'A satellite image showing a boat on a lake',
        'An aerial view of a boat sailing in the ocean',
        'A satellite photo of a boat docked at a marina',
        'A boat captured in a remote sensing image',
        'A satellite image of a boat with a wake behind it',
        'An aerial photograph of a boat in a river'
    ],
    'breakwater': [
        'A satellite photo of a breakwater',
        'An aerial image of a breakwater',
        'Breakwater, a type of coastal structure',
        'Breakwater',
        'A satellite photo of a breakwater, a type of coastal structure',
        'A satellite photo of a breakwater, a barrier protecting a harbor',
        'An overhead view capturing a breakwater',
        'A high-resolution satellite image of a breakwater',
        'A barrier protecting a harbor',
        'A satellite image showing a breakwater along a coastline',
        'An aerial view of a breakwater with waves breaking against it',
        'A satellite photo of a breakwater extending into the sea',
        'A breakwater captured in a remote sensing image',
        'A satellite image of a breakwater with calm waters behind it',
        'An aerial photograph of a breakwater in a busy port'
    ],
    'bridge': [
        'A satellite photo of a bridge',
        'An aerial image of a bridge',
        'Bridge, a type of infrastructure',
        'Bridge',
        'A satellite photo of a bridge, a type of infrastructure',
        'A satellite photo of a bridge, a structure spanning a physical obstacle',
        'An overhead view capturing a bridge',
        'A high-resolution satellite image of a bridge',
        'A structure spanning a physical obstacle',
        'A satellite image showing a bridge over a river',
        'An aerial view of a bridge connecting two land masses',
        'A satellite photo of a bridge with traffic',
        'A bridge captured in a remote sensing image',
        'A satellite image of a bridge in a mountainous region',
        'An aerial photograph of a bridge in an urban area'
    ],
    'car': [
        'A satellite photo of a car',
        'An aerial image of a car',
        'Car, a type of vehicle',
        'Car',
        'A satellite photo of a car, a type of vehicle',
        'A satellite photo of a car, a road vehicle for personal transportation',
        'An overhead view capturing a car',
        'A high-resolution satellite image of a car',
        'A road vehicle for personal transportation',
        'A satellite image showing a car on a highway',
        'An aerial view of a car parked in a driveway',
        'A satellite photo of a car in a parking lot',
        'A car captured in a remote sensing image',
        'A satellite image of a car driving through a city',
        'An aerial photograph of a car on a rural road'
    ],
    'car_parking': [
        'A satellite photo of a car parking',
        'An aerial image of a car parking',
        'Car Parking, a type of parking facility',
        'Car Parking',
        'A satellite photo of a car parking, a type of parking facility',
        'A satellite photo of a car parking, an area designated for parking vehicles',
        'An overhead view capturing a car parking',
        'A high-resolution satellite image of a car parking',
        'An area designated for parking vehicles',
        'A satellite image showing a car parking lot',
        'An aerial view of a car parking with multiple vehicles',
        'A satellite photo of a car parking in an urban area',
        'A car parking captured in a remote sensing image',
        'A satellite image of a car parking at a shopping mall',
        'An aerial photograph of a car parking at an airport'
    ],
    'cement_concrete_pavement': [
        'A satellite photo of a cement concrete pavement',
        'An aerial image of a cement concrete pavement',
        'Cement Concrete Pavement, a type of pavement',
        'Cement Concrete Pavement',
        'A satellite photo of a cement concrete pavement, a type of pavement',
        'A satellite photo of a cement concrete pavement, a durable surface for roads',
        'An overhead view capturing a cement concrete pavement',
        'A high-resolution satellite image of a cement concrete pavement',
        'A durable surface for roads',
        'A satellite image showing a cement concrete pavement on a highway',
        'An aerial view of a cement concrete pavement in an urban area',
        'A satellite photo of a cement concrete pavement with traffic',
        'A cement concrete pavement captured in a remote sensing image',
        'A satellite image of a cement concrete pavement in a residential area',
        'An aerial photograph of a cement concrete pavement in a rural area'
    ],
    'chimney': [
        'A satellite photo of a chimney',
        'An aerial image of a chimney',
        'Chimney, a type of industrial structure',
        'Chimney',
        'A satellite photo of a chimney, a type of industrial structure',
        'A satellite photo of a chimney, a vertical structure for venting smoke',
        'An overhead view capturing a chimney',
        'A high-resolution satellite image of a chimney',
        'A vertical structure for venting smoke',
        'A satellite image showing a chimney at a factory',
        'An aerial view of a chimney with smoke rising',
        'A satellite photo of a chimney in an industrial area',
        'A chimney captured in a remote sensing image',
        'A satellite image of a chimney in a power plant',
        'An aerial photograph of a chimney in a residential area'
    ],
    'coal_yard': [
        'A satellite photo of a coal yard',
        'An aerial image of a coal yard',
        'Coal Yard, a type of storage facility',
        'Coal Yard',
        'A satellite photo of a coal yard, a type of storage facility',
        'A satellite photo of a coal yard, an area for storing coal',
        'An overhead view capturing a coal yard',
        'A high-resolution satellite image of a coal yard',
        'An area for storing coal',
        'A satellite image showing a coal yard at a power plant',
        'An aerial view of a coal yard with piles of coal',
        'A satellite photo of a coal yard in an industrial area',
        'A coal yard captured in a remote sensing image',
        'A satellite image of a coal yard with machinery',
        'An aerial photograph of a coal yard in a rural area'
    ],
    'containment_vessel': [
        'A satellite photo of a containment vessel',
        'An aerial image of a containment vessel',
        'Containment Vessel, a type of industrial structure',
        'Containment Vessel',
        'A satellite photo of a containment vessel, a type of industrial structure',
        'A satellite photo of a containment vessel, a structure for containing hazardous materials',
        'An overhead view capturing a containment vessel',
        'A high-resolution satellite image of a containment vessel',
        'A structure for containing hazardous materials',
        'A satellite image showing a containment vessel at a nuclear plant',
        'An aerial view of a containment vessel in an industrial complex',
        'A satellite photo of a containment vessel with safety features',
        'A containment vessel captured in a remote sensing image',
        'A satellite image of a containment vessel in a chemical plant',
        'An aerial photograph of a containment vessel in a remote area'
    ],
    'cooling_tower': [
        'A satellite photo of a cooling tower',
        'An aerial image of a cooling tower',
        'Cooling Tower, a type of industrial structure',
        'Cooling Tower',
        'A satellite photo of a cooling tower, a type of industrial structure',
        'A satellite photo of a cooling tower, a structure for cooling water',
        'An overhead view capturing a cooling tower',
        'A high-resolution satellite image of a cooling tower',
        'A structure for cooling water',
        'A satellite image showing a cooling tower at a power plant',
        'An aerial view of a cooling tower with steam rising',
        'A satellite photo of a cooling tower in an industrial area',
        'A cooling tower captured in a remote sensing image',
        'A satellite image of a cooling tower in a chemical plant',
        'An aerial photograph of a cooling tower in a remote area'
    ],
    'crane': [
        'A satellite photo of a crane',
        'An aerial image of a crane',
        'Crane, a type of construction equipment',
        'Crane',
        'A satellite photo of a crane, a type of construction equipment',
        'A satellite photo of a crane, a machine for lifting heavy objects',
        'An overhead view capturing a crane',
        'A high-resolution satellite image of a crane',
        'A machine for lifting heavy objects',
        'A satellite image showing a crane at a construction site',
        'An aerial view of a crane lifting materials',
        'A satellite photo of a crane in an urban area',
        'A crane captured in a remote sensing image',
        'A satellite image of a crane at a port',
        'An aerial photograph of a crane in a rural area'
    ],
    'dock': [
        'A satellite photo of a dock',
        'An aerial image of a dock',
        'Dock, a type of maritime structure',
        'Dock',
        'A satellite photo of a dock, a type of maritime structure',
        'A satellite photo of a dock, a structure for mooring boats',
        'An overhead view capturing a dock',
        'A high-resolution satellite image of a dock',
        'A structure for mooring boats',
        'A satellite image showing a dock at a marina',
        'An aerial view of a dock with boats',
        'A satellite photo of a dock in a harbor',
        'A dock captured in a remote sensing image',
        'A satellite image of a dock with fishing boats',
        'An aerial photograph of a dock in a coastal town'
    ],
    'engineering_vehicle': [
        'A satellite photo of an engineering vehicle',
        'An aerial image of an engineering vehicle',
        'Engineering Vehicle, a type of heavy equipment',
        'Engineering Vehicle',
        'A satellite photo of an engineering vehicle, a type of heavy equipment',
        'A satellite photo of an engineering vehicle, a machine used in construction',
        'An overhead view capturing an engineering vehicle',
        'A high-resolution satellite image of an engineering vehicle',
        'A machine used in construction',
        'A satellite image showing an engineering vehicle at a construction site',
        'An aerial view of an engineering vehicle moving earth',
        'A satellite photo of an engineering vehicle in a quarry',
        'An engineering vehicle captured in a remote sensing image',
        'A satellite image of an engineering vehicle in a mining area',
        'An aerial photograph of an engineering vehicle in a rural area'
    ],
    'flood_dam': [
        'A satellite photo of a flood dam',
        'An aerial image of a flood dam',
        'Flood Dam, a type of dam',
        'Flood Dam',
        'A satellite photo of a flood dam, a type of dam',
        'A satellite photo of a flood dam, a structure for controlling floodwaters',
        'An overhead view capturing a flood dam',
        'A high-resolution satellite image of a flood dam',
        'A structure for controlling floodwaters',
        'A satellite image showing a flood dam in a river',
        'An aerial view of a flood dam with water flowing through',
        'A satellite photo of a flood dam in a rural area',
        'A flood dam captured in a remote sensing image',
        'A satellite image of a flood dam with a reservoir',
        'An aerial photograph of a flood dam in a mountainous region'
    ],
    'foundation_pit': [
        'A satellite photo of a foundation pit',
        'An aerial image of a foundation pit',
        'Foundation Pit, a type of excavation',
        'Foundation Pit',
        'A satellite photo of a foundation pit, a type of excavation',
        'A satellite photo of a foundation pit, a dug-out area for building foundations',
        'An overhead view capturing a foundation pit',
        'A high-resolution satellite image of a foundation pit',
        'A dug-out area for building foundations',
        'A satellite image showing a foundation pit at a construction site',
        'An aerial view of a foundation pit with construction equipment',
        'A satellite photo of a foundation pit in an urban area',
        'A foundation pit captured in a remote sensing image',
        'A satellite image of a foundation pit in a residential area',
        'An aerial photograph of a foundation pit in a rural area'
    ],
    'gas_station': [
        'A satellite photo of a gas station',
        'An aerial image of a gas station',
        'Gas Station, a type of fueling station',
        'Gas Station',
        'A satellite photo of a gas station, a type of fueling station',
        'A satellite photo of a gas station, a facility for refueling vehicles',
        'An overhead view capturing a gas station',
        'A high-resolution satellite image of a gas station',
        'A facility for refueling vehicles',
        'A satellite image showing a gas station on a highway',
        'An aerial view of a gas station with vehicles',
        'A satellite photo of a gas station in an urban area',
        'A gas station captured in a remote sensing image',
        'A satellite image of a gas station in a rural area',
        'An aerial photograph of a gas station in a suburban area'
    ],
    'genset': [
        'A satellite photo of a genset',
        'An aerial image of a genset',
        'Genset, a type of generator',
        'Genset',
        'A satellite photo of a genset, a type of generator',
        'A satellite photo of a genset, a device for generating electricity',
        'An overhead view capturing a genset',
        'A high-resolution satellite image of a genset',
        'A device for generating electricity',
        'A satellite image showing a genset at a construction site',
        'An aerial view of a genset in an industrial area',
        'A satellite photo of a genset in a remote area',
        'A genset captured in a remote sensing image',
        'A satellite image of a genset at a power plant',
        'An aerial photograph of a genset in a rural area'
    ],
    'goods_yard': [
        'A satellite photo of a goods yard',
        'An aerial image of a goods yard',
        'Goods Yard, a type of storage facility',
        'Goods Yard',
        'A satellite photo of a goods yard, a type of storage facility',
        'A satellite photo of a goods yard, an area for storing goods',
        'An overhead view capturing a goods yard',
        'A high-resolution satellite image of a goods yard',
        'An area for storing goods',
        'A satellite image showing a goods yard at a railway station',
        'An aerial view of a goods yard with containers',
        'A satellite photo of a goods yard in an industrial area',
        'A goods yard captured in a remote sensing image',
        'A satellite image of a goods yard with trucks',
        'An aerial photograph of a goods yard in a rural area'
    ],
    'gravity_dam': [
        'A satellite photo of a gravity dam',
        'An aerial image of a gravity dam',
        'Gravity Dam, a type of dam',
        'Gravity Dam',
        'A satellite photo of a gravity dam, a type of dam',
        'A satellite photo of a gravity dam, a structure relying on its weight for stability',
        'An overhead view capturing a gravity dam',
        'A high-resolution satellite image of a gravity dam',
        'A structure relying on its weight for stability',
        'A satellite image showing a gravity dam in a river',
        'An aerial view of a gravity dam with a reservoir',
        'A satellite photo of a gravity dam in a mountainous region',
        'A gravity dam captured in a remote sensing image',
        'A satellite image of a gravity dam with water flowing through',
        'An aerial photograph of a gravity dam in a remote area'
    ],
    'ground_track_field': [
        'A satellite photo of a ground track field',
        'An aerial image of a ground track field',
        'Ground Track Field, a type of sports facility',
        'Ground Track Field',
        'A satellite photo of a ground track field, a type of sports facility',
        'A satellite photo of a ground track field, an area for track and field events',
        'An overhead view capturing a ground track field',
        'A high-resolution satellite image of a ground track field',
        'An area for track and field events',
        'A satellite image showing a ground track field at a school',
        'An aerial view of a ground track field with athletes',
        'A satellite photo of a ground track field in an urban area',
        'A ground track field captured in a remote sensing image',
        'A satellite image of a ground track field surrounded by trees',
        'An aerial photograph of a ground track field in a rural area'
    ],
    'intersection': [
        'A satellite photo of an intersection',
        'An aerial image of an intersection',
        'Intersection, a type of road junction',
        'Intersection',
        'A satellite photo of an intersection, a type of road junction',
        'A satellite photo of an intersection, a point where two or more roads meet',
        'An overhead view capturing an intersection',
        'A high-resolution satellite image of an intersection',
        'A point where two or more roads meet',
        'A satellite image showing an intersection in an urban area',
        'An aerial view of an intersection with traffic lights',
        'A satellite photo of an intersection with heavy traffic',
        'An intersection captured in a remote sensing image',
        'A satellite image of an intersection in a suburban area',
        'An aerial photograph of an intersection in a rural area'
    ],
    'lattice_tower': [
        'A satellite photo of a lattice tower',
        'An aerial image of a lattice tower',
        'Lattice Tower, a type of tower',
        'Lattice Tower',
        'A satellite photo of a lattice tower, a type of tower',
        'A satellite photo of a lattice tower, a structure made of intersecting metal strips',
        'An overhead view capturing a lattice tower',
        'A high-resolution satellite image of a lattice tower',
        'A structure made of intersecting metal strips',
        'A satellite image showing a lattice tower in a rural area',
        'An aerial view of a lattice tower with antennas',
        'A satellite photo of a lattice tower in an industrial area',
        'A lattice tower captured in a remote sensing image',
        'A satellite image of a lattice tower in a mountainous region',
        'An aerial photograph of a lattice tower in a remote area'
    ],
    'roundabout': [
        'A satellite photo of a roundabout',
        'An aerial image of a roundabout',
        'Roundabout, a type of road junction',
        'Roundabout',
        'A satellite photo of a roundabout, a type of road junction',
        'A satellite photo of a roundabout, a circular intersection for traffic flow',
        'An overhead view capturing a roundabout',
        'A high-resolution satellite image of a roundabout',
        'A circular intersection for traffic flow',
        'A satellite image showing a roundabout in an urban area',
        'An aerial view of a roundabout with vehicles',
        'A satellite photo of a roundabout with landscaping',
        'A roundabout captured in a remote sensing image',
        'A satellite image of a roundabout in a suburban area',
        'An aerial photograph of a roundabout in a rural area'
    ],
    'runway': [
        'A satellite photo of a runway',
        'An aerial image of a runway',
        'Runway, a type of airport infrastructure',
        'Runway',
        'A satellite photo of a runway, a type of airport infrastructure',
        'A satellite photo of a runway, a strip for aircraft takeoff and landing',
        'An overhead view capturing a runway',
        'A high-resolution satellite image of a runway',
        'A strip for aircraft takeoff and landing',
        'A satellite image showing a runway at an airport',
        'An aerial view of a runway with an airplane',
        'A satellite photo of a runway in a remote area',
        'A runway captured in a remote sensing image',
        'A satellite image of a runway with markings',
        'An aerial photograph of a runway in an urban area'
    ],
    'ship': [
        'A satellite photo of a ship',
        'An aerial image of a ship',
        'Ship, a type of watercraft',
        'Ship',
        'A satellite photo of a ship, a type of watercraft',
        'A satellite photo of a ship, a large vessel for sea travel',
        'An overhead view capturing a ship',
        'A high-resolution satellite image of a ship',
        'A large vessel for sea travel',
        'A satellite image showing a ship in the ocean',
        'An aerial view of a ship docked at a port',
        'A satellite photo of a ship in a harbor',
        'A ship captured in a remote sensing image',
        'A satellite image of a ship with a wake behind it',
        'An aerial photograph of a ship in a river'
    ],
    'smoke': [
        'A satellite photo of smoke',
        'An aerial image of smoke',
        'Smoke, a type of atmospheric phenomenon',
        'Smoke',
        'A satellite photo of smoke, a type of atmospheric phenomenon',
        'A satellite photo of smoke, a visible suspension of particles in the air',
        'An overhead view capturing smoke',
        'A high-resolution satellite image of smoke',
        'A visible suspension of particles in the air',
        'A satellite image showing smoke rising from a fire',
        'An aerial view of smoke over a forest',
        'A satellite photo of smoke from an industrial area',
        'Smoke captured in a remote sensing image',
        'A satellite image of smoke over a city',
        'An aerial photograph of smoke in a rural area'
    ],
    'soccer_ball_field': [
        'A satellite photo of a soccer ball field',
        'An aerial image of a soccer ball field',
        'Soccer Ball Field, a type of sports facility',
        'Soccer Ball Field',
        'A satellite photo of a soccer ball field, a type of sports facility',
        'A satellite photo of a soccer ball field, an area for playing soccer',
        'An overhead view capturing a soccer ball field',
        'A high-resolution satellite image of a soccer ball field',
        'An area for playing soccer',
        'A satellite image showing a soccer ball field in a park',
        'An aerial view of a soccer ball field with players',
        'A satellite photo of a soccer ball field in an urban area',
        'A soccer ball field captured in a remote sensing image',
        'A satellite image of a soccer ball field surrounded by trees',
        'An aerial photograph of a soccer ball field at a school'
    ],
    'storehouse': [
        'A satellite photo of a storehouse',
        'An aerial image of a storehouse',
        'Storehouse, a type of storage facility',
        'Storehouse',
        'A satellite photo of a storehouse, a type of storage facility',
        'A satellite photo of a storehouse, a building for storing goods',
        'An overhead view capturing a storehouse',
        'A high-resolution satellite image of a storehouse',
        'A building for storing goods',
        'A satellite image showing a storehouse in an industrial area',
        'An aerial view of a storehouse with trucks',
        'A satellite photo of a storehouse in a rural area',
        'A storehouse captured in a remote sensing image',
        'A satellite image of a storehouse with goods',
        'An aerial photograph of a storehouse in a suburban area'
    ],
    'substation': [
        'A satellite photo of a substation',
        'An aerial image of a substation',
        'Substation, a type of electrical facility',
        'Substation',
        'A satellite photo of a substation, a type of electrical facility',
        'A satellite photo of a substation, a facility for transforming voltage',
        'An overhead view capturing a substation',
        'A high-resolution satellite image of a substation',
        'A facility for transforming voltage',
        'A satellite image showing a substation in an urban area',
        'An aerial view of a substation with power lines',
        'A satellite photo of a substation in a rural area',
        'A substation captured in a remote sensing image',
        'A satellite image of a substation with transformers',
        'An aerial photograph of a substation in a suburban area'
    ],
    'tank': [
        'A satellite photo of a tank',
        'An aerial image of a tank',
        'Tank, a type of military vehicle',
        'Tank',
        'A satellite photo of a tank, a type of military vehicle',
        'A satellite photo of a tank, an armored fighting vehicle',
        'An overhead view capturing a tank',
        'A high-resolution satellite image of a tank',
        'An armored fighting vehicle',
        'A satellite image showing a tank in a military base',
        'An aerial view of a tank in a training area',
        'A satellite photo of a tank in a desert',
        'A tank captured in a remote sensing image',
        'A satellite image of a tank in a forest',
        'An aerial photograph of a tank in a rural area'
    ],
    'taxiway': [
        'A satellite photo of a taxiway',
        'An aerial image of a taxiway',
        'Taxiway, a type of airport infrastructure',
        'Taxiway',
        'A satellite photo of a taxiway, a type of airport infrastructure',
        'A satellite photo of a taxiway, a path for aircraft to move on the ground',
        'An overhead view capturing a taxiway',
        'A high-resolution satellite image of a taxiway',
        'A path for aircraft to move on the ground',
        'A satellite image showing a taxiway at an airport',
        'An aerial view of a taxiway with an airplane',
        'A satellite photo of a taxiway in a remote area',
        'A taxiway captured in a remote sensing image',
        'A satellite image of a taxiway with markings',
        'An aerial photograph of a taxiway in an urban area'
    ],
    'tennis_court': [
        'A satellite photo of a tennis court',
        'An aerial image of a tennis court',
        'Tennis Court, a type of sports facility',
        'Tennis Court',
        'A satellite photo of a tennis court, a type of sports facility',
        'A satellite photo of a tennis court, an area for playing tennis',
        'An overhead view capturing a tennis court',
        'A high-resolution satellite image of a tennis court',
        'An area for playing tennis',
        'A satellite image showing a tennis court in a park',
        'An aerial view of a tennis court with players',
        'A satellite photo of a tennis court in an urban area',
        'A tennis court captured in a remote sensing image',
        'A satellite image of a tennis court surrounded by trees',
        'An aerial photograph of a tennis court at a school'
    ],
    'terminal': [
        'A satellite photo of a terminal',
        'An aerial image of a terminal',
        'Terminal, a type of airport infrastructure',
        'Terminal',
        'A satellite photo of a terminal, a type of airport infrastructure',
        'A satellite photo of a terminal, a building for passenger services',
        'An overhead view capturing a terminal',
        'A high-resolution satellite image of a terminal',
        'A building for passenger services',
        'A satellite image showing a terminal at an airport',
        'An aerial view of a terminal with airplanes',
        'A satellite photo of a terminal in an urban area',
        'A terminal captured in a remote sensing image',
        'A satellite image of a terminal with passengers',
        'An aerial photograph of a terminal in a rural area'
    ],
    'toll_gate': [
        'A satellite photo of a toll gate',
        'An aerial image of a toll gate',
        'Toll Gate, a type of road infrastructure',
        'Toll Gate',
        'A satellite photo of a toll gate, a type of road infrastructure',
        'A satellite photo of a toll gate, a point for collecting tolls',
        'An overhead view capturing a toll gate',
        'A high-resolution satellite image of a toll gate',
        'A point for collecting tolls',
        'A satellite image showing a toll gate on a highway',
        'An aerial view of a toll gate with vehicles',
        'A satellite photo of a toll gate in an urban area',
        'A toll gate captured in a remote sensing image',
        'A satellite image of a toll gate in a rural area',
        'An aerial photograph of a toll gate in a suburban area'
    ],
    'tower_crane': [
        'A satellite photo of a tower crane',
        'An aerial image of a tower crane',
        'Tower Crane, a type of construction equipment',
        'Tower Crane',
        'A satellite photo of a tower crane, a type of construction equipment',
        'A satellite photo of a tower crane, a tall crane for lifting heavy materials',
        'An overhead view capturing a tower crane',
        'A high-resolution satellite image of a tower crane',
        'A tall crane for lifting heavy materials',
        'A satellite image showing a tower crane at a construction site',
        'An aerial view of a tower crane lifting materials',
        'A satellite photo of a tower crane in an urban area',
        'A tower crane captured in a remote sensing image',
        'A satellite image of a tower crane at a port',
        'An aerial photograph of a tower crane in a rural area'
    ],
    'truck': [
        'A satellite photo of a truck',
        'An aerial image of a truck',
        'Truck, a type of vehicle',
        'Truck',
        'A satellite photo of a truck, a type of vehicle',
        'A satellite photo of a truck, a large vehicle for transporting goods',
        'An overhead view capturing a truck',
        'A high-resolution satellite image of a truck',
        'A large vehicle for transporting goods',
        'A satellite image showing a truck on a highway',
        'An aerial view of a truck in a warehouse',
        'A satellite photo of a truck in an industrial area',
        'A truck captured in a remote sensing image',
        'A satellite image of a truck in a rural area',
        'An aerial photograph of a truck in a suburban area'
    ],
    'truck_parking': [
        'A satellite photo of a truck parking',
        'An aerial image of a truck parking',
        'Truck Parking, a type of parking facility',
        'Truck Parking',
        'A satellite photo of a truck parking, a type of parking facility',
        'A satellite photo of a truck parking, an area designated for parking trucks',
        'An overhead view capturing a truck parking',
        'A high-resolution satellite image of a truck parking',
        'An area designated for parking trucks',
        'A satellite image showing a truck parking lot',
        'An aerial view of a truck parking with multiple trucks',
        'A satellite photo of a truck parking in an industrial area',
        'A truck parking captured in a remote sensing image',
        'A satellite image of a truck parking at a logistics center',
        'An aerial photograph of a truck parking in a rural area'
    ],
    'unfinished_building': [
        'A satellite photo of an unfinished building',
        'An aerial image of an unfinished building',
        'Unfinished Building, a type of construction site',
        'Unfinished Building',
        'A satellite photo of an unfinished building, a type of construction site',
        'A satellite photo of an unfinished building, a structure under construction',
        'An overhead view capturing an unfinished building',
        'A high-resolution satellite image of an unfinished building',
        'A structure under construction',
        'A satellite image showing an unfinished building in an urban area',
        'An aerial view of an unfinished building with construction equipment',
        'A satellite photo of an unfinished building in a suburban area',
        'An unfinished building captured in a remote sensing image',
        'A satellite image of an unfinished building in a rural area',
        'An aerial photograph of an unfinished building in a remote area'
    ],
    'vapor': [
        'A satellite photo of vapor',
        'An aerial image of vapor',
        'Vapor, a type of atmospheric phenomenon',
        'Vapor',
        'A satellite photo of vapor, a type of atmospheric phenomenon',
        'A satellite photo of vapor, a visible suspension of water droplets in the air',
        'An overhead view capturing vapor',
        'A high-resolution satellite image of vapor',
        'A visible suspension of water droplets in the air',
        'A satellite image showing vapor rising from a cooling tower',
        'An aerial view of vapor over a lake',
        'A satellite photo of vapor from an industrial area',
        'Vapor captured in a remote sensing image',
        'A satellite image of vapor over a city',
        'An aerial photograph of vapor in a rural area'
    ],
    'wind_mill': [
        'A satellite photo of a wind mill',
        'An aerial image of a wind mill',
        'Wind Mill, a type of renewable energy structure',
        'Wind Mill',
        'A satellite photo of a wind mill, a type of renewable energy structure',
        'A satellite photo of a wind mill, a structure for generating wind power',
        'An overhead view capturing a wind mill',
        'A high-resolution satellite image of a wind mill',
        'A structure for generating wind power',
        'A satellite image showing a wind mill in a rural area',
        'An aerial view of a wind mill with rotating blades',
        'A satellite photo of a wind mill in a windy area',
        'A wind mill captured in a remote sensing image',
        'A satellite image of a wind mill in a mountainous region',
        'An aerial photograph of a wind mill in a remote area'
    ]
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







