import os
import matplotlib as mpl
import cv2
mpl.use('Qt5Agg')
from xml.etree import ElementTree as et
import json
from commonlibs.common_tools import *
from tqdm import tqdm


def todota(ann_folder, dst_ann_folder):
    file_list = os.listdir(ann_folder)
    all_names = set()
    # 提取xml文件，获得bbox、标签
    for ann_file in tqdm(file_list):
        xml_file = ann_folder + '/' + ann_file
        # 开始提取
        tree = et.parse(xml_file)
        root = tree.getroot()

        objects = root.find('objects').findall('object')
        # 获得bbox和label
        for obj in objects:
            try:
                points = [[float(p_coord) for p_coord in p.text.split(',')]
                          for p in obj.find('points').findall('point')]
                name = obj.find('possibleresult').find('name').text
                name = name.replace(' ', '_')
                all_names.add(name)
            except ValueError:
                print(('Wrong Value in file %s' % (str(xml_file))))
    print(f'Total {len(all_names)} names')
    print(sorted(list(all_names)))

data_root = '/data/space2/huangziyue/FAIR1M2.0/'
#

todota(data_root + 'labelXml',
       data_root + 'labelXml')