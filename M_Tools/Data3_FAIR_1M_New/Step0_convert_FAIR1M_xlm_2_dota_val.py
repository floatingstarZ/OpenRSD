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
    mkdir(dst_ann_folder)
    all_names = set()
    # 提取xml文件，获得bbox、标签
    for img_id, img_file in enumerate(file_list):
        (img_name, ext) = os.path.splitext(img_file)
        xml_file = ann_folder + '/' + img_name + '.xml'
        dst_ann_file = dst_ann_folder + '/' + img_name + '.txt'
        # 开始提取
        tree = et.parse(xml_file)
        root = tree.getroot()
        print(img_name)

        objects = root.find('objects').findall('object')
        # 获得bbox和label
        with open(dst_ann_file, 'wt+') as f:
            for obj in objects:
                try:
                    points = [[float(p_coord) for p_coord in p.text.split(',')]
                              for p in obj.find('points').findall('point')]
                    # drop the last point(same as the first point)
                    points = points[:-1]
                    name = obj.find('possibleresult').find('name').text
                    name = name.replace(' ', '_').lower()
                    all_names.add(name)
                except ValueError:
                    print(('Wrong Value in file %s' % (str(img_id))))
                for (p1, p2) in points:
                    f.write('%.1f %.1f ' % (p1, p2))
                # difficulty set to 0
                f.write(name + ' 0 \n')
    print(f'Total {len(all_names)} names')
    print(sorted(list(all_names)))

data_root = '/data/space2/huangziyue/FAIR1M2.0/validation'
#

todota(data_root + '/labelXml',
       data_root + '/labelTxt')


