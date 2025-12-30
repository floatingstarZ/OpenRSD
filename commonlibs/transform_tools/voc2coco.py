import os
import matplotlib as mpl
import cv2
mpl.use('Qt5Agg')
from xml.etree import ElementTree as et
import json
from commonlibs.transform_tools.coco_annotation_template import COCOTmp
from commonlibs.common_tools import *


def make_voc_coco(data_folder, ann_folder, seg_file, out_ann,
                  src_cocotmp=None, savejson=True):
    with open(seg_file, 'r') as f:
        ids = f.readlines()
    ids = [id.strip('\n') for id in ids]
    file_list = os.listdir(data_folder)
    # cat_name_list = ['person',
    #         'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
    #         'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    #         'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    # 以下是mmdet的版本

    cat_name_list = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    coco_res = COCOTmp(src_cocotmp)
    id2name, name2id = coco_res.auto_generate_categories(cat_name_list)
    coco_res.fill_info('VOC 2012 2007 annotations')
    bbox_id = 0
    n_img = 0
    # 提取xml文件，获得bbox、标签
    for img_id, img_file in enumerate(file_list):
        (img_name, extension) = os.path.splitext(img_file)
        xml_file = ann_folder + '/' + img_name + '.xml'
        # 开始提取
        tree = et.parse(xml_file)
        root = tree.getroot()
        # 获得文件路径
        file_name = root.find('filename')
        file_name = file_name.text
        img = cv2.imread(data_folder + '/' + file_name)
        file_path = data_folder + '/' + file_name
        img_id = os.path.splitext(file_name)[0]
        # print(img_id)
        if img_id not in ids:
            continue
        print('%d / %d' % (n_img, len(ids)))
        n_img += 1
        # file_name只是file_name而已
        coco_res.add_image(file_name, img.shape[0], img.shape[1], img.shape[2], img_id)
        objects = root.findall('object')
        # 获得bbox和label
        for obj in objects:
            bbox = obj.find('bndbox')
            try:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                difficult = int(obj.find('difficult').text)
                if difficult:
                    print('Diff')
                h = xmax - xmin
                w = ymax - ymin
                area = h * w
                name = obj.find('name').text
                others = dict(difficult=difficult)
                coco_res.add_ann(area, 0, img_id,
                                 name2id[name], [xmin-1, ymin-1, h, w], bbox_id, others=others)
                bbox_id += 1
                if name not in name2id.keys():
                    raise Exception('Wrong Name: %s in file %s' % (name, file_path))
            except ValueError:
                print(('Wrong Value in file %s' % (file_path)))
    if savejson:
        coco_res.save_ann(out_ann)
    return coco_res



data_root = 'D:/DataBackup/VOC'
coco_ann_folder = data_root + '/coco_annotations'
mkdir(coco_ann_folder)

voc2012_root = data_root + '/VOC2012_trainval'
data_folder = voc2012_root + '/JPEGImages'
ann_folder = voc2012_root + '/Annotations'
seg_folder = voc2012_root + '/ImageSets/Main'
seg_file = seg_folder + '/' + 'trainval.txt'
ann_file = None
ann_2012 = make_voc_coco(data_folder, ann_folder, seg_file, ann_file,
                        savejson=False)

voc2007_root = data_root + '/VOC2007_trainval'
data_folder = voc2007_root + '/JPEGImages'
ann_folder = voc2007_root + '/Annotations'
seg_folder = voc2007_root + '/ImageSets/Main'
seg_file = seg_folder + '/' + 'trainval.txt'
ann_file = coco_ann_folder + '/trainval_coco_ann.json'
make_voc_coco(data_folder, ann_folder, seg_file, ann_file,
                         src_cocotmp=ann_2012, savejson=True)

voc2007_root = data_root + '/VOC2007_test'
data_folder = voc2007_root + '/JPEGImages'
ann_folder = voc2007_root + '/Annotations'
seg_folder = voc2007_root + '/ImageSets/Main'
seg_file = seg_folder + '/' + 'test.txt'
ann_file = coco_ann_folder + '/test_coco_ann.json'
make_voc_coco(data_folder, ann_folder, seg_file, ann_file, savejson=True)






