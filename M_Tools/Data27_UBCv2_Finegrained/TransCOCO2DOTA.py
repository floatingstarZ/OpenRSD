from commonlibs.transform_tools.data_transform import coco_transform
from commonlibs.common_tools import *
import os
import shutil
import cv2
from xml.etree import ElementTree as et
import json
from commonlibs.common_tools import *
from commonlibs.transform_tools.data_transform import coco_transform

from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch

def cv2_mask2rbbox(bi_mask):
    """
    cv2的方法获得旋转框角度
    :param bi_mask:
    :return:
    """
    contours, hierarchy = cv2.findContours(bi_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    # 点最多的那个，作为contours
    if len(contours) == 0 or len(contours[0]) == 0:
        return 0, 0, 1, 1, 0
    max_contour = max(contours, key=len)
    [(xc, yc), (w, h), theta] = cv2.minAreaRect(max_contour)
    theta = theta * 2 * np.pi / 360
    return np.array([xc, yc, w, h, theta])

def rbbox2poly(dboxes):
    """cv2.boxPoints
    :param dboxes: (x_ctr, y_ctr, w, h, angle)
        (numboxes, 5)
    :return: quadranlges:
        (numboxes, 8)
    """
    if type(dboxes) == np.ndarray:
        cs = np.cos(dboxes[:, 4])
        ss = np.sin(dboxes[:, 4])
    elif type(dboxes) == torch.Tensor:
        cs = torch.cos(dboxes[:, 4])
        ss = torch.sin(dboxes[:, 4])
    else:
        raise Exception('Wrong type of dboxes in rbbox to poly')
    w = dboxes[:, 2]# - 1
    h = dboxes[:, 3]# - 1

    ## change the order to be the initial definition

    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]

    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    x1 = x1.reshape(-1, 1)
    y1 = y1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)
    x3 = x3.reshape(-1, 1)
    y3 = y3.reshape(-1, 1)
    x4 = x4.reshape(-1, 1)
    y4 = y4.reshape(-1, 1)
    if type(dboxes) == np.ndarray:
        polys = np.concatenate((x1, y1, x2, y2,
                                x3, y3, x4, y4), axis=1)
    elif type(dboxes) == torch.Tensor:
        polys = torch.cat([x1, y1, x2, y2,
                           x3, y3, x4, y4], dim=1)

    return polys



def coco_2_dota(img_folder, ann_file_path, result_folder):
    img_infos, id2name = coco_transform(jsonload(ann_file_path))

    label_folder = result_folder + '/labelTxt'
    image_folder = result_folder + '/images'
    mkdir(result_folder)
    mkdir(label_folder)
    mkdir(image_folder)


    for img_count, img_info in enumerate(img_infos.values()):
        img_file_name = img_info['file_name']
        img_path = img_folder + '/' + img_file_name
        new_img_path = image_folder + '/' + \
                       os.path.splitext(img_file_name)[0] + '.png'
        if len(img_info['anns']) == 0:
            continue
        img = cv2.imread(img_path)
        cv2.imwrite(new_img_path, img)

        # shutil.copyfile(img_path, image_folder + '/' + img_file_name)

        ann_file_name = os.path.splitext(img_file_name)[0] + '.txt'
        ann_file_path = label_folder + '/' + ann_file_name
        with open(ann_file_path, 'wt+') as f:
            for ann in img_info['anns']:
                if type(ann['segmentation']) != list:
                    continue
                try:
                    poly = np.array(ann['segmentation'])
                except:
                    print(ann['segmentation'])
                    continue
                # print(poly)
                mask_H = int(np.max(poly) + 10)
                mask_W = mask_H
                mask_img = np.zeros([mask_W, mask_H], dtype=np.uint8)
                cv2.fillConvexPoly(mask_img,
                                   np.array(poly, dtype=np.int32).reshape(-1, 2),
                                   1)
                rbbox = cv2_mask2rbbox(mask_img)
                rpoly = rbbox2poly(np.array(rbbox).reshape(-1, 5))
                rpoly = rpoly.flatten()
                s = ''
                for coord in rpoly:
                    s += '%.1f ' % coord
                s += 'building 0\n'
                f.write(s)
            print('Save ann: %s' % ann_file_path)

        # if img_count > 100:
        #     break
        print('Done %d / %d' % (img_count+1, len(img_infos)))


data_folder = '/data/space2/huangziyue/OOD_RSOD_Datasets/Data27_UBCv2_Finegrained'
ann_folder = data_folder + '/annotations'
out_root = '/data/space2/huangziyue/OOD_RSOD_Datasets/Data27_UBCv2_Finegrained_DOTA'
mkdir(out_root)

ann_file = ann_folder + '/roof_fine_12_train.json'
img_folder = data_folder + '/images/train'
coco_2_dota(img_folder, ann_file, out_root + '/train')
#
ann_file = ann_folder + '/roof_fine_12_val.json'
img_folder = data_folder + '/images/val'
coco_2_dota(img_folder, ann_file, out_root + '/val')
