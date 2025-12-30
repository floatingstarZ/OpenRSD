from commonlibs.analysis_tools.cocoeval_tools.coco_cls import COCO
from commonlibs.common_tools import *
import os

def split_cocos(file_path, img_ids=None):
    """
    build coco class for every img id in img_ids
    one img id, one coco class
    :param file_path: annotation file path
    :param img_ids: image ids
    :return: cocos: {img_id : COCO class}
    """
    dataset = jsonload(file_path, msg=False)
    if not img_ids:
        img_ids = [img_info['id'] for img_info in dataset['images']]
    cocos = {}
    datasets = {img_id:
                    dict(annotations=[],
                         categories=dataset['categories'],
                         images=[],
                         info=dataset['info'],
                         licenses=dataset['licenses']
                         )
                for img_id in img_ids}
    for ann in dataset['annotations']:
        img_id = ann['image_id']
        if img_id in datasets.keys():
            datasets[img_id]['annotations'].append(ann)
    for img_info in dataset['images']:
        img_id = img_info['id']
        if img_id in datasets.keys():
            datasets[img_id]['images'].append(img_info)

    for img_id, d in datasets.items():
        c = COCO(d)
        cocos[img_id] = c
    return cocos

def split_resfiles(res_file_path, img_ids=None):
    if img_ids == None:
        return {}
    results = jsonload(res_file_path, msg=False)
    resfiles = {img_id: [] for img_id in img_ids}
    for r in results:
        img_id = r['image_id']
        if img_id in img_ids:
            resfiles[img_id].append(r)
    return resfiles


if __name__ == '__main__':
    ann_file_path = '../../../test_data/instances_val2017.json'
    res_file = '../../../test_data/RESULT_retinanet_x101.bbox.json'

    cocos = split_cocos(ann_file_path, img_ids=[37777])
    resfiles = split_resfiles(res_file, img_ids=[37777])
    a = 0


