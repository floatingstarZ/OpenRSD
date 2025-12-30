import os
import json
import cv2
from commonlibs.transform_tools.coco_annotation_template import COCOTmp
fpath = './annotation_val.odgt'
def load_file(fpath):#fpath是具体的文件 ，作用：#str to list
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records

def crowdhuman2coco(odgt_path, img_folder, json_path):
    records = load_file(odgt_path)
    cat_name_list = ['person']  #, 'mask']
    coco_res = COCOTmp()
    coco_res.fill_info('crowd human annotations')

    id2name, name2id = coco_res.auto_generate_categories(cat_name_list)
    bbox_id = 0
    img_count = 0
    for recode in records:
        img_id = recode['ID']
        gtboxes = recode['gtboxes']
        file_name = img_id + '.jpg'
        img_file = img_folder + '/' + file_name
        img = cv2.imread(img_file)
        coco_res.add_image(file_name, img.shape[0], img.shape[1], img.shape[2], img_id)
        for ann in gtboxes:
            x, y, h, w = ann['fbox']  # full box only
            if ann['tag'] == 'mask':
                #print('Pass mask')
                continue
            coco_res.add_ann(h*w, 0, img_id, name2id[ann['tag']], ann['fbox'], bbox_id)
            bbox_id += 1
        img_count += 1
        print('%d / %d' % (img_count, len(records)))

    coco_res.save_ann(json_path)






# crowdhuman2coco('D:/DataBackup/CrowdHuman/annotation_val.odgt',
#                 'D:/DataBackup/CrowdHuman/Images',
#                 'D:/DataBackup/CrowdHuman/annotation_coco_val.json')

crowdhuman2coco('D:/DataBackup/CrowdHuman/annotation_train.odgt',
                'D:/DataBackup/CrowdHuman/Images',
                'D:/DataBackup/CrowdHuman/annotation_coco_train.json')