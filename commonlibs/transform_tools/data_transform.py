from commonlibs.common_tools import *

def coco_transform(coco_ann):
    """

    :param coco_ann: coco style annotations 
    :return: 
    index : {img_id: bboxes(bbox, cat_id), img_path, ...}，简化版的信息
    id2name: {cat id: name
    """
    img_infos = coco_ann['images']
    img_infos = {info['id']: info for info in img_infos}
    for info in img_infos.values():
        info['anns'] = []
    id2name = {ann['id']: ann['name']
               for ann in coco_ann['categories']
               }

    for ann in coco_ann['annotations']:
        img_id = ann['image_id']
        if img_id not in img_infos.keys():
            print('Pass %s' % str(img_id))
            continue
        img_infos[img_id]['anns'].append(ann)
    return img_infos, id2name

def COCO2index(coco_ann):
    """

    :param coco_ann: coco style annotations 
    :return: 
    index : {img_id: bboxes(bbox, cat_id), img_path, ...}，简化版的信息
    id2name: {cat id: name
    """
    img_infos = coco_ann['images']
    img_infos = {
        info['id']:
            {
                'file_name': info['file_name'],
                'height': info['height'],
                'width': info['width'],
                'bboxes': []}
        for info in img_infos}
    id2name = {ann['id']: ann['name']
               for ann in coco_ann['categories']
               }

    for ann in coco_ann['annotations']:
        img_id = ann['image_id']
        if img_id not in img_infos.keys():
            print('Pass %s' % str(img_id))
            continue
        (x1, y1, h, w) = ann['bbox']
        bbox_info = [x1, y1, h, w, ann['category_id']]
        img_infos[img_id]['bboxes'].append(bbox_info)
    return img_infos, id2name


if __name__ == '__main__':
    test_ann_file = 'D:/DataBackup/COCO/annotations/instances_val2017.json'
    ann = jsonload(test_ann_file)
    index = COCO2index(ann)





