import json
from ctlib.os import jsonsave
class COCOTmp:
    def __init__(self, cocotmp=None):
        self.info = dict(
            description=''
        )
        self.images = []
        self.img_ids = []
        self.img_tmp = dict(
            file_name='',
            height=0,
            width=0,
            channel=0,
            id=0,
        )
        self.licenses = []
        self.annotations = []
        self.max_ann_id = -1
        self.max_img_id = -1
        self.ann_ids = []
        self.ann_tmp = dict(
            segmentation=[],
            area=0,
            iscrowd=0,
            image_id=0,
            bbox=[],
            category_id=0,
            id=0
        )
        self.categories = []
        self.cat_tmp = dict(
            supercategory='',
            id=0,
            name=''
        )
        if cocotmp != None:
            self.info = cocotmp.info
            self.images = cocotmp.images
            self.img_ids = cocotmp.img_ids
            self.licenses = cocotmp.licenses
            self.annotations = cocotmp.annotations
            self.categories = cocotmp.annotations

            self.ann_ids = cocotmp.ann_ids
            if len(self.ann_ids):
                self.max_ann_id = max(self.ann_ids)
            if len(self.img_ids):
                self.max_img_id = max(self.img_ids)


    def fill_info(self, description):
        self.info['description'] = description


    def add_image(self, file_name, height, width, channel, id, others=None):
        """
        :param id: img id
        channel: new added parameters
        :return:
        """
        img_info = dict(
            file_name=file_name,
            height=height,
            width=width,
            channel=channel,
            id=id,
            others=others
        )
        assert id not in self.img_ids
        self.img_ids.append(id)
        self.images.append(img_info)
        self.max_img_id = max(self.max_img_id, id)


    def add_ann(self, area, iscrowd, image_id, category_id, bbox, id=None, segmentation=[], others=None):
        ann = dict(
            segmentation=segmentation,
            area=area,
            iscrowd=iscrowd,
            image_id=image_id,
            bbox=bbox,
            category_id=category_id,
            id=id,
            others=others
        )
        if id:
            assert id not in self.ann_ids
            ann['id'] = id
        else:
            ann['id'] = self.max_ann_id + 1
        self.annotations.append(ann)
        self.ann_ids.append(ann['id'])
        self.max_ann_id = max(self.max_ann_id, ann['id'])

    def fill_categories(self, categories):
        self.categories = categories

    def save_ann(self, save_path):
        self.ann = dict(
            info=self.info,
            images=self.images,
            licenses=self.licenses,
            annotations=self.annotations,
            categories=self.categories
        )
        jsonsave(self.ann, save_path)

    def check_ann(self):
        pass

    def auto_generate_categories(self, cat_names):
        """
        auto_generate_categories and ids, id from 1 to C + 1
        :param cat_names: list
        :return:
        """
        self.categories = []
        for id, name in enumerate(cat_names):
            cat_ann = dict(
                supercategory='',
                id=id+1,
                name=name
            )
            self.categories.append(cat_ann)
        id2name={(id+1): name for id, name in enumerate(cat_names)}
        name2id={name:(id+1) for id, name in enumerate(cat_names)}
        return id2name, name2id

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




