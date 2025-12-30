import json
from commonlibs.common_tools import jsonsave
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

    def add_ann(self, area, iscrowd, image_id, category_id, bbox, id, segmentation=[], others=None):
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
        self.annotations.append(ann)

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





