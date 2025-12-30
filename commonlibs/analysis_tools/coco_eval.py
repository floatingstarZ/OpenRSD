import numpy as npimg_i
from commonlibs.analysis_tools.cocoeval_tools.coco_cls import COCO
from commonlibs.analysis_tools.cocoeval_tools.cocoeval_cls import COCOeval
from commonlibs.analysis_tools.cocoeval_tools.other_tools import split_cocos, split_resfiles

def eval_per_img(result_file, gt_ann_file, show_msg=False, img_ids=None):
    """
    evaluate mAp, ... for every image
    :param result_file: result file
    :param gt_ann_file:  gt coco annotation
    :return: {img_id: eval_results}
    """
    # split coco and result for every image
    cocos = split_cocos(gt_ann_file, img_ids=img_ids)
    img_ids = cocos.keys()
    resfiles = split_resfiles(result_file, img_ids)
    eval_results = {}
    eval_results_str = {}
    # evaluate and get result
    for img_id, coco in cocos.items():
        result_file = resfiles[img_id]
        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        cocoEval = COCOeval(coco, coco_dets, 'bbox', show_msg=show_msg)
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        eval_results[img_id] = cocoEval.stats.tolist()
        eval_results_str[img_id] = cocoEval.stats_str

    return eval_results

if __name__ == '__main__':
    res_file = '../../test_data/RESULT_retinanet_x101.bbox.json'
    ann_file = '../../test_data/instances_val2017.json'
    er = eval_per_img(res_file, ann_file, img_ids=[37777, 1000])
    print(er)


