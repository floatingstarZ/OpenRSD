import json


def transform_to_mmdet_results(img_dets):
    results = []
    for img_id, infos in img_dets.items():
        bboxes = infos['bboxes']
        cat_ids = infos['cat_ids']
        scores = infos['scores']

        for b, cat_id, s in zip(bboxes, cat_ids, scores):
            result = dict(
                image_id=img_id,
                bbox=b,
                score=s,
                other_score=[],
                category_id=cat_id
            )
            results.append(result)
    return results


def mmdet_results_transform(dets):
    img_dets = {}
    for det in dets:
        img_id = det['image_id']
        if img_id not in img_dets.keys():
            img_dets[img_id] = dict(
                bboxes=[],
                cat_ids=[],
                scores=[]
            )
        img_dets[img_id]['bboxes'].append(det['bbox'])
        img_dets[img_id]['cat_ids'].append(det['category_id'])
        img_dets[img_id]['scores'].append(det['score'])
    return img_dets
