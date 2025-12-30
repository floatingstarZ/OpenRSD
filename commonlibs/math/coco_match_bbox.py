import torch
from commonlibs.math_tools.IOU import IOU

def _match_DT_GT(dts, dt_scores, dt_labels,
                gts, gt_labels,
                cat_label, threshold):
    """
    
    :param dts: M x 4, det bboxes, left top right down
    :param dt_labels: M, det bbox labels
    :param dt_scores: M, det bbox scores
    
    :param gts: N x 4, ground truth
    :param gt_labels: N, ground truth labels
    :param cat_label: int, category labels
    :param threshold: float, iou threshold
    :return: 
    matched dt bboxes,  matched dt scores
    matched gt bboxes
    matched dt index: int number
    matched gt index: int number
    """
    # label == category
    # .._indices: index of .. in origin .. order
    gt_indices = torch.arange(0, len(gt_labels))[gt_labels == cat_label].long()
    gt = gts[gt_indices]

    dt_indices = torch.arange(0, len(dt_labels))[dt_labels == cat_label].long()
    dt = dts[dt_indices]
    scores = dt_scores[dt_indices]

    scores, ids = torch.sort(scores, descending=True)
    dt_indices = dt_indices[ids]
    dt = dt[ids]

    # gtm: gt matched dts' id(in input order)
    gtm = -gt_labels.new_ones(len(gt)).long()
    dtm = -gt_labels.new_ones(len(dt)).long()

    if len(gt) == 0 or len(dt) == 0:
        e = gt.new_tensor([]).long()
        return e, e, e, e, e, e, e

    ious = IOU(dt, gt)
    for dind, d in enumerate(dt):
        iou = min([threshold, 1 - 1e-10])
        m = -1
        # match d with gts
        for gind, g in enumerate(gt):
            # if g already matched
            if gtm[gind] != -1:
                continue
            # continue to next gt unless better match made
            if ious[dind, gind] < iou:
                continue
            # save matched gind and iou
            iou = ious[dind, gind]
            m = gind
        # if d doesn't matched any gts
        if m == -1:
            continue
        # save matched ids(in input order)
        gtm[m] = dt_indices[dind]
        dtm[dind] = gt_indices[m]

    matched_dt = dt[dtm > -1]
    matched_dt_score = scores[dtm > -1]
    matched_dt_ids = dt_indices[dtm > -1]
    dtm = dtm[dtm > -1]

    matched_gt = gt[gtm > -1]
    matched_gt_ids = gt_indices[gtm > -1]
    gtm = gtm[gtm>-1]

    return matched_dt, matched_dt_score, matched_dt_ids, \
           matched_gt, matched_gt_ids, dtm, gtm

def match_dt_with_gt(dts, dt_scores, dt_labels,
                     gts, gt_labels,
                     cat_label_list, threshold):
    """

    :param dts: M x 4, det bboxes, left top right down
    :param dt_labels: M, det bbox labels
    :param dt_scores: M, det bbox scores

    :param gts: N x 4, ground truth
    :param gt_labels: N, ground truth labels
    :param cat_label_list: [int], possible category label list
    :param threshold: float, iou threshold
    :return: 
    gtmatched: matched dt idx in dt_labels order, int, -1 -> unmatched 
    matched gt index: bool
    """
    gtmatched = -gts.new_ones(len(gt_labels)).long()
    dtmatched = -gts.new_ones(len(dt_labels)).long()
    if len(gt_labels) == 0 or len(gt_labels) == 0:
        e = gt_labels.new_tensor([]).long()
        return e, e

    for cat_label in cat_label_list:
        matched_dt, matched_dt_score, matched_dt_ids, \
        matched_gt, matched_gt_ids, dtm, gtm = \
            _match_DT_GT(dts, dt_scores, dt_labels,
                     gts, gt_labels, cat_label, threshold)
        gtmatched[matched_gt_ids] = gtm
        dtmatched[matched_dt_ids] = dtm
    return gtmatched, dtmatched

if __name__ == '__main__':
    gts = torch.Tensor([[0, 0, 5, 5],
                        [20, 20, 25, 25]])
    gt_labels = torch.Tensor([0, 1])
    dts = torch.Tensor([[0, 0, 3, 3],
                        [20, 20, 23, 23],
                        [0, 0, 5, 3],
                        [20, 20, 25, 23],
                        [20, 20, 25, 25]])
    dt_score = torch.Tensor([1, 0.5, 1.2, 0.7, 10])
    dt_labels = torch.Tensor([0, 1, 0, 1, 0])
    cat_label_list = torch.Tensor([0, 1])

    threshold = 0.1

    print(match_dt_with_gt(dts, dt_score,dt_labels,
                           gts, gt_labels,
                           cat_label_list, threshold))








