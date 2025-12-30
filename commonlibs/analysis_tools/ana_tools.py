import torch
from commonlibs.math_tools.IOU import bbox_overlaps
from copy import deepcopy


# 第一个版本的error bar
# def cal_error_bar(x):
#     """
#     计算 均值 和 方差
#     :param x: N
#     :return: mean, var
#     """
#     m = torch.mean(x)
#     var = torch.sum((x - m)**2) / max(1, len(x) - 1)
#     return m, var

# 最小值、最大值的error bar
def cal_error_bar(x):
    """
    计算 均值 和 方差
    :param x: N 
    :return: mean, var 
    """
    low = torch.min(x)
    up = torch.max(x)
    return low, up


def match(bboxes1, bboxes2):
    """

    :param bboxes1: M x 4
    :param bboxes2: N x 4
    :return: matched_iou: M, matched_id: M
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return [], []
    ious = bbox_overlaps(bboxes1, bboxes2)
    matched_iou, matched_id = torch.max(ious, dim=1)
    return matched_iou, matched_id


def get_matched_results_no_cat(preds,
                               p_labels,
                               targets,
                               t_labels):
    """
    modified from bbox_nms
    :param preds: M x 4 or M x 4*(C + 1)
    :param targets: N x 4
    :return: 
    """
    match_results = [{'bbox': b.numpy(),
                      'label': int(l.item()),
                      'matched_iou': [],
                      'matched_label': []}
                     for b, l in zip(targets, t_labels)]

    ious, ids = match(preds, targets)

    for i, iou, id in zip(range(len(ious)), ious, ids):
        if iou < 0.5:
            continue
        match_results[id]['matched_iou'].append(float(iou))
        match_results[id]['matched_label'].append(int(p_labels[i]))

    return match_results


def get_matched_results(preds,
                        pred_scores,
                        labels,
                        targets,
                        gt_labels,
                        cats,
                        score_thr=0.05):
    """
    modified from bbox_nms
    :param preds: M x 4 or M x 4*(C + 1)
    pred_scores: M x 1 or M
    :param labels: M: 0~ C+1
    :param targets: N x 4
    :param cats:  C
    :return: 
    gt_match_results
    dt_matched_ious
    """
    # matched_dts：dt scores
    gt_match_results = [{'bbox': b.numpy(),
                         'label': int(l.item()),
                         'matched_iou': [],
                         'matched_label': [],
                         'matched_dt': [],
                         'matched_dts': []}
                        for b, l in zip(targets, gt_labels)]
    dt_matched_ious = torch.zeros(len(preds)).float()
    abs_ids = torch.arange(0, len(targets))
    n_cat = len(cats)
    # 0 -> background
    for c in range(1, n_cat):
        # get predict bboxes in class c
        dt_cls_inds = labels == c
        if not dt_cls_inds.any():
            continue
        # faster style or retina style
        if preds.shape[1] == 4:
            cat_preds = preds[dt_cls_inds, :]
        else:
            cat_preds = preds[dt_cls_inds, c * 4:(c + 1) * 4]
        cat_scores = pred_scores.flatten()[dt_cls_inds]
        # get target bboxes in class c
        gt_cls_inds = gt_labels == c
        gt_cls_abs_ids = abs_ids[gt_cls_inds]
        if not gt_cls_inds.any():
            continue
        cat_targets = targets[gt_cls_inds]
        ious, ids_in_cat = match(cat_preds, cat_targets)

        # dt ious
        dt_matched_ious[dt_cls_inds] = ious

        # gt matched results
        for iou, dt, s, id_in_cat in \
                zip(ious, cat_preds, cat_scores, ids_in_cat):
            abs_id = gt_cls_abs_ids[id_in_cat]
            if iou < 0.5:
                continue
            dt = [float(x) for x in dt]
            gt_match_results[abs_id]['matched_iou'].append(float(iou))
            gt_match_results[abs_id]['matched_label'].append(int(c))
            gt_match_results[abs_id]['matched_dt'].append(dt)
            gt_match_results[abs_id]['matched_dts'].append(float(s))

    return gt_match_results, list(dt_matched_ious.numpy())


def cal_gt_recall_matched(mrs, thrs, accumulate):
    """

    :param mrs: [{bbox=[], label=[], matched_iou=[]}]
    :return: 
    对于每一个阈值：
    1. gt的匹配总数目（图片中）
    2. dt的匹配总数目（图片中）
    3. 每一个gt匹配的dt数目
    4. 每一个gt匹配的dt的error bar：[low-eps, up+eps]，
    5. 每一个gt匹配的dt的score均值
    """

    results = {thr: [0.0, 0.0, [], [], []] for thr in thrs}
    e_thrs = deepcopy(thrs)
    e_thrs.append(max(e_thrs) + 100)
    eps = 1e-6
    for t in range(len(e_thrs) - 1):
        m_g = 0  # 匹配上的gt数目
        m_d = 0  # 总共的匹配dt数目
        m_ds = []  # 每一个gt匹配的dt数目
        m_ds_score_low = []
        m_ds_score_up = []
        m_ds_score_mean = []
        # for every gt
        for mr in mrs:
            matched_iou = mr['matched_iou']
            matched_score = mr['matched_dts']
            matched_iou = torch.Tensor(matched_iou)
            matched_score = torch.Tensor(matched_score)

            if accumulate:
                int_ids = (e_thrs[t] < matched_iou)  # & (matched_iou <= thr + 0.1)
            else:
                int_ids = (e_thrs[t] < matched_iou) & (matched_iou <= e_thrs[t + 1])
            m = int(torch.sum(int_ids))
            m_d += m
            m_ds.append(m)
            if m == 0:
                m_ds_score_low.append(0)
                m_ds_score_up.append(0)
                m_ds_score_mean.append(0)
                continue
            # 计算当前区间的这些det所对应的score的error bar
            low, up = cal_error_bar(matched_score[int_ids].flatten())
            mean = torch.mean(matched_score[int_ids].flatten())
            m_ds_score_mean.append(float(mean))
            m_ds_score_low.append(float(low))
            m_ds_score_up.append(float(up))
            m_g += 1
            # print(m, int_ids)
        # print('KKKKKKKKKKKKKKKKKKKKK')
        results[thrs[t]][0] = int(m_g)
        results[thrs[t]][1] = int(m_d)
        results[thrs[t]][2] = m_ds
        error_bars = [[l - eps, u + eps] for l, u in
                      zip(m_ds_score_low, m_ds_score_up)]
        results[thrs[t]][3] = error_bars
        results[thrs[t]][4] = m_ds_score_mean

    return results


def cal_gt_recall_matched(mrs, thrs, accumulate):
    """

    :param mrs: [{bbox=[], label=[], matched_iou=[]}]
    :return: 
    对于每一个阈值：
    1. gt的匹配总数目（图片中）
    2. dt的匹配总数目（图片中）
    3. 每一个gt匹配的dt数目
    4. 每一个gt匹配的dt的error bar：[low-eps, up+eps]，
    5. 每一个gt匹配的dt的score均值
    """

    results = {thr: [0.0, 0.0, [], [], []] for thr in thrs}
    e_thrs = deepcopy(thrs)
    e_thrs.append(max(e_thrs) + 100)
    eps = 1e-6
    for t in range(len(e_thrs) - 1):
        m_g = 0  # 匹配上的gt数目
        m_d = 0  # 总共的匹配dt数目
        m_ds = []  # 每一个gt匹配的dt数目
        m_ds_score_low = []
        m_ds_score_up = []
        m_ds_score_mean = []
        # for every gt
        for mr in mrs:
            matched_iou = mr['matched_iou']
            matched_score = mr['matched_dts']
            matched_iou = torch.Tensor(matched_iou)
            matched_score = torch.Tensor(matched_score)

            if accumulate:
                int_ids = (e_thrs[t] < matched_iou)  # & (matched_iou <= thr + 0.1)
            else:
                int_ids = (e_thrs[t] < matched_iou) & (matched_iou <= e_thrs[t + 1])
            m = int(torch.sum(int_ids))
            m_d += m
            m_ds.append(m)
            if m == 0:
                m_ds_score_low.append(0)
                m_ds_score_up.append(0)
                m_ds_score_mean.append(0)
                continue
            # 计算当前区间的这些det所对应的score的error bar
            low, up = cal_error_bar(matched_score[int_ids].flatten())
            mean = torch.mean(matched_score[int_ids].flatten())
            m_ds_score_mean.append(float(mean))
            m_ds_score_low.append(float(low))
            m_ds_score_up.append(float(up))
            m_g += 1
            # print(m, int_ids)
        # print('KKKKKKKKKKKKKKKKKKKKK')
        results[thrs[t]][0] = int(m_g)
        results[thrs[t]][1] = int(m_d)
        results[thrs[t]][2] = m_ds
        error_bars = [[l - eps, u + eps] for l, u in
                      zip(m_ds_score_low, m_ds_score_up)]
        results[thrs[t]][3] = error_bars
        results[thrs[t]][4] = m_ds_score_mean

    return results


def count_dt(dt_ious, thrs, accumulate):
    """

    :param mrs: [{bbox=[], label=[], matched_iou=[]}]
    :return: 
        avg_matched_dt_count：平均匹配个数（正例，也就是匹配上的进行计算）
        avg_precision：平均 平均精度（正例）
        best_precision：平均 最佳精度（正例）
        recall：recall值
    """

    results = {thr: 0 for thr in thrs}

    n_dt = len(dt_ious)
    if n_dt == 0:
        return results
    dt_ious = torch.Tensor(dt_ious)

    e_thrs = deepcopy(thrs)
    e_thrs.append(max(thrs) + 100)
    for t in range(len(e_thrs) - 1):
        if accumulate:
            int_ids = (e_thrs[t] < dt_ious)
        else:
            int_ids = (e_thrs[t] < dt_ious) & (dt_ious <= e_thrs[t + 1])
        results[thrs[t]] = int(torch.sum(int_ids))
    return results


