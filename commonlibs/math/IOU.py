import torch

def singleIOU(gt, bboxes):
    """
    
    :param gt: left top right down
    :param bboxes: N * 4
    :return: 
    """
    [x1, y1, x2, y2] = gt
    inter_lt = (gt[0].max(bboxes[:, 0]), gt[1].max(bboxes[:, 1]))
    inter_rd = (gt[2].min(bboxes[:, 2]), gt[3].min(bboxes[:, 3]))
    z = torch.Tensor([0.0])
    inter_w = (inter_rd[0] - inter_lt[0]).max(z)
    inter_h = (inter_rd[1] - inter_lt[1]).max(z)
    inter_area = inter_w * inter_h
    area_gt = ((x2 - x1)*(y2 - y1)).max(z)
    if area_gt <= 0:
        return torch.zeros(bboxes.shape[0])
    area_bboxes = ((bboxes[:, 2] - bboxes[:, 0]) * \
                  (bboxes[:, 3] - bboxes[:, 1])).max(z)
    IOU = inter_area / (area_gt + area_bboxes - inter_area)
    return IOU

def IOU(gts, bboxes):
    """
    
    :param gts: M * 4
    :param bboxes: N * 4
    :return: M * N
    """
    IOU = []
    for gt in gts:
        IOU.append(singleIOU(gt, bboxes).reshape(1, -1))
    return torch.cat(IOU, dim=0)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])
    return ious


if __name__ == '__main__':
    gts = torch.Tensor([[3, 3, 6, 6],
                       [0, 0, 1, 1],
                       [1, 1, 1, 1]])
    bboxes = torch.Tensor([[3,3,6,6],
                           [4,4,7,7],
                           [2,2,5,5],
                           [3,2,6,5]])
    print(singleIOU(gts[0], bboxes))
    print(singleIOU(torch.Tensor([0, 0, 1, 1]), bboxes))
    print(singleIOU(torch.Tensor([1, 1, 1, 1]), bboxes))
    print(IOU(gts, bboxes))
























