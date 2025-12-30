import torch
from commonlibs.math_tools.IOU import IOU

def _nms(scores, distances, threshold):
    """
    graph based nms
    :param scores: N Tensor. scores for finding maximum
    :param distances: N x N Tensor. distances of nodes
    :param threshold: float. dis < threshold => neighborhood
    :return: 
    local maximum scores(sorted), 
    local maximum indexes(in scores) [0, 1]
    """
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    suppressed = scores.new_zeros(len(scores))
    maximum = scores.new_zeros(len(scores))
    max_in_sorted = scores.new_zeros(len(scores))

    suppressed_num = 0

    for idx_in_sorted, idx in enumerate(sorted_indices):
        if suppressed[idx]:
            continue
        distance = distances[idx, :]
        maximum[idx] = 1
        max_in_sorted[idx_in_sorted] = 1

        neighbor_indices = distance <= threshold
        neighbor_num = torch.sum(neighbor_indices)
        # +1 means including him self
        suppressed_num += neighbor_num - torch.sum(suppressed[neighbor_indices]) + 1
        suppressed[neighbor_indices] = 1
        suppressed[idx] = 1

        if suppressed_num == len(scores):
            break
    max_scores = sorted_scores[max_in_sorted == 1]
    max_indices = maximum == 1

    return max_scores, max_indices

def nms(scores, bboxes, threshold):
    """
    
    :param scores: N
    :param bboxes: N x 4
    :return: max_scores, max_bboxes, max_index
    """
    ious = IOU(bboxes, bboxes)
    max_scores, max_indices = _nms(scores, ious, threshold)
    max_bboxes = bboxes[max_indices]

    return max_scores, max_bboxes, max_indices

if __name__ == '__main__':
    score = torch.Tensor([-0.1, 0.3, 0.1, 0.5, 0.1, 8, 9, 10, 11])
    coordinate = torch.Tensor([0, 1, 2, 6, 7, 8, 9, 9.5, 10])
    iou = torch.zeros([len(score), len(score)])
    for i, s1 in enumerate(coordinate):
        for j, s2 in enumerate(coordinate):
            iou[i, j] = (s1 - s2)**2
    print(iou)
    print(_nms(score, iou, 1))



