import torch
def clsmap2scoremap(cls_maps, use_softmax=False, Cat=80, Anchor=9, score_shel=0):
    """
    
    :param cls_maps: Cat * Anchor x H x W
    :param use_softmax: score如果使用了sigmoid，就可以不进行softmax操作, use_softmax可以设置为False
    :param Cat: 类别
    :param Anchor: Anchor个数
    :param score_shel: score threshold
    :return: score_maps: Anchor x H x W，每个anchor的最大score
             cat_maps: Cat x H x W，每个anchor的最大score对应的category   
    """
    map = torch.Tensor(cls_maps)
    assert map.shape[0] == Cat * Anchor
    _, H, W = cls_maps.shape
    score_maps = torch.zeros(Anchor, H, W)
    cat_maps = torch.zeros(Anchor, H, W)
    # 将map的每一个Anchor对应的那些channel转换为one_hot编码，并且滤除掉小于指定阈值的位置
    for a in range(Anchor):
        score_map = map[a * Cat: (a + 1) * Cat]
        if use_softmax:
            score_map = score_map.softmax(0)
        (max_score, ind) = torch.max(score_map, 0)
        cat_maps[a] = ind

        max_score[max_score <= score_shel] = 0
        score_maps[a] = max_score

    return score_maps, cat_maps


def clsmap2onehotmap(cls_maps, use_softmax=False, Cat=80, Anchor=9, score_shel=0):
    """

    :param cls_maps: Cat * Anchor x H x W
    :param use_softmax: score如果使用了sigmoid，就可以不进行softmax操作, use_softmax可以设置为False
    :param Cat: 类别
    :param Anchor: Anchor个数
    :param score_shel: score threshold
    :return: one_hot_maps: Anchor * (Cat + 1) x H x W，one-hot编码
    """
    map = torch.Tensor(cls_maps)
    assert map.shape[0] == Cat * Anchor
    _, H, W = cls_maps.shape
    one_hot_maps = torch.zeros(Anchor * (Cat + 1), H, W)
    # 将map的每一个Anchor对应的那些channel转换为one_hot编码，并且滤除掉小于指定阈值的位置
    for a in range(Anchor):
        score_map = map[a * Cat: (a + 1) * Cat]
        if use_softmax:
            score_map = score_map.softmax(0)
        (max_score, ind) = torch.max(score_map, 0)

        # if score < max_score, classify as background
        ind = ind + 1
        ind[max_score <= score_shel] = 0

        one_hot_map = torch.zeros(Cat + 1, H, W)
        one_hot_map = one_hot_map.scatter_(0, ind.unsqueeze(0), 1)
        one_hot_maps[a*(Cat+1): (a+1)*(Cat+1)] = one_hot_map
    return one_hot_maps


