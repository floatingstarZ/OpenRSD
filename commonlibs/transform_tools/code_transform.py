import torch

# 各种编码之间的转化，one-hot, label


def label2onehot(label, Cat):
    label = label.view(-1, 1).long()
    one_hot_map = torch.zeros(len(label), Cat + 1)
    # scatter: dimension 1 , value 1
    one_hot_map = one_hot_map.scatter_(1, label, 1)
    return one_hot_map


if __name__ == '__main__':
    a = torch.Tensor([1, 2, 3, 3, 2, 1])
    print(label2onehot(a, 3))







