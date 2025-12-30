trunc = 1e-10
cls_score = (torch.sqrt(cls_score + trunc) - math.sqrt(trunc)) / \
            (math.sqrt(1 + trunc) - math.sqrt(trunc))
cls_score = - torch.log((1 + 2 * trunc) / (cls_score + trunc) - 1)