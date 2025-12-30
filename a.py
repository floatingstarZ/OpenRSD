import os
from ctlib.dota import *
all_names = []
root = '/data/space2/huangziyue/FMoW'
for i in range(6):
    ann_dir = f'{root}/FMoW_Part{i}/annotations'
    a, names = load_dota(ann_dir)
    all_names.extend(names)
all_names = sorted(list(set(all_names)))
print(all_names)
