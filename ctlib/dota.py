import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
def load_dota(ann_dir):
    ann_infos = dict()
    ann_files = sorted(list(os.listdir(ann_dir)))
    all_names = set()
    for ann_file in tqdm(ann_files):
        ann_name = Path(ann_file).stem
        try:
            ann_pth = ann_dir + '/' + ann_file
            with open(ann_pth) as f:
                lines = f.readlines()
                lines = [l.strip().split(' ') for l in lines]
            gt_polys = []
            gt_names = []
            for l in lines:
                poly = [float(coord) for coord in l[:8]]
                poly = np.array(poly)
                gt_polys.append(poly)
                gt_names.append(l[8])
                all_names.add(l[8])
            ann_infos[ann_name] = dict(
                polys=gt_polys,
                names=gt_names
            )
        except:
            print(f'Error in {ann_file}')
    return ann_infos, sorted(list(all_names))

