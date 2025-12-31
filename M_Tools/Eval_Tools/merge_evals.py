import os

from ctlib.os import *
from pathlib import Path

def merge_evals(eval_files):
    merge_eval_results = dict()
    for eval_file in eval_files:
        eval_results = jsonload(eval_file)
        data_name = Path(eval_file).stem
        map = eval_results['dota/mAP']
        merge_eval_results[data_name] = float(map)

    for map in merge_eval_results.values():
        print(map, end=' ')


if __name__ == "__main__":
    """
    eval_dir = f'{eval_root}/9_11_flex_rtm_v2_mini_test_Epoch_24'
    eval_dir = f'{eval_root}/9_22_A09_flex_rtm_v3_1_fast_neg_Epoch_24'
    eval_dir = f'{eval_root}/9_22_A09_flex_rtm_v3_1_fast_neg_UsingAux_Epoch_24'
    eval_dir = f'{eval_root}/9_22_A09_flex_rtm_v3_with_neg_Epoch_24'
    eval_dir = f'{eval_root}/10_8_A09_flex_rtm_v3_with_image_text_long_Epoch_32'
    eval_dir = f'{eval_root}/10_8_A09_flex_rtm_v3_with_image_text_Epoch_36'
    eval_dir = f'{eval_root}/10_8_A09_flex_rtm_v3_with_neg_softmax1_Epoch_24'
    eval_dir = f'{eval_root}/10_15_A10_Gen_rtm_v1_self_label_Epoch_24'
     eval_dir = f'{eval_root}/10_15_A10_Gen_rtm_v1_self_label_UsingAux_Epoch_24'
    eval_dir = f'{eval_root}/10_21_A10_Gen_rtm_v2_neg_img_Epoch_12'

    """
    eval_root = '/data/space2/huangziyue/mmdet_results/TEST_EVAL'
    eval_dir = f'{eval_root}/10_23_A10_Gen_rtm_v2_round2_Epoch_12'

    eval_files = []
    for file in sorted(os.listdir(eval_dir)):
        if not Path(file).suffix == '.json':
            continue
        eval_files.append(os.path.join(eval_dir, file))
    print(eval_files)
    merge_evals(eval_files)



