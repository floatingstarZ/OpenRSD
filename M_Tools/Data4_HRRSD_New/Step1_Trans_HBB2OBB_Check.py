# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules

from mmcv.ops.roi_align_rotated import RoIAlignRotated
from mmdet.structures.bbox import bbox2roi
from mmrotate.registry import MODELS
import numpy as np
import torch
from collections import OrderedDict
from pathlib import Path
from commonlibs.common_tools import *
import open_clip
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union
from mmengine.structures.instance_data import InstanceData

from mmrotate.registry import DATASETS
from torch.utils.data import DataLoader
from ctlib.rbox import obb2poly, obb2xyxy
from ctlib.vis import draw_polys
from copy import deepcopy
import cv2
import math
from ctlib.os import *
import os
import numpy as np
import torch
# TODO: support fuse_conv_bn and format_only
def parse_args(args=''):
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    os.chdir('../../')
    epoch = 24
    cfg = 'A08_e_rtm_v2_M2_AnytoRotate'
    tst_cfg = 'A08_e_rtm_v2_M2_AnytoRotate'

    args = [f'./M_configs/A08_Large_Pretrain/{tst_cfg}.py',
            f'./results/MMR_AD_{cfg}/epoch_{epoch}.pth',
            '--work-dir', './results/TEST',
            '--out', f'./results/TEST/{tst_cfg}_Epoch_{epoch}.pkl'
            ]


    args = parse_args(args)

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    #############################################################################
    data_root = '/data/space2/huangziyue/TGRS_HRRSD/train_val'
    img_dir = f'{data_root}/images'
    ann_dir = f'{data_root}/labelTxt'
    class_name = ['airplane', 'baseball_diamond', 'basketball_court', 'bridge',
                  'crossroad', 'ground_track_field', 'harbor', 'parking_lot', 'ship',
                  'storage_tank', 't_junction', 'tennis_court', 'vehicle']

    metainfo = dict(
        classes=class_name,
        # 注意：这个字段在最新版本中换成了小写
        palette=[(220, 20, 60), ]
        # 画图时候的颜色，随便设置即可
    )

    test_pipeline = [
        dict(type='mmdet.LoadImageFromFile', file_client_args= dict(backend='disk')),
        dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
        dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
        dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
        dict(
            type='mmdet.Pad', size=(1024, 1024),
            pad_val=dict(img=(114, 114, 114))),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor'))
    ]
    test_dataset =dict(
        type='mmrotate.DOTADataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=ann_dir,
        data_prefix=dict(img_path=img_dir),
        img_shape=(1024, 1024),
        test_mode=True,
        pipeline=test_pipeline)

    dataloader_cfg = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False))
    sampler_seed = 2024
    dataset = DATASETS.build(test_dataset)
    sampler_cfg = dataloader_cfg.pop('sampler')
    sampler = DATA_SAMPLERS.build(
        sampler_cfg,
        default_args=dict(dataset=dataset, seed=sampler_seed))
    batch_sampler = None
    collate_fn_cfg = dict(type='pseudo_collate')
    collate_fn_type = collate_fn_cfg.pop('type')
    if isinstance(collate_fn_type, str):
        collate_fn = FUNCTIONS.get(collate_fn_type)
    else:
        collate_fn = collate_fn_type
    collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler if batch_sampler is None else None,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        **dataloader_cfg)


    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    runner.load_or_resume()

    model = runner.model
    model.eval()
    count = 0

    vis_dir = './Step1_Trans_HBB2OBB'

    mkdir(vis_dir)
    with torch.no_grad():
        for data_info in data_loader:
            if count > 50:
                break
            data = model.data_preprocessor(data_info, False)
            data_samples = data['data_samples']
            for sample in data_samples:
                sample.gt_instances.labels = sample.gt_instances.labels * 0

            all_results_list, out_proposals_list = model.predict(data['inputs'],
                                                                 data['data_samples'],
                                                                 rescale=True,
                                                                 gt_obb2hbb=False)
            count += 1

            for r_id, results_list in enumerate(all_results_list):
                pred_bboxes = results_list[0].bboxes
                gt_bboxes = out_proposals_list[r_id][0].bboxes
                img_pth = data['data_samples'][0].img_path

                img = cv2.imread(img_pth)
                img_name = Path(img_pth).stem
                ## ----- Org -> Inputs(模型推理)
                ## ----- Org -> Resize(可视化)
                ## ----- scale_factor = H_Inputs / H_Org
                ## ----- gt_bboxes: Org -> Scale to -> Inputs
                ## ----- pred_bboxes: Inputs -> Scale to -> Org
                ## ----- 所以：gt_bboxes 先 Scale to -> Org，然后一起缩放就行
                sw1, sh1 = data['data_samples'][0].scale_factor
                H, W, _ = img.shape
                sw2, sh2 = W / 800, H / 800
                # print('H, W', H, W)
                # print('gt_bboxes', obb2poly(gt_bboxes))
                # gt_bboxes[:, :4] /= torch.tensor([sw2, sh2,
                #                                   sw2, sh2]).to(gt_bboxes.device)[None, ...]
                vis_gt_bboxes = deepcopy(gt_bboxes)
                vis_gt_bboxes[:, :4] /= torch.tensor([sw1 * sw2, sh1 * sh2,
                                                      sw1 * sw2, sh1 * sh2]).to(gt_bboxes.device)[None, ...]
                # print('gt_bboxes', obb2poly(vis_gt_bboxes))
                vis_pred_bboxes = deepcopy(pred_bboxes)
                vis_pred_bboxes[:, :4] /= torch.tensor([sw2, sh2,
                                                        sw2, sh2]).to(gt_bboxes.device)[None, ...]
                # print('pred_bboxes', obb2poly(vis_pred_bboxes))
                img = cv2.resize(img, (800, 800))

                gt_polys = obb2poly(vis_gt_bboxes).detach().cpu()
                #---- gt红色
                draw_polys(img, gt_polys.numpy(), color=(0, 0, 255), thickness=1)
                pred_polys = obb2poly(vis_pred_bboxes).detach().cpu()
                #---- dt绿色
                draw_polys(img, pred_polys.numpy(), color=(0, 255, 0), thickness=1)
                img_pth = vis_dir + f'/{img_name}_I{r_id}.png'
                cv2.imwrite(img_pth, img)
                print(f'Save:{img_pth}, n_gt: {len(gt_polys)}, n_pred: {len(pred_polys)}')


if __name__ == '__main__':
    main()
