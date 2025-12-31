# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from mmcv.ops import box_iou_rotated

def obb2hbb(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs : [x_ctr,y_ctr,w,h,angle]

    Returns:
        outer hbb in obb format
    """
    N = obboxes.shape[0]
    if N == 0:
        return obboxes
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    xyxy = torch.cat([center - bias, center + bias], dim=-1)
    x1, y1, x2, y2 = torch.split(xyxy, [1, 1, 1, 1], dim=-1)
    a = torch.zeros_like(x1).to(obboxes.device)
    hbb = torch.cat([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, a], dim=-1)

    return hbb

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
    epoch = 36
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
    data_root = '/data/space2/huangziyue/xView_New_800_600/train'
    img_dir = f'{data_root}/images'
    ann_dir = f'{data_root}/annfiles'
    class_name = ['Aircraft_Hangar', 'Barge', 'Building', 'Bus', 'Cargo_Truck',
                  'Cargo_or_Container_Car', 'Cement_Mixer', 'Construction_Site',
                  'Container_Crane', 'Container_Ship', 'Crane_Truck',
                  'Damaged_Building', 'Dump_Truck', 'Engineering_Vehicle',
                  'Excavator', 'Facility', 'Ferry', 'Fishing_Vessel',
                  'Fixed-wing_Aircraft', 'Flat_Car', 'Front_loader_or_Bulldozer',
                  'Ground_Grader', 'Haul_Truck', 'Helicopter', 'Helipad', 'Hut_or_Tent',
                  'Locomotive', 'Maritime_Vessel', 'Mobile_Crane', 'Motorboat',
                  'Oil_Tanker', 'Passenger_Car', 'Passenger_Vehicle',
                  'Passenger_or_Cargo_Plane', 'Pickup_Truck', 'Pylon', 'Railway_Vehicle',
                  'Reach_Stacker', 'Sailboat', 'Scraper_or_Tractor',
                  'Shed', 'Shipping_Container', 'Shipping_container_lot',
                  'Small_Aircraft', 'Small_Car', 'Storage_Tank', 'Straddle_Carrier',
                  'Tank_car', 'Tower', 'Tower_crane', 'Trailer', 'Truck',
                  'Truck_Tractor', 'Truck_Tractor_with_Box_Trailer',
                  'Truck_Tractor_with_Flatbed_Trailer', 'Truck_Tractor_with_Liquid_Tank',
                  'Tugboat', 'Utility_Truck', 'Vehicle_Lot', 'Yacht']
    id2name = {i: c for i, c in enumerate(class_name)}

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
    ##################################################################################
    # vis_dir = './Step1_Trans_HBB2OBB'
    #
    # mkdir(vis_dir)
    # with torch.no_grad():
    #     for data_info in data_loader:
    #         if count > 50:
    #             break
    #         data = model.data_preprocessor(data_info, False)
    #         data_samples = data['data_samples']
    #         for sample in data_samples:
    #             sample.gt_instances.labels = sample.gt_instances.labels * 0
    #
    #         all_results_list, out_proposals_list = model.predict(data['inputs'],
    #                                                              data['data_samples'],
    #                                                              rescale=True,
    #                                                              gt_obb2hbb=False)
    #         count += 1
    #
    #         for r_id, results_list in enumerate(all_results_list):
    #             pred_bboxes = results_list[0].bboxes
    #             gt_bboxes = out_proposals_list[r_id][0].bboxes
    #             img_pth = data['data_samples'][0].img_path
    #
    #             img = cv2.imread(img_pth)
    #             img_name = Path(img_pth).stem
    #             ## ----- Org -> Inputs(模型推理)
    #             ## ----- Org -> Resize(可视化)
    #             ## ----- scale_factor = H_Inputs / H_Org
    #             ## ----- gt_bboxes: Org -> Scale to -> Inputs
    #             ## ----- pred_bboxes: Inputs -> Scale to -> Org
    #             ## ----- 所以：gt_bboxes 先 Scale to -> Org，然后一起缩放就行
    #             sw1, sh1 = data['data_samples'][0].scale_factor
    #             H, W, _ = img.shape
    #             sw2, sh2 = W / 800, H / 800
    #             vis_gt_bboxes = deepcopy(gt_bboxes)
    #             vis_gt_bboxes[:, :4] /= torch.tensor([sw1 * sw2, sh1 * sh2,
    #                                                   sw1 * sw2, sh1 * sh2]).to(gt_bboxes.device)[None, ...]
    #             vis_pred_bboxes = deepcopy(pred_bboxes)
    #             vis_pred_bboxes[:, :4] /= torch.tensor([sw2, sh2,
    #                                                     sw2, sh2]).to(gt_bboxes.device)[None, ...]
    #             img = cv2.resize(img, (800, 800))
    #
    #             gt_polys = obb2poly(vis_gt_bboxes).detach().cpu()
    #             #---- gt红色
    #             draw_polys(img, gt_polys.numpy(), color=(0, 0, 255), thickness=1)
    #             pred_polys = obb2poly(vis_pred_bboxes).detach().cpu()
    #             #---- dt绿色
    #             draw_polys(img, pred_polys.numpy(), color=(0, 255, 0), thickness=1)
    #             img_pth = vis_dir + f'/{img_name}_I{r_id}.png'
    #             cv2.imwrite(img_pth, img)
    #             print(f'Save:{img_pth}, n_gt: {len(gt_polys)}, n_pred: {len(pred_polys)}')

    iou_thr = 0.7
    out_ann_dir = f'{data_root}/Step1_Trans_HBB2OBB_IoU{iou_thr}'

    mkdir(out_ann_dir)
    with torch.no_grad():
        for data_info in data_loader:
            # count += 1
            # if count > 50:
            #     break

            data = model.data_preprocessor(data_info, False)
            data_samples = data['data_samples']
            gt_labels = []
            for sample in data_samples:
                gt_labels.append(deepcopy(sample.gt_instances.labels))
                sample.gt_instances.labels = sample.gt_instances.labels * 0

            all_results_list, out_proposals_list = model.predict(data['inputs'],
                                                                 data['data_samples'],
                                                                 rescale=True,
                                                                 gt_obb2hbb=False,
                                                                 sample_time=64,
                                                                 noise_scale=0.01)
            sw1, sh1 = data['data_samples'][0].scale_factor
            all_pred_rboxes = []
            for r_id, results_list in enumerate(all_results_list):
                pred_boxes = results_list[0].bboxes
                all_pred_rboxes.append(deepcopy(pred_boxes))
            all_pred_rboxes = torch.stack(all_pred_rboxes, dim=1) # N x R x 8
            gt_rboxes = deepcopy(data['data_samples'][0].gt_instances.bboxes.tensor)
            gt_rboxes[:, :4] /= torch.tensor([sw1, sh1,
                                              sw1, sh1]).to(gt_rboxes.device)[None, ...]
            if len(gt_rboxes) == 0:
                print(f'Pass empty file: {img_name}')
                continue
            out_pred_polys = []
            for pred_boxes, gt_b in zip(all_pred_rboxes, gt_rboxes):
                ##### ----- 利用外接框一致性筛选不合理的预测框
                ext_hbbs = obb2hbb(pred_boxes)
                ious = box_iou_rotated(ext_hbbs, gt_b[None, ...]).flatten() # R
                valid_pred_boxes = pred_boxes[ious >= iou_thr]

                if len(valid_pred_boxes) == 0:
                    out_pred_polys.append(obb2poly(gt_b[None, ...])[0])
                    print('No valid ious: ', ious)
                else:
                    ##### ----- 去掉角度离群值
                    pred_angle = valid_pred_boxes[:, -1]
                    angle_mean = torch.mean(pred_angle, dim=-1)
                    if len(pred_angle) > 0:
                        angle_std = torch.std(pred_angle)
                    else:
                        angle_std = torch.tensor([1.0]).to(angle_mean.device)[0]
                    angle_delta = torch.abs(pred_angle - angle_mean)
                    valid_angle_idx = angle_delta < angle_std
                    valid_pred_boxes = valid_pred_boxes[valid_angle_idx]

                    if torch.sum(valid_pred_boxes) == 0:
                        out_pred_polys.append(obb2poly(gt_b[None, ...])[0])
                        print('No valid angles: ', pred_angle, angle_delta, angle_mean, angle_std)
                    else:
                        mean_box = torch.mean(valid_pred_boxes, dim=0)
                        mean_poly = obb2poly(mean_box[None, ...])[0]
                        out_pred_polys.append(mean_poly)
                # ##### ----- 可以添加一个额外操作，去掉偏差值，先不做
                # norm_polys = polys / torch.abs(poly_m[None, ...])            # R x 8
                # poly_std = torch.mean(torch.std(norm_polys, dim=0), dim=-1)  # N
            out_pred_polys = torch.stack(out_pred_polys)
            gt_label = gt_labels[0]

            img_pth = data['data_samples'][0].img_path
            img_name = Path(img_pth).stem
            ann_file_path = out_ann_dir + '/' + img_name + '.txt'
            print(f'Save: {ann_file_path}')
            with open(ann_file_path, 'wt+') as f:
                assert len(gt_label) == len(out_pred_polys)
                for l, poly in zip(gt_label, out_pred_polys):
                    poly = poly.detach().cpu().numpy().tolist()
                    x1, y1, x2, y2, x3, y3, x4, y4 = poly
                    name = id2name[int(l)]
                    difficulty = 0
                    f.write('%.3f %.3f %.3f %.3f '
                            '%.3f %.3f %.3f %.3f %s %d\n' %
                            (x1, y1,
                             x2, y2,
                             x3, y3,
                             x4, y4,
                             name, difficulty))

    ######################################################## Old version
    # with torch.no_grad():
    #     for data_info in data_loader:
    #         count += 1
    #         if count > 50:
    #             break
    #
    #         data = model.data_preprocessor(data_info, False)
    #         data_samples = data['data_samples']
    #         gt_labels = []
    #         for sample in data_samples:
    #             gt_labels.append(deepcopy(sample.gt_instances.labels))
    #             sample.gt_instances.labels = sample.gt_instances.labels * 0
    #
    #         all_results_list, out_proposals_list = model.predict(data['inputs'],
    #                                                              data['data_samples'],
    #                                                              rescale=True,
    #                                                              gt_obb2hbb=False,
    #                                                              sample_time=33,
    #                                                              noise_scale=0.05)
    #         all_pred_polys = []
    #         for r_id, results_list in enumerate(all_results_list):
    #             pred_bboxes = results_list[0].bboxes
    #             pred_polys = obb2poly(pred_bboxes)
    #             all_pred_polys.append(pred_polys)
    #         all_polys = torch.stack(all_pred_polys, dim=0) # R x N x 8
    #         norm_polys = all_polys / torch.mean(all_polys, dim=0, keepdim=True) #  R x N x 8
    #         poly_std = torch.mean(torch.std(norm_polys, dim=0), dim=-1)  #  N
    #         poly_mean = torch.mean(all_polys, dim=0) #  N x 8
    #         gt_label = gt_labels[0]
    #
    #         img_pth = data['data_samples'][0].img_path
    #         img_name = Path(img_pth).stem
    #         ann_file_path = out_ann_dir + '/' + img_name + '.txt'
    #         print(f'Save: {ann_file_path}')
    #         with open(ann_file_path, 'wt+') as f:
    #             assert len(gt_label) == len(poly_mean) == len(poly_std)
    #             for l, poly, std in zip(gt_label, poly_mean, poly_std):
    #                 poly = poly.detach().cpu().numpy().tolist()
    #                 x1, y1, x2, y2, x3, y3, x4, y4 = poly
    #                 name = id2name[int(l)]
    #                 difficulty = 0
    #                 std = float(std)
    #                 f.write('%.3f %.3f %.3f %.3f '
    #                         '%.3f %.3f %.3f %.3f %s %d %.3f\n' %
    #                         (x1, y1,
    #                          x2, y2,
    #                          x3, y3,
    #                          x4, y4,
    #                          name, difficulty, std))



if __name__ == '__main__':
    main()
