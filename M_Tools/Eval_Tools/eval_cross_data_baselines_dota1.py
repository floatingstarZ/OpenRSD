# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
PartID = 7
os.environ["CUDA_VISIBLE_DEVICES"] = f"{PartID}"

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules
from collections import OrderedDict

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


def main(tst_cfg_pth, ckpt_pth, work_dir, results_pth, eval_pth,
         data_root, img_dir, ann_dir, class_names, img_scale,):

    args = [f'{tst_cfg_pth}',
            f'{ckpt_pth}',
            '--work-dir', f'{work_dir}',
            '--out', f'{results_pth}',
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

    #################################
    # cfg.custom_hooks = None
    #############################################################################
    class_names = class_names
    id2name = {i: c for i, c in enumerate(class_names)}

    metainfo = dict(
        classes=class_names,
        # 注意：这个字段在最新版本中换成了小写
        palette=[(220, 20, 60), ]
        # 画图时候的颜色，随便设置即可
    )

    test_pipeline = [
        dict(type='mmdet.LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
        dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
        dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
        dict(
            type='mmdet.Pad', size=img_scale,
            pad_val=dict(img=(114, 114, 114))),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor'))
    ]
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.metainfo = metainfo
    cfg.test_dataloader.dataset.ann_file = ann_dir
    cfg.test_dataloader.dataset.data_prefix.img_path = img_dir
    cfg.test_dataloader.dataset.img_shape = (800, 800)
    cfg.test_dataloader.dataset.pipeline = test_pipeline

    ######################################################

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    eval_results = runner.test()
    jsonsave(eval_results, eval_pth)
    # print(eval_results)

if __name__ == '__main__':
    from M_Tools.Base_Data_infos.OOD_data_infos import ood_data_infos
    from ctlib.os import *
    os.chdir('/opt/data/nfs/huangziyue/Projects/MMRotate_AD')

    # val_using_aux = False
    # prefix = '10_28_A10_Gen_rtm_v3_0_pretrain'
    # cfg_name = 'A10_Gen_rtm_v3_0_pretrain'
    # epoch = 24
    base_data = 'Data1_DOTA1'
    exp_infos = OrderedDict(
        RTMDet_L=dict(
            cfg_pth=f'G02_Baselines/{base_data}/G02_Baselines_{base_data}_M10_RTMDet_L.py',
            cfg_name=f'G02_Baselines_{base_data}_M10_RTMDet_L',
            epoch=36,
        ),
        RtnNetOBB=dict(
            cfg_pth=f'G02_Baselines/{base_data}/G02_Baselines_{base_data}_M1_RtnNetOBB.py',
            cfg_name=f'G02_Baselines_{base_data}_M1_RtnNetOBB',
            epoch=12,
        ),
        RoITrans=dict(
            cfg_pth=f'G02_Baselines/{base_data}/G02_Baselines_{base_data}_M2_RoITrans.py',
            cfg_name=f'G02_Baselines_{base_data}_M2_RoITrans',
            epoch=12,
        ),
        S2ANet=dict(
            cfg_pth=f'G02_Baselines/{base_data}/G02_Baselines_{base_data}_M3_S2ANet.py',
            cfg_name=f'G02_Baselines_{base_data}_M3_S2ANet',
            epoch=12,
        ),
        R3Det_KFIoU=dict(
            cfg_pth=f'G02_Baselines/{base_data}/G02_Baselines_{base_data}_M4_R3Det_KFIoU.py',
            cfg_name=f'G02_Baselines_{base_data}_M4_R3Det_KFIoU',
            epoch=12,
        ),
        GM5_ORCNN_R50=dict(
            cfg_pth=f'G02_Baselines/{base_data}/G02_Baselines_{base_data}_M5_ORCNN_R50.py',
            cfg_name=f'G02_Baselines_{base_data}_M5_ORCNN_R50',
            epoch=12,
        ),
        ORCNN_Swin_T=dict(
            cfg_pth=f'G02_Baselines/{base_data}/G02_Baselines_{base_data}_M6_ORCNN_Swin_T.py',
            cfg_name=f'G02_Baselines_{base_data}_M6_ORCNN_Swin_T',
            epoch=12,
        ),
        M9_RTMDet_M=dict(
            cfg_pth=f'G02_Baselines/{base_data}/G02_Baselines_{base_data}_M9_RTMDet_M.py',
            cfg_name=f'G02_Baselines_{base_data}_M9_RTMDet_M',
            epoch=36,
        ),
    )
    #########################################
    val_support_classes = ['baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field', 'harbor', 'helicopter',
                     'large-vehicle', 'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
                     'storage-tank', 'swimming-pool', 'tennis-court']
    val_dataset_flag = 'Data1_DOTA1'
    #########################################
    count = 0
    if PartID == -1:
        PartID = len(exp_infos) - 1

    for exp_name, model_info in exp_infos.items():
        if count != PartID:
            count += 1
            continue
        else:
            count += 1
        print(model_info)

        cfg_name = model_info['cfg_name']
        cfg_pth = model_info['cfg_pth']
        epoch = model_info['epoch']

        tst_cfg_pth = f'./M_configs/{cfg_pth}'
        ckpt_pth = f'./results/MMR_AD_{cfg_name}/epoch_{epoch}.pth'
        work_root = './results/TEST_EVAL'
        work_dir = f'{work_root}/eval_logs'
        eval_results_root = f'{work_root}/11_18_Cross_DOTA2Base_{cfg_name}_Epoch_{epoch}'

        mkdir(work_root)
        mkdir(work_dir)
        mkdir(eval_results_root)

        results_dir = eval_results_root
        eval_dir = eval_results_root

        error_data_set_names = []
        for data_name in ood_data_infos.keys():
            # if data_name not in ['Data26_SODA_A_800_150',]:
            #     continue
            if data_name not in ['Data21_VEDAI', 'Data23_UCAS_AOD', 'Data24_CORS_ADD_OBB',
                                 'Data25_DOSR', 'Data26_SODA_A_800_150']:
                continue

            cfg = ood_data_infos[data_name]
            results_pth = f'{results_dir}/{data_name}.pkl'
            eval_pth = f'{results_dir}/{data_name}.json'
            main(tst_cfg_pth,
                 ckpt_pth=ckpt_pth,
                 work_dir=work_dir,
                 results_pth=results_pth,
                 eval_pth=eval_pth,

                 data_root=cfg['data_root'],
                 img_dir=cfg['val_img_dir'],
                 ann_dir=cfg['val_cross_ann_dir'],
                 class_names=val_support_classes,
                 img_scale=cfg['img_scale'],
                 )
        if len(error_data_set_names) > 0:
            with open(f'{eval_results_root}/error_logs.txt', 'wt+') as f:
                for name in error_data_set_names:
                    f.write(f'{name}\n')
