# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules


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
    # #  ------------ 正常测试
    epoch = 4
    cfg = 'G02_Baselines_Data6_Xview_M1_RtnNetOBB'
    tst_cfg = 'G02_Baselines_Data6_Xview_M1_RtnNetOBB'

    # args = [f'./M_configs/G02_Baselines/Data6_Xview/{tst_cfg}.py',
    #         f'./results/MMR_AD_{cfg}/epoch_{epoch}.pth',
    #         '--work-dir', './results/TEST',
    #         '--out', f'./results/TEST/{tst_cfg}_Epoch_{epoch}.pkl'
    #         ]
    #
    # epoch = 8
    # cfg = 'A14_UpBound_GT_DATA2_DIOR_R'
    # tst_cfg = 'A14_UpBound_GT_DATA2_DIOR_R'
    # args = [f'./M_configs/A14_UpperBound/{tst_cfg}.py',
    #         f'./results/MMR_AD_{cfg}/epoch_{epoch}.pth',
    #         '--work-dir', './results/TEST',
    #         '--out', f'./results/TEST/{tst_cfg}_Epoch_{epoch}.pkl'
    #         ]
    #

    # epoch = 12
    # cfg = 'G02_Baselines_Data1_DOTA2_M7_ORCNN_ViT_B_MTP'
    # tst_cfg = cfg
    # args = [f'./M_configs/G02_Baselines/Data1_DOTA2/{tst_cfg}.py',
    #         f'./results/MMR_AD_{cfg}/epoch_{epoch}.pth',
    #         '--work-dir', './results/TEST',
    #         '--out', f'./results/TEST/{tst_cfg}_Epoch_{epoch}.pkl'
    #         ]

    # ------------ MiniTest
    # epoch = 24
    # cfg = 'A01_rtmdet_m_dior_v9_S0'
    #
    # args = [f'./M_configs/MiniTest/{cfg}.py',
    #         f'./results/MMR_AD_{cfg}/epoch_{epoch}.pth',
    #         '--work-dir', './results/TEST',
    #         '--out', f'./results/TEST/MiniTest_{cfg}_Epoch_{epoch}.pkl'
    #         ]
    # ------------ MiniTest
    # args = [f'./M_configs/A04_YOLO_RoI/A04_yolo_roi_base.py',
    #         f'./results/MMR_AD_A04_yolo_roi_base/epoch_2.pth',
    #         '--work-dir', './results/TEST',]

    # epoch = 24
    # cfg = 'A12_flex_rtm_v3_1_self_training_unfz'
    # tst_cfg = cfg
    # args = [f'./M_configs/A12_SelfTrain/{tst_cfg}.py',
    #         f'./results/MMR_AD_{cfg}/epoch_{epoch}.pth',
    #         '--work-dir', './results/TEST',
    #         '--out', f'./data/HBB_Eval_Outputs/{tst_cfg}_Epoch_{epoch}.pkl'
    #         ]

    epoch = 8
    cfg = 'A10_flex_rtm_v3_1_formal_with_hbb'
    data_name = 'Data11_WHU_Mix'

    args = [f'./M_configs/A10_Large_Pretrain_Stage3/'
            f'A10_flex_rtm_v3_1_formal_with_hbb_VAL_{data_name}.py',

            f'./results/MMR_AD_{cfg}/epoch_{epoch}.pth',

            '--work-dir', './results/TEST',

            '--out', f'./data/HBB_Eval_Outputs/'
                     f'/{cfg}_{data_name}.pkl'
            ]

    #################################################
    # epoch = 24
    # cfg = 'A10_flex_rtm_v3_1_formal'
    #
    # args = [f'./M_configs/A10_Large_Pretrain_Stage3/'
    #         f'A10_flex_rtm_v3_1_formal.py',
    #         f'./results/MMR_AD_{cfg}/epoch_{epoch}.pth',
    #         '--work-dir', './results/TEST',
    #         ]


    epoch = 1
    cfg = 'A13_Hin_rtm_v6_OneHead'

    args = [f'./M_configs/A13_InContext/'
            f'{cfg}.py',
            f'./results/MMR_AD_{cfg}/epoch_{epoch}.pth',
            '--work-dir', './results/TEST',
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
    runner.test()


if __name__ == '__main__':
    main()
