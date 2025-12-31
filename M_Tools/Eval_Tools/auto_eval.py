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


def main(tst_cfg_pth, ckpt_pth, work_dir, results_pth, eval_pth,
         data_root, img_dir, ann_dir, class_names, img_scale,
         val_support_classes,
         val_dataset_flag,
         eval_mode='Normal',
         val_using_aux=False):

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
    cfg.model.val_support_classes = val_support_classes
    cfg.model.val_dataset_flag = val_dataset_flag
    cfg.model.val_using_aux = val_using_aux
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
    from M_Tools.Eval_Tools.eval_configs.mini_test_data_cfgs import eval_cfgs
    from ctlib.os import *
    os.chdir('/opt/data/nfs/huangziyue/Projects/MMRotate_AD')

    #####################################################################
    # prefix = '9_6_flex_rtm_v1_mini_test'
    # epoch = 24
    # tst_cfg_pth = './M_configs/A09_Large_Pretrain_Stage2/A09_flex_rtm_v1_base.py'
    # ckpt_pth = f'./results/MMR_AD_A09_flex_rtm_v1_base/epoch_{epoch}.pth'
    # work_root = './results/TEST_EVAL'
    # work_dir = f'{work_root}/eval_logs'
    # eval_results_root = f'{work_root}/{prefix}_Epoch_{epoch}'
    # mkdir(work_root)
    # mkdir(work_dir)
    # mkdir(eval_results_root)
    #
    # results_dir = eval_results_root
    # eval_dir = eval_results_root
    #
    # error_data_set_names = []
    # for data_name in eval_cfgs.keys():
    #     cfg = eval_cfgs[data_name]
    #     results_pth = f'{results_dir}/{data_name}.pkl'
    #     eval_pth = f'{results_dir}/{data_name}.json'
    #     try:
    #         main(tst_cfg_pth,
    #              ckpt_pth=ckpt_pth, work_dir=work_dir, results_pth=results_pth, eval_pth=eval_pth,
    #              data_root=cfg['data_root'], img_dir=cfg['img_dir'],
    #              ann_dir=cfg['ann_dir'], class_names=cfg['class_names'], img_scale=cfg['img_scale'],
    #              val_support_classes=cfg['class_names'],
    #              val_dataset_flag=cfg['val_dataset_flag']
    #              )
    #     except:
    #         error_data_set_names.append(data_name)
    # if len(error_data_set_names) > 0:
    #     with open(f'{eval_results_root}/error_logs.txt', 'wt+') as f:
    #         for name in error_data_set_names:
    #             f.write(f'{name}\n')
    #####################################################################
    """
    val_using_aux = True
    prefix = '9_22_A09_flex_rtm_v3_1_fast_neg_UsingAux'
    cfg_name = 'A09_flex_rtm_v3_1_fast_neg'
    
    val_using_aux = False
    prefix = '9_22_A09_flex_rtm_v3_with_neg'
    cfg_name = 'A09_flex_rtm_v3_with_neg'
    epoch = 24
    
    val_using_aux = False
    prefix = '10_8_A09_flex_rtm_v3_with_image_text_long'
    cfg_name = 'A09_flex_rtm_v3_with_image_text_long'
    epoch = 32
    
    val_using_aux = False
    prefix = '10_8_A09_flex_rtm_v3_with_neg_softmax1'
    cfg_name = 'A09_flex_rtm_v3_with_neg_softmax1'
    epoch = 24
    
    val_using_aux = True
    prefix = '10_15_A10_Gen_rtm_v1_self_label_UsingAux'
    cfg_name = 'A10_Gen_rtm_v1_self_label'
    epoch = 24
    
    val_using_aux = False
    prefix = '10_21_A10_Gen_rtm_v2_neg_img'
    cfg_name = 'A10_Gen_rtm_v2_neg_img'
    epoch = 12
    
    val_using_aux = False
    prefix = '10_23_A10_Gen_rtm_v2_round2'
    cfg_name = 'A10_Gen_rtm_v2_round2'
    epoch = 12
    
    
    """
    val_using_aux = False
    prefix = '10_28_A10_Gen_rtm_v3_0_pretrain'
    cfg_name = 'A10_Gen_rtm_v3_0_pretrain'
    epoch = 24
    tst_cfg_pth = f'./M_configs/A10_Large_Pretrain_Stage3/{cfg_name}.py'
    ckpt_pth = f'./results/MMR_AD_{cfg_name}/epoch_{epoch}.pth'
    work_root = './results/TEST_EVAL'
    work_dir = f'{work_root}/eval_logs'
    eval_results_root = f'{work_root}/{prefix}_Epoch_{epoch}'
    mkdir(work_root)
    mkdir(work_dir)
    mkdir(eval_results_root)

    results_dir = eval_results_root
    eval_dir = eval_results_root

    error_data_set_names = []
    for data_name in eval_cfgs.keys():
        cfg = eval_cfgs[data_name]
        results_pth = f'{results_dir}/{data_name}.pkl'
        eval_pth = f'{results_dir}/{data_name}.json'
        main(tst_cfg_pth,
             ckpt_pth=ckpt_pth, work_dir=work_dir, results_pth=results_pth, eval_pth=eval_pth,
             data_root=cfg['data_root'], img_dir=cfg['img_dir'],
             ann_dir=cfg['ann_dir'], class_names=cfg['class_names'], img_scale=cfg['img_scale'],
             val_support_classes=cfg['class_names'],
             val_dataset_flag=data_name,
             val_using_aux=val_using_aux
             )
        # try:
        #     main(tst_cfg_pth,
        #          ckpt_pth=ckpt_pth, work_dir=work_dir, results_pth=results_pth, eval_pth=eval_pth,
        #          data_root=cfg['data_root'], img_dir=cfg['img_dir'],
        #          ann_dir=cfg['ann_dir'], class_names=cfg['class_names'], img_scale=cfg['img_scale'],
        #          val_support_classes=cfg['class_names'],
        #          val_dataset_flag=cfg['val_dataset_flag']
        #          )
        # except:
        #     error_data_set_names.append(data_name)
    if len(error_data_set_names) > 0:
        with open(f'{eval_results_root}/error_logs.txt', 'wt+') as f:
            for name in error_data_set_names:
                f.write(f'{name}\n')
