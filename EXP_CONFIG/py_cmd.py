# python tst_py_cmd.py --gpus 6,7
# python tst_py_cmd.py --

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import argparse
import sys
from EXP_CONFIG.CONFIGS.base import Project_root
import os
sys.path.append(Project_root)


cfgs = dict()
from EXP_CONFIG.CONFIGS.all_config import all_cfgs
cfgs.update(all_cfgs)

# cfgs.update(cndata_cfgs)

import numpy
import torch
from commonlibs.common_tools import *
from pathlib import Path
GEN_SHELLS = Path('./EXP_CONFIG/Generated_Shells')


# print(cfgs, obb_cfgs)
def exe_cmd(cmd):
    print('#'*100)
    print('CMD: ', cmd)
    print('#'*100)
    os.system(cmd)

def generate_sheel(shell_path, cmd):
    pass

def parse_args():
    parser = argparse.ArgumentParser(
        description='parse exp cmd')
    parser.add_argument('model', help='model name')
    parser.add_argument('-d', help='devices id, 0~9')
    parser.add_argument('-c', help='control, list->list models, state->model的状态')
    parser.add_argument('-resume', help='latest -> latest or int -> epoch')
    parser.add_argument('-obb_to_hbb', help='将obb转换为hbb，并进行hbb的评估，通常与-m test结合使用')
    parser.add_argument('-r', default='det', help='设置runner')


    parser.add_argument('-load_results',
                        action='store_true',
                        help=' does not inference ,just evaluate, default=True')
    parser.add_argument('-port',default=29510,
                        help='结点端口'),
    parser.add_argument('-addr',default='127.0.0.10',
                        help='结点初始IP')


    parser.add_argument('-val_epoch', action='store_true')

    parser.add_argument('-m', help='mode, train or test', default='train')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.c:
        # 显示模型列表
        if args.c == 'list':
            show_dict(cfgs, 0)
        # 显示模型状态
        if args.c == 'state':
            cfg_states = {}
            for model_name, cfg in cfgs.items():
                cfg_state = dict(
                    exist=False,
                    train_state='',
                    test_state=[]
                )
                work_dir = cfg['work_dir']
                # 模型存在
                if os.path.exists(work_dir):
                    cfg_state['exist'] = True
                    # 训练状态
                    work_files = os.listdir(work_dir)
                    if os.path.exists(cfg['cp_file']):
                        cfg_state['train_state'] = 'Done'
                    else:
                        epoch_files = [f for f in work_files if f[0:5] == 'epoch'
                                       and f[-4:] == '.pth']
                        epoch_ints = [int(f[6:-4]) for f in epoch_files
                                      if f[6:-4].isdigit()]
                        if len(epoch_ints) > 0:
                            max_epoch_int = max(epoch_ints)
                            cfg_state['train_state'] = 'epoch_%d.pth' % max_epoch_int
                        else:
                            cfg_state['train_state'] = 'Not Training Yet'
                    # 测试状态
                    if os.path.exists(cfg['result']):
                        cfg_state['test_state'].append('Inference Done')
                    if 'dota_eval_results' in cfg.keys() \
                            and os.path.exists(cfg['dota_eval_results']):
                        cfg_state['test_state'].append('DOTA Evaluate Done')
                    if 'eval_results' in cfg.keys() \
                            and os.path.exists(cfg['eval_results']):
                        cfg_state['test_state'].append('Evaluate Done')
                    if 'submission_dir' in cfg.keys() \
                            and os.path.exists(cfg['submission_dir']):
                        cfg_state['test_state'].append('Submit OK')
                    cfg_states[model_name] = cfg_state
                # 模型不存在
                else:
                    cfg_state['train_state'] = 'Not exist'
                    cfg_states[model_name] = cfg_state
                    continue
            max_len_model_name = max([len(m) for m in cfg_states.keys()])
            print('=' * (max_len_model_name + 100))
            table_tmp = '%-{}s\t|%-{}s\t|%-{}s'.format(max_len_model_name
                                                       , 15, 25)

            print(table_tmp %
                  ('NAME', 'TRAIN_STATE', 'TEST_STATE'))

            for model_name, cfg_state in cfg_states.items():
                print(table_tmp %
                      (model_name, str(cfg_state['train_state']), str(cfg_state['test_state'])))
            print('=' * (max_len_model_name + 100))




    else:
        model = args.model
        if model not in cfgs.keys():
            raise Exception("%s not in cfg keys: %s" %(model, str(list(cfgs.keys()))
            ))

        cfg = cfgs[model]
        ##########################
        os.chdir('..')
        ##########################
        devs = [int(i) for i in args.d.split(',')]
        if len(devs) == 0:
            raise Exception('no deveices ')
        if args.m == 'train':
            if args.r == 'det':
                launch_cmd = \
                    'CUDA_VISIBLE_DEVICES=%s ' \
                    'PORT=%s ' \
                    'bash ./tools/my_dist_train.sh ' \
                    '%s %d --work-dir %s --cfg-options randomness.seed=2024' % \
                    (args.d,
                     str(args.port),
                     cfg['config'], len(devs), cfg['work_dir'])
            elif args.r == 'cls':
                launch_cmd = \
                    'CUDA_VISIBLE_DEVICES=%s ' \
                    'PORT=%s ' \
                    'bash ./tools/my_dist_train_mmcls.sh ' \
                    '%s %d --work-dir %s' % \
                    (args.d,
                     str(args.port),
                     cfg['config'], len(devs), cfg['work_dir'])

            elif args.r == 'few':
                launch_cmd = \
                    'CUDA_VISIBLE_DEVICES=%s ' \
                    'PORT=%s ' \
                    'bash ./tools/my_dist_train_mmfewshot.sh ' \
                    '%s %d --work-dir %s' % \
                    (args.d,
                     str(args.port),
                     cfg['config'], len(devs), cfg['work_dir'])
            elif args.r == 'resume':
                launch_cmd = \
                    'CUDA_VISIBLE_DEVICES=%s ' \
                    'PORT=%s ' \
                    'bash ./tools/my_dist_train.sh ' \
                    '%s %d --work-dir %s --resume' % \
                    (args.d,
                     str(args.port),
                     cfg['config'], len(devs), cfg['work_dir'])
        elif args.m == 'test':
            # launch_cmd = \
            #     'CUDA_VISIBLE_DEVICES=%s ' \
            #     'PORT=%s ' \
            #     'bash ./tools/my_dist_train.sh ' \
            #     '%s %d --work-dir %s --seed 42' % \
            #     (args.d,
            #      str(args.port),
            #      cfg['config'], len(devs), cfg['work_dir'])
            launch_cmd = 'CUDA_VISIBLE_DEVICES=%s python ./tools/test.py ' \
                  '%s %s ' \
                  '--format-only ' \
                  '--eval-options submission_dir=%s ' % \
                  (args.d,
                   cfg['config'],
                   cfg['cp_file'],
                   cfg['submission_dir'])
        else:
            raise Exception(args.m)

        cmd = launch_cmd
        print('#' * 100)
        print('Lunch: %s' % cmd)
        print('#' * 100)

        exe_cmd(cmd)
