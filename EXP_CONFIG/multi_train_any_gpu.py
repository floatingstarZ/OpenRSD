import os
# print(os.path.abspath(os.path.dirname(__file__))) #输出绝对路径
# 获取当前文件的绝对路径
Project_root = os.path.abspath(os.path.dirname(__file__))
# 去除当前文件的其他路径，保留项目根目录
backback_id = Project_root.find('EXP_CONFIG')
Project_root = Project_root[:backback_id-1]
import sys
sys.path.append(Project_root)

import multiprocessing
import time
from pynvml import *
import numpy as np


def has_attr(s, word_list):
    for word in word_list:
        if word in s:
            return True
    return False

def modify_pycmd(cmd, gpus=None, ctrl=None):
    """

    :param cmd: python py_cmd.py model_name -m train
    ，不包含-d的信息
    :param gpus:
    :return:
    """
    assert gpus
    for id in gpus:
        ctrl[id] = 1
    gpu_str = '%d' % gpus[0]
    if len(gpus) > 1:
        for i in range(1, len(gpus)):
            gpu_str +=',%d' % gpus[i]
    cmd = cmd.strip()
    cmd += ' -d ' + gpu_str
    print('*'*100)
    print(cmd)
    print('*'*100)

    os.system(cmd)

    for id in gpus:
        ctrl[id] = 0

def get_gpu_infos(deviceCount):
    infos = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        infos.append(dict(total=info.total / 1024 ** 3,
                          free=info.free / 1024 ** 3,
                          used=info.used / 1024 ** 3))
    return infos

def get_available_gpu_ids(deviceCount, max_used=10, max_used_gpu=8):
    infos = get_gpu_infos(deviceCount)
    av_ids = []
    for id, info in enumerate(infos):
        if info['used'] < max_used:
            av_ids.append(id)
        # if len(av_ids) >= max_used_gpu:
        #     break
    return av_ids
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='parse exp cmd')
    parser.add_argument('-k', help='has attribute keys', nargs='+', default=[])
    parser.add_argument('-c', help='start cmd count', default=10)
    parser.add_argument('-r', default='det', help='设置runner')
    parser.add_argument('-d', default=None, help='devices id, 0~9')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 自动化测试命令
    from EXP_CONFIG.CONFIGS.all_config import all_cfgs
    args = parse_args()
    white_list = args.k
    print('%'*100, '\n white_list:', white_list)

    cfgs = all_cfgs
    gpu_nums = []
    cmds = []
    # 筛选出已经训练好的模型，并形成commands
    cmd_count = int(args.c)
    for model_name, cfg in cfgs.items():
        work_dir = cfg['work_dir']
        if os.path.exists(cfg['cp_file']):
            print('Pass')
            continue
        if len(white_list) != 0 and not has_attr(model_name, white_list):
            continue
        gpu_nums.append(cfg['gpu'])
        if args.r == 'det':
            cmds.append('python py_cmd.py %s -m train -val_epoch -port %d -addr %s'
                        % (model_name, 29500 +
                           cmd_count,
                           '127.0.0.' + str(cmd_count)))
        elif args.r == 'cls':
            cmds.append('python py_cmd.py %s -r cls -m train -val_epoch -port %d -addr %s'
                        % (model_name, 29500 +
                           cmd_count,
                           '127.0.0.' + str(cmd_count)))
        elif args.r == 'few':
            cmds.append('python py_cmd.py %s -r few -m train -val_epoch -port %d -addr %s'
                        % (model_name, 29500 +
                           cmd_count,
                           '127.0.0.' + str(cmd_count)))
        elif args.r == 'resume':
            cmds.append('python py_cmd.py %s -r resume -m train -val_epoch -port %d -addr %s'
                        % (model_name, 29500 +
                           cmd_count,
                           '127.0.0.' + str(cmd_count)))
        cmd_count += 4

    ##############
    # cmds = ['python DataParallel_py_cmd.py JiNan_ms_fsaf_12_4 -m train',
    #         'python DataParallel_py_cmd.py JiNan_fusion_fsaf_12_4 -m train',
    #         'python DataParallel_py_cmd.py JiNan_ms_grid_rcnn_12_4 -m train']
    # 筛选出已经训练好的模型，并形成commands
    for cmd in cmds:
        print(cmd)
    print(gpu_nums)
    # 最大同时存在的task数目
    MAX_TASK = 10

    nvmlInit()  # 初始化
    print("Driver: ", nvmlSystemGetDriverVersion())  # 显示驱动信息
    # >>> Driver: 384.xxx

    # 查看设备
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("GPU", i, ":", nvmlDeviceGetName(handle))
    # 允许使用的GPU
    allowed_gpus = [0,1,2,3,4,5,6,7,8,9]
    if args.d is not None:
        devs = [int(i) for i in args.d.split(',')]
        allowed_gpus = devs

    # 已经使用的GPU列表(已经分配的)
    used_gpu = multiprocessing.Array("i", np.zeros(deviceCount, dtype=np.int32).tolist())
    # used_gpu = multiprocessing.Array("i", np.zeros(deviceCount, dtype=np.int32).tolist())
    # running_task =  multiprocessing.Array("i", np.zeros(deviceCount, dtype=np.int32).tolist())

    # 在这里设置队伍列表
    # tasks = [
    #     dict(fun=modify_pycmd, kwargs=dict(cmd='python py_cmd.py DIOR_libra_faster_rcnn_full -m test',ctrl=used_gpu), used_gpu=1),
    #     dict(fun=modify_pycmd, kwargs=dict(cmd='python py_cmd.py DIOR_pafpn_full -m test',ctrl=used_gpu), used_gpu=1),
    # ]
    tasks = [dict(fun=modify_pycmd,
                  kwargs=dict(cmd=cmd, ctrl=used_gpu), used_gpu=gpu)
             for gpu, cmd in zip(gpu_nums, cmds)]
    # tasks = [dict(fun=modify_pycmd, kwargs=dict(cmd='python py_cmd.py DOTA_obb_tv_faster_rcnn_RoITrans -m train',
    #                                             ctrl=used_gpu), used_gpu=2)
    #          ]
    # tasks = [dict(fun=modify_pycmd, kwargs=dict(cmd='python py_cmd.py COCO_retinanet -d 3,4 -m train',
    #                                             ctrl=used_gpu), used_gpu=2)
    #          ]
    task_id = 0
    p_list = []

    # 开始运行
    while(True):
        if task_id == len(tasks):
            break
        task = tasks[task_id]
        task_used = task['used_gpu']

        # 可以使用的 和 已经使用的集合 的 并集
        available_gpu_ids = get_available_gpu_ids(deviceCount, max_used=0.5)
        used_gpu_ids = np.arange(deviceCount)[np.array(list(used_gpu), dtype=np.bool_)].tolist()
        available_gpu_ids = list(set(available_gpu_ids) - set(used_gpu_ids))
        not_allowed_gpus = set(list(range(deviceCount))) - set(allowed_gpus)
        available_gpu_ids = list(set(available_gpu_ids) - set(not_allowed_gpus))


        # 如果没有足够GPU可以使用，则继续检查
        if len(available_gpu_ids) < task_used or \
                np.sum(np.array(list(used_gpu))) >= MAX_TASK:
            print('Wait')
            time.sleep(3)
            continue

        # 设置GPU数量
        task['kwargs']['gpus'] = available_gpu_ids[0:task_used]
        print("available_gpu_ids: %s, used: %s" %
              (available_gpu_ids, task['kwargs']['gpus']))

        # 启动进程
        p = multiprocessing.Process(target=task['fun'], kwargs=task['kwargs'])
        p_list.append(p)
        p_list[-1].start()
        # 等待一秒，让进程设置一下used_gpu
        time.sleep(3)


        task_id += 1
    for p in p_list:
        p.join()

    print('Done')

    nvmlShutdown()


