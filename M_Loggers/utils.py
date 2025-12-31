import os
from pathlib import Path
from commonlibs.common_tools import *
from commonlibs.io_tools.cachedict import CacheDict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import OrderedDict

import time
logger_root = './MMROTATE_LOGGERS'
mkdir(logger_root)
class FlashLogger:
    """
    快速存储的logger
    new->建立新key，存储之前的旧key数据，并只保留旧的key值
    add->在新key中添加数据
    """
    def __init__(self,
                 root,
                 name,
                 msg=False,
                 cache_limit=10000000,
                 auto_save=True,
                 over_write=True,
                 mode='wt+'):
        l_dir = root + '/' + name
        cache_folder = l_dir

        self.data = dict()
        self.mode = mode
        self.iter = 0
        self.last_key = None
        self.msg = msg
        self.cache_limit = cache_limit
        self.auto_save = auto_save
        # write
        if mode in ['w', 'wt+']:
            # make new folder
            if mode == 'w':
                for i in range(1000):
                    cache_folder = l_dir + '_%d' % i
                    if not os.path.exists(cache_folder):
                        break
            self.cache_folder = cache_folder
            mkdir(self.cache_folder)
        # read only
        elif mode == 'r':
            pass

    def new(self, key):
        """
        :param key: new key
        :return: add a new key
        """
        # save data before new(the last key)
        if len(self.data) >= self.cache_limit:
            raise Exception('Error!! Cache Full')
        if self.auto_save:
            if self.last_key:
                self.save()
        # create new key
        self.last_key = key
        if key in self.data.keys():
            print('WARNING!!! KEY: %s EXISTED IN CACHE DICT' % key)
        self.data[key] = dict()

    def new_iter(self):
        self.new(self.iter)
        self.iter += 1

    def add(self, value_name, value):
        """
        add data to self.last_key
        :param value_name:
        :param value:
        :return:
        """
        assert self.last_key
        self.data[self.last_key][value_name] = value

    def add_to_list(self, name, value):
        # init list
        if name not in self.data[self.last_key].keys():
            self.add(name, [])
        self.data[self.last_key][name].append(value)

    def save(self):
        pklsave(self.data[self.last_key],
                self.cache_folder + '/' + str(self.last_key) + '.pkl',
                msg=self.msg)
        self.data[self.last_key] = dict()

    def load(self):
        for f in os.listdir(self.cache_folder):
            data = pklload(self.cache_folder + '/' + f,
                           msg=self.msg)
            key = Path(f).stem
            self.data[key] = data
        return self.data

class AutoLogger:
    """
    快速存储的logger
    new->建立新key，存储之前的旧key数据，并只保留旧的key值
    add->在新key中添加数据
    """
    def __init__(self,
                 root,
                 name,
                 msg=False,
                 cache_limit=10000000,
                 timestamp=True,
                 mode='w'):
        if timestamp:
            timestamp = '_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
        else:
            timestamp = ''
        self.cache_folder = root + '/%s' % name
        self.name = name

        self.data = dict()
        self.mode = mode
        self.iter = 0
        self.last_key = None
        self.msg = msg
        self.cache_limit = cache_limit
        # write
        if mode == 'w':
            # make new folder
            self.cache_folder += timestamp
            mkdir(self.cache_folder)
        # read only
        elif mode == 'r':
            if not os.path.exists(self.cache_folder):
                raise Exception('CacheFolder of logger: %s does not exist'
                                % self.cache_folder)

    def new(self):
        """
        :return: add a new key
        """
        key = str(self.iter).zfill(8)
        # save data before new(the last key)
        if len(self.data) >= self.cache_limit:
            raise Exception('Error!! Cache Full')
        if self.last_key:
            self.save()
            self.data[self.last_key] = dict()
        # create new key
        self.last_key = key
        if key in self.data.keys():
            print('WARNING!!! KEY: %s EXISTED IN CACHE DICT' % key)
        self.data[key] = dict()
        self.iter += 1

    def add_to_list(self, name, value):
        # init list
        if name not in self.data[self.last_key].keys():
            assert self.last_key
            self.data[self.last_key][name] = []
        self.data[self.last_key][name].append(value)

    def __setitem__(self, key, value):
        self.add_to_list(key, value)

    def save(self):
        pklsave(self.data[self.last_key],
                self.cache_folder + '/' + str(self.last_key) + '.pkl',
                msg=self.msg)

    def load(self):
        files = sorted(os.listdir(self.cache_folder))
        for f in files:
            data = pklload(self.cache_folder + '/' + f,
                           msg=self.msg)
            key = Path(f).stem
            self.data[key] = data
        return self.data

def to_array(x):
    if type(x) == torch.Tensor:
        return x.detach().cpu().numpy()
    elif type(x) == np.array:
        return x
    elif type(x) == list:
        return [to_array(i) for i in x]
    elif type(x) == tuple:
        return [to_array(i) for i in x]
    elif type(x) == dict:
        return {k: to_array(v) for k, v in x.items()}
    elif type(x) == OrderedDict:
        d = OrderedDict()
        for k, v in x.items():
            d[k] = to_array(v)
        return d
    else:
        return x


def merge(logger_dir, logger_name):
    logger = AutoLogger(logger_dir, logger_name, mode='r')
    data = logger.load()
    data = list(data.values())[:-1]
    merged_data = {k: []for k in data[0].keys()}
    for d in data:
        for k, v in d.items():
            merged_data[k].append(v)
    merged_pth = logger_dir + '/%s.pkl' % logger_name
    pklsave(merged_data, merged_pth)

# def plot(x):
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#
#     fig_name = result_name + '_PR_Curves_ignore_matched'
#     plt.title(fig_name)
#     fig_pth = fig_root + '/' + fig_name + '.png'
#     plt.savefig(fig_pth)
#     print('Save: %s' % fig_pth)
#     plt.close()
