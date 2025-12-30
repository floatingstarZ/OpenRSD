from commonlibs.common_tools import *
from commonlibs.transform_tools.type_transform import *
import os
import sys
dict()
class CacheDict:
    """
    使用字典，如果超出cache_size或者超过cache_num则进行缓存，缓存到cache_folder中。
    """
    dim=1024*1024*1024 # Gb
    inf=1e16
    def __init__(self, cache_foler='./', cache_size=2, cache_num=10, over_write_folder = True,
                 msg=False):
        """
        :param cache_foler: 缓存文件夹
        :param cache_size: 缓存最大内存大小，以CacheDict.dim作为单位
        :param cache_num: 缓存最大data个数
        :param over_write_folder: 是否复用cache folder
        :param over_write_folder: 是否显示中间信息
        """
        if over_write_folder:
            mkdir(cache_foler)
        else:
            count = 1
            org_cache_foler = cache_foler
            while True:
                if os.path.exists(cache_foler):
                    cache_foler = org_cache_foler + '_%d' % count
                    count += 1
                else:
                    break
            os.mkdir(cache_foler)
            print('Make dir: %s' % cache_foler)
        self.cache_foler = cache_foler
        self.cache_size = cache_size * CacheDict.dim
        self.cache_num = cache_num
        self.content = {}
        self.cache_used = 0
        self.msg = msg

    def _put_into_disk(self, file_name, data):
        """
        将data放入缓存文件夹中
        :param file_name:
        :param data:
        :return:
        """
        data = all_to_numpy(data)
        pklsave(data, self.cache_foler + '/' + file_name, msg=self.msg)

    def _get_from_disk(self, file_name):
        return pklload(self.cache_foler + '/' + file_name, msg=self.msg)

    def put(self, name, data):
        """
        装入data
        :param name:
        :param data:
        :return:
        """
        data_size = sys.getsizeof(data)
        self.content[name] = dict(
            file_name=name + '.pkl',
            in_mem=True
        )
        if data_size + self.cache_used > self.cache_size \
            or len(self.content) > self.cache_num:
            self.content[name]['in_mem'] = False
            self.content[name]['data'] = None
            self._put_into_disk(self.content[name]['file_name'], data)
        else:
            self.content[name]['data'] = data
            self.cache_used += data_size

    def get(self, name):
        """
        取出data，并返回状态（是否在内存当中）
        :param name:
        :return: in_mem, data
        """
        if name not in self.content.keys():
            raise Exception('KeyError: %s' % str(name))
        if self.content[name]['in_mem']:
            return True, self.content[name]['data']
        else:
            data = self._get_from_disk(self.content[name]['file_name'])
            return False, data

    def items(self):
        keys = self.content.keys()
        for name in keys:
            status, data = self.get(name)
            yield name, data

    def keys(self):
        for k in self.content.keys():
            yield k

    def values(self):
        for name in self.content.keys():
            status, data = self.get(name)
            yield data

    def save_all(self):
        """
        save all data into cache folder
        :return
        """
        for name, value in self.content.items():
            if value['in_mem']:
                self._put_into_disk(value['file_name'], value['data'])
                # data_size = sys.getsizeof(value['data'])
                value['in_mem'] = False
                value['data'] = None
        self.cache_used = 0

    def load_all(self):
        """
        从 folder 中载入数据，覆盖现有的数据。

        :return: 超过上限部分的大小
        """
        cache_files = os.listdir(self.cache_foler)
        in_mem_count = self.count_in_mem()
        for cf in cache_files:
            name = os.path.splitext(cf)[0]
            self.content[name] = dict(
                file_name=name + '.pkl',
                in_mem=True,
                data=None
            )
            if self.cache_used > self.cache_size \
                or in_mem_count >= self.cache_num:
                self.content[name]['in_mem'] = False
                continue
            data = pklload(self.cache_foler + '/' + cf, msg=self.msg)
            self.cache_used += sys.getsizeof(data)
            self.content[name]['data'] = data
            in_mem_count += 1
        return max(self.cache_used - self.cache_size, 0)

    def clear(self):
        self.content = {}
        self.cache_used = 0

    def count_in_mem(self):
        """
        计算在内存中的个数
        :return:
        """
        count = 0
        for k, v in self.content.items():
            if v['in_mem']:
                count += 1
        return count


    def __str__(self):
        s = ''
        for k, v in self.content.items():
            s += '%s: %s\n' % (str(k), str(v))
        return s

    def __getitem__(self, name):
        status, data = self.get(name)
        return data

    def __setitem__(self, key, value):
        self.put(key, value)

    def __len__(self):
        return len(self.content)




if __name__ == '__main__':
    c = CacheDict('./test_folder/A', cache_size=0, cache_num=3)
    c.put('a', 1)
    c.put('b', 2)
    c.put('c', 3)
    c.put('d', 4)
    print(c)
    print(c['a'])
    print(c['c'])
    print(c.get('d'))
    c.save_all()
    print(c)
    c.put('e', 5)
    print(c)
    c.load_all()
    print(c)
    print(c.get('e'))
    c.put('f', np.array([1,2,3]))
    print(list(c.items()))
    print(list(c.keys()))
    print(list(c.values()))
    print('h' in c.keys())
    c['a'] = 100
    c['h'] = 50
    print(c)
    print(list(c.values()))

    print(c.get('j'))




