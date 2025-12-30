import os


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print('Make dir: %s' % dir_path)

