import matplotlib.pyplot as plt

from MY_LOGGERS.utils import logger_root, AutoLogger
from commonlibs.common_tools import *
import numpy as np
import cv2

# CLv2_IOU_H07_L01
# CLv2_Base
# CLv2_IOU_H07_L01_Encoder
# CLv2_IOU_H99_L00_Keep_H99_L00
# CLv2_Base_Neg_Pos
# CLv2_IOU_H80_L00_Keep_H80
logger_name = 'RHCv2_TP'
from MY_LOGGERS.utils import merge
logger_root = logger_root + '/' + logger_name
merge(logger_root, logger_name)
m_logger_pth = logger_root + '/' + logger_name + '.pkl'
results = pklload(m_logger_pth)
a = 0
fig_dir = logger_root + '/' + logger_name + '_vis'
mkdir(fig_dir)
interval = 10
for k, values in results.items():
    n_int = len(values) // interval
    means = []
    for i in range(n_int):
        int_value = values[i*interval: (i+1) * interval]
        int_value = [v[0].reshape(-1, 1) for v in int_value]
        int_value = np.concatenate(int_value)
        m = np.mean(int_value)
        v = np.var(int_value)
        means.append(m)
    print(means)
    plt.xlabel('n_Interval')
    plt.ylabel('%s' % k)
    plt.title('%s_Means' % k)
    plt.plot(range(n_int), means)

    fig_pth = fig_dir + '/%s_Int_%d' % (k, interval)
    plt.savefig(fig_pth)
    plt.close()
    print('Save: %s' % fig_pth)



