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
fig_dir = logger_root + '/OD024_compare_vis'
mkdir(fig_dir)

from MY_LOGGERS.utils import merge
# log_names = ['RHC', 'RHC_Base', 'RHC_D_256',
#              'RHC_M_095', 'RHC_OnlyPos_use_pt', 'RHC_use_pt']
# base_result = pklload(logger_root + '/RHC/RHC.pkl')

# log_names = ['RHC_Base',
#              'RHC_use_pt_FP',
#              'RHC_use_pt_FN',
#              'RHC_OnlyPos_use_pt_FP',
#              'RHC_OnlyPos_use_pt_FN']
# log_names = ['RHCv2_Base',
#              'RHCv2_TP',
#              'RHCv2_FP',
#              'RHCv2_FN',
#              'RHCv2_FN_FP']
log_names = ['OD024_Base',
             'OD024_Q_All_P_All',
             'OD024_Q_All_P_KNN',
             'OD024_Q_All_P_IOU']
result_dict = {}
for name in log_names:
    logger_name = name

    l_dir = logger_root + '/' + logger_name
    merge(l_dir, logger_name)
    m_logger_pth = l_dir + '/' + logger_name + '.pkl'
    results = pklload(m_logger_pth)
    result_dict[m_logger_pth] = results

base_result = pklload(logger_root + '/OD024_Base/OD024_Base.pkl')
keys = list(base_result.keys())
print(keys)
interval = 20

for k in keys:
    plt.title('%s_Means' % k)
    plt.xlabel('n_Interval')
    plt.ylabel('%s' % k)
    for name in log_names:
        m_logger_pth = logger_root + '/%s/%s.pkl' % (name, name)
        results = result_dict[m_logger_pth]

        if k not in results.keys():
            continue
        values = results[k]
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

        plt.plot(range(n_int), means, label=name)

    fig_pth = fig_dir + '/%s_Int_%d' % (k, interval)
    plt.legend()
    plt.savefig(fig_pth)
    plt.close()
    print('Save: %s' % fig_pth)



