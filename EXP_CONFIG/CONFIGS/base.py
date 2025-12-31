# Project_root = '/home/huangziyue/Projects/RS_BasicDetection'
# Project_root = '/home/huangziyue/Projects/RS_BasicDetection'
# print('##'*1000)
import os
# print(os.path.abspath(os.path.dirname(__file__))) #输出绝对路径
# 获取当前文件的绝对路径
Project_root = os.path.abspath(os.path.dirname(__file__))
# 去除当前文件的其他路径，保留项目根目录
backback_id = Project_root.find('EXP_CONFIG')
Project_root = Project_root[:backback_id-1]
print('Project root: ', Project_root)
