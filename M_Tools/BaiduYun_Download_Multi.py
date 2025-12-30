from bypy import ByPy
bp = ByPy()
print(bp.list())
print('((((((((((((((((((((((((')
"""
  r"Dataset-OPT2017-JPEGImages",
  "/extra_data/space2/huangziyue/Dataset-OPT2017-JPEGImages"

  r"FAIR1M2.0",
  "/extra_data/space2/huangziyue/FAIR1M2.0"
百度云文件夹路径：./我的应用数据/bypy/... 
"""
bp.download(
  r"FAIR1M2.0",
  "/data/space2/huangziyue/FAIR1M2.0"
)
print(bp.list())

# bp.upload(
#   "/extra_data/space2/huangziyue/HRRSD_800_0.tar.gz",
#   r"HRRSD_800_0.tar.gz",
# )
# print(bp.list())






