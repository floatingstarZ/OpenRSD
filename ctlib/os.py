import os
import json
import pickle as pkl
import traceback
import cv2
import pandas as pd

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print('Make dir: %s' % dir_path)

def pklsave(obj, file_path, msg=True):
    with open(file_path, 'wb+') as f:
        pkl.dump(obj, f)
        if msg:
            print('SAVE OBJ: %s' % file_path)

def jsonsave(obj, file_path, msg=True):
    with open(file_path, 'wt+') as f:
        json.dump(obj, f)
        if msg:
            print('SAVE JSON: %s' % file_path)

def pklload(file_path, msg=True):
    with open(file_path, 'rb') as f:
        if msg:
            print('LOAD OBJ: %s' % file_path)
        try:
            return pkl.load(f)
        except Exception:
            traceback.print_exc()

def jsonload(file_path, msg=True):
    with open(file_path, 'r') as f:
        if msg:
            print('LOAD OBJ: %s' % file_path)
        try:
            return json.load(f)
        except EOFError:
            print('EOF Error %s' % file_path)

def excel_save(dict_data, file_path, msg=True, T=True):
    writer = pd.ExcelWriter(file_path, engine='openpyxl')  # 创建数据存放路径
    df = pd.DataFrame(dict_data)
    if T:
        df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)  # 转置
    else:
        df2 = df
    if msg:
        print(df)
    df2.to_excel(writer)
    writer.save()  # 文件保存
    writer.close()  # 文件关闭
    if msg:
        print('Save: %s' % file_path)

################ Image Save
def img_save(img, path, with_text=True):
    cv2.imwrite(path, img)
    if with_text:
        print('Save: %s' % path)




