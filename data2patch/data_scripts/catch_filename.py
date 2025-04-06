import os

# os.listdir()方法获取文件夹名字，返回数组
def getAllFiles(targetDir):
    listFiles = os.listdir(targetDir)
    return listFiles

files = getAllFiles(r"/home/zhao/dc/datasets/DIV2K/HR_patch")
files.sort()

# 写入list到txt文件中
with open(r"/home/zhao/dc/code/CAMixerSR-main/code_base/basicsr/data/meta_info/dc_DIV2K800sub_GT.txt",'w+',encoding='utf-8')as f:
    # 列表生成式
    f.writelines([str(i)+" (240,240,3)\n" for i in files])


