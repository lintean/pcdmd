# coding=UTF-8
from __future__ import print_function
import os
import re
from decimal import Decimal
import numpy as np
import argparse

def __get_last_line(filename):
    try:
        filesize = os.path.getsize(filename)
        if filesize == 0:
            return None
        else:
            with open(filename, 'rb') as fp:  # to use seek from end, must use mode 'rb'
                offset = -2  # initialize offset
                while -offset < filesize:  # offset cannot exceed file size
                    # read # offset chars from eof(represent by number '2')
                    fp.seek(offset, 2)
                    lines = fp.readlines()  # read from fp to eof
                    if len(lines) >= 2:  # if contains at least 2 lines
                        # then last line is totally included
                        return lines[-1]
                    else:
                        offset *= 2  # enlarge offset
                fp.seek(0)
                lines = fp.readlines()
                return lines[-1]
    except FileNotFoundError:
        logger.error(filename + ' not found!')
        return ""


# 假设要扫描指定文件夹下的文件，包含子文件夹，调用scan_files("/export/home/test/")
# 假设要扫描指定文件夹下的特定后缀的文件（比方jar包），包含子文件夹，调用scan_files("/export/home/test/", postfix=".jar")
# 假设要扫描指定文件夹下的特定前缀的文件（比方test_xxx.py）。包含子文件夹，调用scan_files("/export/home/test/", prefix="test_")
def scan_files(directory, prefix=None, postfix=None):
    files_list = []

    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root, special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root, special_file))
            else:
                files_list.append(os.path.join(root, special_file))

    return files_list

# 将元素中的数字转换为int后再排序
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

# 将元素中的字符串和数字分割开
def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def output_result(document):
    output = "result: \n"
    totle = 0
    file_list = scan_files(document, postfix=".log")
    file_list.sort(key=str2int)
    print(*file_list, sep="\n")

    for i in range(len(file_list)):
        filename = file_list[i]
        string = __get_last_line(filename).decode()
        output = output + string
        # 匹配小数部分，然后平均
        string = re.search("(-?\d+)(\.\d+)?", string).group()
        string = Decimal(string).quantize(Decimal('0.0000'))
        totle = totle + float(string)
    average = totle / len(file_list)
    output = output + "average: " + str(average)
    print(output)

if __name__ == "__main__":
    # 暂时先合并三个文件，后续再扩展
    parser = argparse.ArgumentParser(description="Get result")
    parser.add_argument("-d", "--document", type=str, default="./result/cm_textCNN_4cm/",
                        help="the orgin document need to get result")
    args = parser.parse_args()
    output_result(args.document)
