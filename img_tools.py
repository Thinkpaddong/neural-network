"""
一些图像处理工具
"""
import os
from PIL import Image
from os.path import join
import shutil
import matplotlib.pyplot as plt
from cfg import train_pic_folder,train_pic_folder2
import numpy as np

def get_bin_table(threshold=140):
    """
    获取灰度转二值的映射table
    :param threshold:
    :return:
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    return table

def sum_9_region(img, x, y):
    """
    9邻域框,以当前点为中心的田字框,黑点个数,作为移除一些孤立的点的判断依据
    :param img: Image
    :param x:
    :param y:
    :return:
    """
    cur_pixel = img.getpixel((x, y))  # 当前像素点的值
    width = img.width
    height = img.height

    if cur_pixel == 1:  # 如果当前点为白色区域,则不统计邻域值
        return 0

    if y == 0:  # 第一行
        if x == 0:  # 左上顶点,4邻域
            # 中心点旁边3个点
            sum = cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 4 - sum
        elif x == width - 1:  # 右上顶点
            sum = cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1))

            return 4 - sum
        else:  # 最上非顶点,6邻域
            sum = img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 6 - sum
    elif y == height - 1:  # 最下面一行
        if x == 0:  # 左下顶点
            # 中心点旁边3个点
            sum = cur_pixel \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x, y - 1))
            return 4 - sum
        elif x == width - 1:  # 右下顶点
            sum = cur_pixel \
                  + img.getpixel((x, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y - 1))

            return 4 - sum
        else:  # 最下非顶点,6邻域
            sum = cur_pixel \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x, y - 1)) \
                  + img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x + 1, y - 1))
            return 6 - sum
    else:  # y不在边界
        if x == 0:  # 左边非顶点
            sum = img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))

            return 6 - sum
        elif x == width - 1:  # 右边非顶点
            # print('%s,%s' % (x, y))
            sum = img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1))

            return 6 - sum
        else:  # 具备9领域条件的
            sum = img.getpixel((x - 1, y - 1)) \
                  + img.getpixel((x - 1, y)) \
                  + img.getpixel((x - 1, y + 1)) \
                  + img.getpixel((x, y - 1)) \
                  + cur_pixel \
                  + img.getpixel((x, y + 1)) \
                  + img.getpixel((x + 1, y - 1)) \
                  + img.getpixel((x + 1, y)) \
                  + img.getpixel((x + 1, y + 1))
            return 9 - sum


def remove_noise_pixel(img, noise_point_list):
    """
    根据噪点的位置信息，消除二值图片的黑点噪声
    :type img:Image
    :param img:
    :param noise_point_list:
    :return:
    """
    for item in noise_point_list:
        img.putpixel((item[0], item[1]), 1)

def resize(img):
    '''
    裁剪：传入一个元组作为参数
    元组里的元素分别是：（距离图片左边界距离x， 距离图片上边界距离y，距离图片左边界距离+裁剪框宽度x+w，距离图片上边界距离+裁剪框高度y+h）
    '''
    # 截取图片和这个案例一样
    x =5
    y =3
    w =48
    h =12
    region = img.crop((x, y, x + w, y + h))
    return region

def resize1(img):
    '''
    裁剪：传入一个元组作为参数
    元组里的元素分别是：（距离图片左边界距离x， 距离图片上边界距离y，距离图片左边界距离+裁剪框宽度x+w，距离图片上边界距离+裁剪框高度y+h）
    '''
    # 截取图片和这个案例一样
    x =7
    y =3
    w =48
    h =13
    region = img.crop((x, y, x + w, y + h))
    return region

def get_clear_bin_image(image):
    """
    进行修改
    获取干净的二值化的图片。
    图像的预处理：
    1. 先转化为灰度
    2. 再二值化
    3. 然后清除噪点
    :type img:Image
    :return:
    """
    imgry = image.convert('L')  # 转化为灰度图
    imgry.save(join(r'C:\Users\ThinkPad_dong\Desktop',"1.jpg"))
    imgry1=resize1(imgry)        # 调整大小
    imgry1.save(join(r'C:\Users\ThinkPad_dong\Desktop',"2.jpg"))
    table = get_bin_table()
    out = imgry1.point(table, '1')  # 变成二值图片:0表示黑色,1表示白色
    out.save(join(r'C:\Users\ThinkPad_dong\Desktop',"3.jpg"))
    noise_point_list = []  # 通过算法找出噪声点,第一步比较严格,可能会有些误删除的噪点
    for x in range(out.width):
        for y in range(out.height):
            res_9 = sum_9_region(out, x, y)
            if (0 < res_9 < 3) and out.getpixel((x, y)) == 0:  # 找到孤立点
                pos = (x, y)  #
                noise_point_list.append(pos)
    remove_noise_pixel(out, noise_point_list)
    out.save(join(r'C:\Users\ThinkPad_dong\Desktop', "4.jpg"))
    return out


def get_crop_imgs(img):
    """
    按照图片的特点,进行切割,这个要根据具体的验证码来进行工作. # 见本例验证图的结构原理图
    分割图片是传统机器学习来识别验证码的重难点，如果这一步顺利的话，则多位验证码的问题可以转化为1位验证字符的识别问题
    :param img:
    :return:
    """
    child_img_list = []
    for i in range(4):
        x = i * (9 + 4)  # 见原理图
        y = 0
        child_img = img.crop((x, y, x + 9, y + 13)) #本实验进行修改的
        child_img_list.append(child_img)
    return child_img_list

def get_crop_imgs1(img):
    """
    按照图片的特点,进行切割,这个要根据具体的验证码来进行工作. # 见本例验证图的结构原理图
    分割图片是传统机器学习来识别验证码的重难点，如果这一步顺利的话，则多位验证码的问题可以转化为1位验证字符的识别问题
    :param img:
    :return:
    """
    child_img_list = []
    for i in range(4):
        x = i * (9 + 4)  # 见原理图
        y = 0
        child_img = img.crop((x, y, x + 9, y + 13)) #本实验进行修改的
        child_img_list.append(child_img)
    return child_img_list

def cut_text(text):
    textArr=[]
    for x in text:
        textArr.append(x)
    return textArr      #返回一个列表每个字符


#对于好多照片的操作
def save_crop_imgs(bin_clear_image_path, child_img_list,cut_pic_folder):
    """
    输入：整个干净的二化图片
    输出：每张切成4版后的图片集
    保存切割的图片

    例如： A.png ---> A-1.png,A-2.png,... A-4.png 并保存，这个保存后需要去做label标记的
    :param bin_clear_image_path: xxxx/xxxxx/xxxxx.png 主要是用来提取切割的子图保存的文件名称
    :param child_img_list:
    :return:
    """
    full_file_name = os.path.basename(bin_clear_image_path)  # 文件名称
    full_file_name_split = full_file_name.split('.')
    file_name = full_file_name_split[0]                      #一张图片名
    file_name_list=cut_text(file_name)
    i = 0
    for child_img in child_img_list:
        cut_img_file_name = file_name + '-' + ("%s" % i)+'_'+("%s.jpg" % file_name_list[i])
        print(cut_img_file_name)
        child_img.save(join(cut_pic_folder, cut_img_file_name))
        i += 1


# 训练素材准备：文件目录下面的图片的批量操作
def batch_get_all_bin_clear(origin_pic_folder,bin_clear_folder):
    """
    训练素材准备。
    批量操作：获取所有去噪声的二值图片
    :return:
    """

    file_list = os.listdir(origin_pic_folder)
    for file_name in file_list:
        file_full_path = os.path.join(origin_pic_folder, file_name)
        image = Image.open(file_full_path)
        #这部分程序是自己写的
        out=get_clear_bin_image(image)
        file_full_path1=os.path.join(bin_clear_folder, file_name)
        out.save(file_full_path1)

def batch_cut_images(bin_clear_folder):
    """
    训练素材准备。
    批量操作：分割切除所有 "二值 -> 除噪声" 之后的图片，变成所有的单字符的图片。然后保存到相应的目录，方便打标签
    """

    file_list = os.listdir(bin_clear_folder)
    for file_name in file_list:
        bin_clear_img_path = os.path.join(bin_clear_folder, file_name)
        img = Image.open(bin_clear_img_path)

        child_img_list = get_crop_imgs(img)
        save_crop_imgs(bin_clear_img_path, child_img_list)  # 将切割的图进行保存，后面打标签时要用


def rang(path,path1):
    #列出文档
    file_list = os.listdir(path)
    id=[]                                                #存储文件名中的id
    for i in range(len(file_list)):
        id.append(file_list[i].split('_')[1])
    id=set(id)                                           #取出唯一的id值，用于建立文件夹
    sort_folder_number = list(id)                        #把集合id转化为列表类型
    for number in sort_folder_number:
        new_folder_path = os.path.join(path1,'%s'%number)#新文件夹路径
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)                 #提取出文档名称内的id，并根据id决定将发往指定文件夹
    for i in range(len(file_list)):
        old_file_path = os.path.join(path,file_list[i])
        fid=file_list[i].split('_')[1]
        new_file_path = os.path.join(path1,'%s'%(fid),file_list[i])
        shutil.move(old_file_path,new_file_path)
if __name__=="__main__":

    pass
