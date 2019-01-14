import numpy as np
import os
from PIL import Image
from cfg import test_set_folder,train_pic_folder,home_root,train_pic_folder2,test_set_folder2
from os.path import join
from img_tools import get_clear_bin_image,get_crop_imgs,cut_text
np.set_printoptions(threshold=np.inf)

def getFileArr(dir,file_name_npy):
    '''
    
    :param dir: 生成数据集图片的路径
    :return: 返回生成的数据集
    '''
    result_arr=[]
    label_list=[]
    map={}
    map_file_result={}
    map_file_label={}
    map_new={}
    count_label=0
    count=0
    file_all_name=[]
    file_list = os.listdir(dir)
    for file_name in file_list:
        file_full_path = os.path.join(dir, file_name)
        image = Image.open(file_full_path)
        img=get_clear_bin_image(image)
        file_name_list=cut_text(file_name.split('.')[0])
        child_img_list = get_crop_imgs(img)

        for i in range(4):
            label=file_name_list[i]
            file=file_name.split('.')[0]+'-'+str(i)+'_'+file_name_list[i]
            file_all_name.append(file)
            map[file]=label
            if label not in label_list:
                label_list.append(label)
                map_new[label]=label
                count_label=count_label+1
            result=np.array([])
            img_arr=np.array(feature(np.asarray(child_img_list[i])))
            result=np.concatenate((result,img_arr))
            map_file_result[file]=result
            result_arr.append(result)
            count=count+1
    for file in file_all_name:
        map_file_label[file]=map_new[map[file]]
    ret_arr=[]
    for file in file_all_name:
        each_list=[]
        result=map_file_result[file]
        label=map_file_label[file]
        each_list.append(result)
        each_list.append(label)
        ret_arr.append(each_list)
    np.save(join(home_root,file_name_npy), ret_arr)
    return ret_arr
def load_data(train_dir,test_dir):
    train_data=np.load(train_dir)
    test_data=np.load(test_dir)
    X_train,y_train=train_data[:,0],train_data[:,1]
    X_test,y_test=test_data[:,0],test_data[:,1]
    X_train_list=[]
    X_test_list=[]
    label_i=0
    for i in X_train:
        j=i.tolist()
        j.append(int(y_train[label_i]))
        label_i+=1
        X_train_list.append(j)
    label_i=0
    for i in X_test:
        j=i.tolist()
        j.append(int(y_test[label_i]))
        label_i+=1
        X_test_list.append(j)
    return (X_train_list,X_test_list)

def feature(A):
    midx = int(A.shape[1] / 2) + 1
    midy = int(A.shape[0] / 2) + 1
    feature1 = A[0:midy, 0:midx].mean()
    feature2 = A[midy:A.shape[0], 0:midx].mean()
    feature3 = A[0:midy, midx:A.shape[1]].mean()
    feature4 = A[midy:A.shape[0], midx:A.shape[1]].mean()
    feature5 = A[midy - 1:midy + 4, midx - 1:midx + 4].mean()
    AF = [feature1, feature2, feature3, feature4, feature5]
    return AF

if __name__=="__main__":
    getFileArr(train_pic_folder2,'train_data.npy')
    getFileArr(test_set_folder2, 'test_data.npy')
    (X_train, X_test)=load_data(join(home_root,'train_data.npy'),join(home_root, 'test_data.npy'))
    pass