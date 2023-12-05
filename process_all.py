import os
import cv2

def process_all_train_data(input_path1,input_path2,output_path):
    '''
    处理所有训练图片

    Args:
        input_path1: the file_path where store all canny edge pictures
        input_path2: the file_path where store all HED edge pictures
        output_path: the train_data path
    
    Author: Bi Xiaoyang
    '''

    files_path1 = []
    files_path2 = []

    for label in sorted(os.listdir(input_path1)): #label：来源哪个数据集
        for fname in os.listdir(os.path.join(input_path1, label)):
            files_path1.append(os.path.join(input_path1, label, fname)) #图片的文件名

    for label in sorted(os.listdir(input_path2)): #label：来源哪个数据集
        for fname in os.listdir(os.path.join(input_path2, label)):
            files_path2.append(os.path.join(input_path2, label, fname)) #图片的文件名

    assert(len(files_path1)==len(files_path2))

    for i in range(len(files_path1)):
        file1 = files_path1[i]
        file2 = files_path2[i]
        print(file1, file2)
        name1 = file1.split("\\")
        label = name1[-2] #获取数据集名，如met-1
        fname = name1[-1] #获取文件名，如met_0.jpg
        arguments_strOut = os.path.join(output_path, label, fname)

        pic1 = cv2.imread(file1)
        pic1 = cv2.resize(pic1, (512, 512))
        pic2 = cv2.imread(file2)
        pic2 = cv2.resize(pic2, (512, 512))
        # print(pic1)
        # print(pic2)
        alpha = 0.4  # 调整权重以减弱pic2的作用

        # 通过加权平均来融合图像
        train_data = cv2.addWeighted(pic1, 1 - alpha, pic2, alpha, 0)

        # 使用高斯滤波减少噪声
        train_data = cv2.GaussianBlur(train_data, (5, 5), 0)

        cv2.imwrite(arguments_strOut,train_data)


if __name__ == "__main__":
    input_path1 = "D:\FYP\dataset\canny_pic"
    input_path2 = "D:\FYP\dataset\processed_pic"
    output_path = "D:/FYP/dataset/train_data"
    os.makedirs("D:/FYP/dataset/train_data/" , exist_ok=True)
    names = ['Brush','Harvard','met-1','met-2','Princeton-1','Princeton-2','Smithsonian-1','Smithsonian-2','Smithsonian-3','Smithsonian-4','Smithsonian-5','Smithsonian-6','Smithsonian-7']
    for name in names:
        os.makedirs("D:/FYP/dataset/train_data/%s" % name , exist_ok=True)
    process_all_train_data(input_path1,input_path2,output_path)