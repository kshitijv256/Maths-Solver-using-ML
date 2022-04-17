
import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_images_from_folder(folder):
    folder = 'trainImg/'+folder    # change folder name to folder path where images are present
    train_data=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        img=~img
        if img is not None:
            im_resize = cv2.resize(img,(28,28))
            im_resize=np.reshape(im_resize,(784,1))
            train_data.append(im_resize)
    return train_data

data =[]

#lable '0'
data=load_images_from_folder('0')
for i in range(0,len(data)):
    data[i]=np.append(data[i],['0'])
print('0:',len(data))

#lable '1'
data1=load_images_from_folder('1')
for i in range(0,len(data1)):
    data1[i]=np.append(data1[i],['1'])
data=np.concatenate((data,data1))
print('1:',len(data))

#lable '2'
data2=load_images_from_folder('2')

for i in range(0,len(data2)):
    data2[i]=np.append(data2[i],['2'])
data=np.concatenate((data,data2))
print('2:',len(data))

#lable '3'
data3=load_images_from_folder('3')

for i in range(0,len(data3)):
    data3[i]=np.append(data3[i],['3'])
data=np.concatenate((data,data3))
print('3:',len(data))

#lable '4'
data4=load_images_from_folder('4')

for i in range(0,len(data4)):
    data4[i]=np.append(data4[i],['4'])
data=np.concatenate((data,data4))
print('4:',len(data))

#lable '5'
data5=load_images_from_folder('5')

for i in range(0,len(data5)):
    data5[i]=np.append(data5[i],['5'])
data=np.concatenate((data,data5))
print('5:',len(data))

#lable '6'
data6=load_images_from_folder('6')

for i in range(0,len(data6)):
    data6[i]=np.append(data6[i],['6'])
data=np.concatenate((data,data6))
print('6:',len(data))

#lable '7'
data7=load_images_from_folder('7')

for i in range(0,len(data7)):
    data7[i]=np.append(data7[i],['7'])
data=np.concatenate((data,data7))
print('7:',len(data))

#lable '8'
data8=load_images_from_folder('8')

for i in range(0,len(data8)):
    data8[i]=np.append(data8[i],['8'])
data=np.concatenate((data,data8))
print('8:',len(data))

#lable '9'
data9=load_images_from_folder('9')

for i in range(0,len(data9)):
    data9[i]=np.append(data9[i],['9'])
data=np.concatenate((data,data9))
print('9:',len(data))
                
#assign '-' = 10
data10=load_images_from_folder('-')
for i in range(0,len(data10)):
    data10[i]=np.append(data10[i],['10'])
data=np.concatenate((data,data10))
print('- :',len(data))

#assign '+' = 11
data11=load_images_from_folder('+')

for i in range(0,len(data11)):
    data11[i]=np.append(data11[i],['11'])
data=np.concatenate((data,data11))
print('+ :',len(data))

#assign * = 12
data12=load_images_from_folder('times')

for i in range(0,len(data12)):
    data12[i]=np.append(data12[i],['12'])
data=np.concatenate((data,data12))
print('* :',len(data))

'''
Do not add these files when using machine learning models like SVM 
it can mess up your accuracy

'''
# data13=load_images_from_folder('divv')
# for i in range(0,len(data13)):
#     data13[i]=np.append(data13[i],['13'])
# data=np.concatenate((data,data13))
# print('/ :',len(data))

df=pd.DataFrame(data,index=None)
df.to_csv('dataset.csv',index=False)    # Use any name for dataset
print('Dataset saved..')

# g =np.reshape(data[100][:-1],(45,45))
# plt.imshow(g)

