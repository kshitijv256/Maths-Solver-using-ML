'''
Driver file for input image processing and result prediction
written on 4/14/2022 by kshitijv256

'''

import cv2
import os
import numpy as np
from utils import pick
from PIL import Image
from skimage.morphology import skeletonize

'''
I used SVM classifier here because that gave me the best accuracy 
Try out different classifiers if you would like

'''

# Load model
model = pick('./saved/SVM2.pickle') 

# Load Image

## Just for fun ##
# print('Choice: ')
# folder = './real'
# x = os.listdir(folder)
# [print(i) for i in x]
# inp = input('Select: ').strip()
# if inp+'.jpg' in x:
#     path = os.path.join(folder,inp+'.jpg')
# elif inp+'.jpeg' in x:
#     path = os.path.join(folder,inp+'.jpeg')
# elif inp+'.png' in x:
#     path = os.path.join(folder,inp+'.png')
# else:
#     path = os.path.join(folder,inp)

path = "test.png"   ## Set Input Image ath here

x = y = cv2.imread(path)
img = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)


cv2.imshow("wo",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours
if img is not None:
    img=~img
    ret,thresh=cv2.threshold(img,127,255,0)
    ctrs,ret=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    ## To Draw Contours ##
    # image_copy =x
    # cv2.drawContours(image=image_copy, contours=ctrs, contourIdx=-1,color=(200, 0, 200),thickness=2, lineType=cv2.LINE_AA)
    # cv2.imshow('Contours', image_copy)
    # cv2.waitKey(0)
    # cv2.imwrite('contours.jpg', image_copy)
    # cv2.destroyAllWindows()



    w=int(28)
    h=int(28)
    testset=[]

    rects=[]
    for c in cnt :
        x,y,w,h= cv2.boundingRect(c)
        rect=[x,y,w,h]
        rects.append(rect)

    bool_rect=[]
    for r in rects:
        l=[]
        for rec in rects:
            flag=0
            if rec!=r:
                if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                    flag=1
                l.append(flag)
            if rec==r:
                l.append(0)
        bool_rect.append(l)

    dump_rect=[]
    for i in range(0,len(cnt)):
        for j in range(0,len(cnt)):
            if bool_rect[i][j]==1:
                area1=rects[i][2]*rects[i][3]
                area2=rects[j][2]*rects[j][3]
                if(area1==min(area1,area2)):
                    dump_rect.append(rects[i])

    final_rect=[i for i in rects if i not in dump_rect]

    for r in final_rect:
        x=r[0]
        y=r[1]
        w=r[2]
        h=r[3]
        im_crop =thresh[y:y+h+10,x:x+w+10]

        height, width = im_crop.shape
        x = height if height > width else width
        y = height if height > width else width

        ## To make contours square
        square= np.zeros((x,y), np.uint8)
        square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = im_crop
        im_crop = square
        
        im_crop = im_crop/255.0

        #im_crop = skeletonize(im_crop,method='lee')
        

        img = cv2.resize(im_crop,(28,28))
        cv2.imshow("work",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img=np.reshape(img,(784))
        img = img.reshape(1,-1)
        testset.append(img)

# Predicting result
s=''
for i in range(len(testset)):
    result=model.predict(testset[i])
    
    if(result[0]==10):
        s=s+'-'
    if(result[0]==11):
        s=s+'+'
    if(result[0]==12):
        s=s+'*'
    # if(result[0]==13):
    #     s=s+'/'
    if(result[0]==0):
        s=s+'0'
    if(result[0]==1):
        s=s+'1'
    if(result[0]==2):
        s=s+'2'
    if(result[0]==3):
        s=s+'3'
    if(result[0]==4):
        s=s+'4'
    if(result[0]==5):
        s=s+'5'
    if(result[0]==6):
        s=s+'6'
    if(result[0]==7):
        s=s+'7'
    if(result[0]==8):
        s=s+'8'
    if(result[0]==9):
        s=s+'9'
print('\n\n\n\n','Result: ')   
print(s," = ",eval(s))
print('\n\n\n') 
