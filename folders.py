import pandas as pd
import cv2
import os
import numpy as np
import csv


csvi = "./Data/archive/Folds.csv"
data = pd.read_csv(csvi)


for i,d in enumerate(data["filename"]):
    print(i)
    if(d.find("benign")!=-1):
        if(data["grp"][i]=='train'):
            img = cv2.imread("./Data/archive/BreaKHis_v1/"+d)
            cv2.imwrite("./Data/Cancer/Train/Benign/"+str(i)+".jpg",img)
            with open('train.csv', 'a', newline='') as csvfile:
                fieldnames = ['Image','Benign']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Image':"./Data/Cancer/Train/Benign/"+str(i)+".jpg", 'Benign':1})
        else:
            img = cv2.imread("./Data/archive/BreaKHis_v1/"+d)
            cv2.imwrite("./Data/Cancer/Test/Benign/"+str(i)+".jpg",img)
            with open('test.csv', 'a', newline='') as csvfile:
                fieldnames = ['Image','Benign']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Image':"./Data/Cancer/Test/Benign/"+str(i)+".jpg", 'Benign':1})

    else:
        if(data["grp"][i]=="train"):
            img = cv2.imread("./Data/archive/BreaKHis_v1/"+d)
            cv2.imwrite("./Data/Cancer/Train/Malignant/"+str(i)+".jpg",img)
            with open('train.csv', 'a', newline='') as csvfile:
                fieldnames = ['Image','Benign']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Image':"./Data/Cancer/Train/Malignant/"+str(i)+".jpg", 'Benign':0})
            pass
        else:
            img = cv2.imread("./Data/archive/BreaKHis_v1/"+d)
            cv2.imwrite("./Data/Cancer/Test/Malignant/"+str(i)+".jpg",img)
            with open('test.csv', 'a', newline='') as csvfile:
                fieldnames = ['Image','Benign']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Image':"./Data/Cancer/Test/Malignant/"+str(i)+".jpg", 'Benign':0})

