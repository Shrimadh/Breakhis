import pandas as pd
import cv2
import os
import numpy as np
import csv


csvi = "C:/Users/DELL/Desktop/Folds.csv"
data = pd.read_csv(csvi)


for i,d in enumerate(data["filename"]):

    if(d.find("benign")!=-1):
        if(data["grp"][i]=='train'):
            img = cv2.imread("C:/Users/DELL/Desktop/archive/BreaKHis_v1/"+d)
            cv2.imwrite("C:/Users/DELL/Desktop/Cancer/Train/Benign/"+str(i)+".jpg",img)
            with open('data.csv', 'a', newline='') as csvfile:
                fieldnames = ['Image','Benign']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Image':"C:/Users/DELL/Desktop/Cancer/Train/Benign/"+str(i)+".jpg", 'Benign':1})
            pass
        else:
            img = cv2.imread("C:/Users/DELL/Desktop/archive/BreaKHis_v1/"+d)
            cv2.imwrite("C:/Users/DELL/Desktop/Cancer/Test/Benign/"+str(i)+".jpg",img)
            with open('test.csv', 'a', newline='') as csvfile:
                fieldnames = ['Image','Benign']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Image':"C:/Users/DELL/Desktop/Cancer/Test/Benign/"+str(i)+".jpg", 'Benign':1})

    else:
        if(data["grp"][i]=="train"):
            img = cv2.imread("C:/Users/DELL/Desktop/archive/BreaKHis_v1/"+d)
            cv2.imwrite("C:/Users/DELL/Desktop/Cancer/Train/Malignant/"+str(i)+".jpg",img)
            with open('data.csv', 'a', newline='') as csvfile:
                fieldnames = ['Image','Benign']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Image':"C:/Users/DELL/Desktop/Cancer/Train/Malignant/"+str(i)+".jpg", 'Benign':0})
            pass
        else:
            img = cv2.imread("C:/Users/DELL/Desktop/archive/BreaKHis_v1/"+d)
            cv2.imwrite("C:/Users/DELL/Desktop/Cancer/Test/Malignant/"+str(i)+".jpg",img)
            with open('test.csv', 'a', newline='') as csvfile:
                fieldnames = ['Image','Benign']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Image':"C:/Users/DELL/Desktop/Cancer/Test/Malignant/"+str(i)+".jpg", 'Benign':0})

