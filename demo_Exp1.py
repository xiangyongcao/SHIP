#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:43:53 2018

@author: xiangyong cao
Email: caoxiangyong45@gmail.com    or   caoxiangyong@mail.xjtu.edu.cn

Demo of experiement 1 in the paper: selecting the best classifier for a given feature
"""

from utils import SHSIC, file_check
import time


start_time = time.time()

# settings
dataset = "salinasA"
feature = 'raw'   
classifiers = ["KNN","GaussNB","LDA","LR","KSVM","DT","RF","GB","MLR"]   
train_size = 0.01   
repeat_num = 1     
model_selection = True  
isdraw = True

if isdraw==True:
    file_check(dataset)

# run 
Cla_Acc_Mean,Cla_Acc_Std,Seg_Acc_Mean,Seg_Acc_Std,df_result = SHSIC(dataset,feature,classifiers,\
                                                                    train_size,repeat_num,\
                                                                    model_selection,isdraw)


print("---------------------------Results Summary-----------------------------")
print("Dataset: "+ dataset)
print("Feature: "+ feature)
print("CLassifier: "+ str(classifiers))
print("The classification result is:")
print(df_result)
print('The best classifier for ' + feature + ' feature is ' + str(classifiers[Seg_Acc_Mean.argmax()]) + '.')
print("Time cost is %.3f" %(time.time()-start_time))
if isdraw==True:
    print('The final classification maps are ')