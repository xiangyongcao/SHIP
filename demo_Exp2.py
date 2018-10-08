#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 18:34:48 2018

@author: xiangyong cao
Email: caoxiangyong45@gmail.com    or   caoxiangyong@mail.xjtu.edu.cn

Demo of experiement 2 in the paper 
"""

from utils import SHSIC, file_check
import time


start_time = time.time()

# settings
dataset = "indianpines"
features = ['raw','pca','lowrank','3ddwt','3dgabor', 'SAE']
classifiers = [["RF"],["KSVM"],["RF"],["GB"],["KSVM"],["RF"]]
train_sizes=[0.01]
repeat_num = 1
model_selection=False
isdraw=True

if isdraw==True:
    file_check(dataset)

DF = {}
for j in range(len(features)):
    for i in range(len(train_sizes)):
        # run 
        Cla_Acc_Mean,Cla_Acc_Std,Seg_Acc_Mean,Seg_Acc_Std,df_result = SHSIC(dataset,features[j],classifiers[j],\
                                                                        train_sizes[i],repeat_num,\
                                                                        model_selection=False,isdraw=True)
        DF[i] = df_result


    print("-----------------------Results Summary-------------------------")
    for i in range(len(train_sizes)):
        print("Classification results are (%.2f):" % (train_sizes[i]))
        print(DF[i])