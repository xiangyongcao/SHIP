# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:39:00 2017

@author: Xiangyong Cao
"""

import scipy.io
import numpy as np
import os
import math
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import spectral as spy
from pygco import cut_simple, cut_simple_vh
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import time
from sklearn.model_selection import GridSearchCV
import shutil


def fstack(Data,n1,random_state):
    n = int(n1)
    h1,w1,b1,num1 = Data.shape[0],Data.shape[1],Data.shape[2],Data.shape[3]
    np.random.seed(random_state)
    if n==1:
#        ind = np.array([np.random.randint(low=0,high=52)])   # high 52 for other dataset
        ind = np.array([0])   # for paviaU only
        Data = Data[:,:,:,ind[0]]
    else:
        ind = np.random.permutation(52)
        Temp = np.zeros((h1,w1,1))
        for i in range(n):
            Temp = np.dstack((Temp,Data[:,:,:,ind[i]]))
        Data = np.delete(Temp,(0),axis=2)
    return Data,ind[0:n]
    
def load_data(string,feature_type='raw'):
#    feature_type = feature_type.split('_')
    DATA_PATH = os.path.join(os.getcwd(),"dataset")
    if feature_type=='raw':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'.mat'))[string]
        ind = 0
    if feature_type=='3ddwt':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'_3ddwt.mat'))[string+'_3ddwt']
        ind = 0
    if feature_type=='moglowrank':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'_moglowrank.mat'))[string+'_moglowrank']
        ind = 0
    if feature_type=='lowrank':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'_lowrank.mat'))[string+'_lowrank']
        ind = 0
    if feature_type=='pca':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'.mat'))[string]
        data_all = Data.transpose(2,0,1).transpose(0,2,1).reshape(Data.shape[2],-1).transpose(1,0)
        pca = PCA(n_components=40)
        data_all = pca.fit_transform(data_all)
        ind = 0
    if feature_type=='SAE':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'_SAE.mat'))[string+'_SAE']
        ind = 0
    if feature_type=='3dgabor':
        Data = scipy.io.loadmat(os.path.join(DATA_PATH, string+'_3dgabor.mat'))[string+'_3dgabor']
        Data,ind = fstack(Data,'1',random_state=0)
    Label = scipy.io.loadmat(os.path.join(DATA_PATH, string+'_gt.mat'))[string+'_gt']
    Data = Data.astype(float)
    # some constant paramaters
    height, width, band = Data.shape[0], Data.shape[1], Data.shape[2]
    num_classes = len(np.unique(Label))-1

#    # Normalizations
#    for b in range(band):
#        temp = Data[:,:,b]
#        Data[:,:,b] = (temp-np.min(temp))/(np.max(temp)-np.min(temp))


    ## Transform tensor data into matrix data, each row represents a spectral vector  
    # transform 3D into 2D 
    if feature_type[0]=='pca':
        data_all = data_all
    else:
        data_all = Data.transpose(2,0,1).transpose(0,2,1).reshape(band,-1).transpose(1,0)
    
    # transform 2D into 1D
    label_all = Label.transpose(1,0).flatten()
    
#    # dimension reduction using PCA
#    if ispca:
#        pca = PCA(n_components=40)
#        data_all = pca.fit_transform(data_all)
    # remove the sepctral vectors whose labels are 0
    data = data_all[label_all!=0]
    label = label_all[label_all!=0]
    
    label_list = list(label_all)
    ind_each_class=[]
    for i in range(1,num_classes+1):
        ind_each_class.append([index for index, value in enumerate(label_list) if value == i]) 
    ind_each_class = np.asarray(ind_each_class)
    return data_all, label_all, data, label, height, width, num_classes, Label, ind, ind_each_class
    

def split_each_class(samples_class_k, labels_class_k, ind_each_class_k,
                     num_train_class_k,one_hot=False,random_state=0):
    idx = np.arange(0, len(samples_class_k))  # get all possible indexes
    np.random.seed(random_state)
    np.random.shuffle(idx)  # shuffle indexes
    num_train_class_k = int(num_train_class_k)
    idx_train = idx[0:num_train_class_k]
    idx_test = idx[num_train_class_k:]  
    X_train_class_k = np.asarray([samples_class_k[i] for i in idx_train])  
    X_test_class_k = np.asarray([samples_class_k[i] for i in idx_test])  
    if one_hot:
        y_train_class_k = np.asarray([labels_class_k[i] for i in idx_train])
        y_test_class_k = np.asarray([labels_class_k[i] for i in idx_test])
    else:
        y_train_class_k = np.asarray([labels_class_k[i] for i in idx_train]).reshape(len(idx_train),1)
        y_test_class_k = np.asarray([labels_class_k[i] for i in idx_test]).reshape(len(idx_test),1)
        tr_index_k = np.asarray([ind_each_class_k[i] for i in idx_train]).reshape(len(idx_train),1)
        te_index_k = np.asarray([ind_each_class_k[i] for i in idx_test]).reshape(len(idx_test),1)
    return X_train_class_k, y_train_class_k,X_test_class_k, y_test_class_k, tr_index_k, te_index_k
    
def list2array(X,isdata=True,one_hot=False):
    if isdata:
        Y = np.zeros(shape=(1,X[0].shape[1]))
        for k in range(len(X)):
            Y = np.vstack((Y,X[k]))
        Y = np.delete(Y,(0),axis=0)
    else:
        if one_hot:
            Y = np.zeros(shape=(1,X[0].shape[1]))
            for k in range(len(X)):
                Y = np.vstack((Y,X[k]))
            Y = np.delete(Y,(0),axis=0)                
        else:
            Y = np.zeros(shape=(1,))
            for k in range(len(X)):
                Y = np.vstack((Y,X[k]))
            Y = np.delete(Y,(0),axis=0)
    return Y 
    
def split_train_test(X, y, train_size, ind_each_class, one_hot=False, random_state=0):
    #sample_each_class, label_each_class, num_train_each, train_rate, proportion):
    num_classes = len(np.unique(y))
    sample_each_class = np.asarray([X[y==k] for k in range(1,num_classes+1)]) 
    if one_hot:
        y_0 = y - 1
        y_onehot = convertToOneHot(y_0)
        label_each_class = np.asarray([y_onehot[y_0==k] for k in range(num_classes)]) 
    label_each_class  = np.asarray([y[y==k] for k in range(1,num_classes+1)]) 
    num_each_class = [len(sample_each_class[k]) for k in range(num_classes)]
    if train_size>=0 and train_size<=1:
        num_train = [math.ceil(train_size * i) for i in num_each_class]
    else:
        num_train = [train_size/num_classes] * num_classes
    X_train, y_train, X_test, y_test, train_indexes, test_indexes = [],[],[],[],[],[]
    for k in range(num_classes):
        X_train_class_k, y_train_class_k, X_test_class_k, y_test_class_k, tr_index_k, te_index_k =\
               split_each_class(sample_each_class[k], label_each_class[k],
                                ind_each_class[k],num_train[k], one_hot, random_state)
        X_train.append(X_train_class_k)
        y_train.append(y_train_class_k)
        X_test.append(X_test_class_k)
        y_test.append(y_test_class_k) 
        train_indexes.append(tr_index_k)
        test_indexes.append(te_index_k)
    X_train = list2array(X_train)
    X_test  = list2array(X_test)
    y_train = list2array(y_train,isdata=False,one_hot=False)
    y_test  = list2array(y_test,isdata=False,one_hot=False)
    train_indexes  = list2array(train_indexes,isdata=False,one_hot=False)
    test_indexes   = list2array(test_indexes,isdata=False,one_hot=False)
    if one_hot==False:
        y_train=y_train.reshape((y_train.shape[0],))
        y_test=y_test.reshape((y_test.shape[0],))
        train_indexes = train_indexes.reshape((train_indexes.shape[0],))
        test_indexes = test_indexes.reshape((test_indexes.shape[0],))
    return X_train,X_test, y_train, y_test, train_indexes, test_indexes


def train_test_map(data_all,train_indexes,test_indexes,label_all,GT_Label):
    train_map = np.zeros(len(data_all))
    test_map  = np.zeros(len(data_all))
    train_indexes = train_indexes.astype(int)
    test_indexes = test_indexes.astype(int)
    train_map[train_indexes] = label_all[train_indexes]
    test_map[test_indexes] = label_all[test_indexes]
    train_map = train_map.reshape(GT_Label.shape[1],GT_Label.shape[0]).transpose(1,0).astype(int)
    test_map  = test_map.reshape(GT_Label.shape[1],GT_Label.shape[0]).transpose(1,0).astype(int)
    return train_map,test_map

 
def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)    
    
def data_summary(y_train,y,num_classes):
    df = pd.DataFrame(np.random.randn(num_classes, 3),
                      index=['class_'+np.str(i) for i in range(1,1+num_classes)],
                  columns=['Train', 'Test', 'Total'])
    df['Train'] = [sum(y_train==i) for i in range(1,num_classes+1)]
    df['Total'] = [sum(y==i) for i in range(1,num_classes+1)]
    df['Test'] = np.array(df['Total']) - np.array(df['Train'])
    return df



def print_data_summary(y_train,y_test,y,num_classes):
    df = pd.DataFrame(np.random.randn(num_classes, 3),
                      index=['class_'+np.str(i) for i in range(1,1+num_classes)],
                  columns=['Train', 'Test', 'Total'])
    df['Train'] = [sum(y_train==i) for i in range(1,num_classes+1)]
    df['Total'] = [sum(y==i) for i in range(1,num_classes+1)]
    df['Test'] = np.array(df['Total']) - np.array(df['Train'])
    print('Summary of training and testing samples:')
    print(df)
    print("Training samples: %d" % len(y_train))
    print("Test samples: %d" % len(y_test))
#    
#def scale_data(X_train,X_test,data_all):
#    scaler = MinMaxScaler()
#    X_train_scaled = scaler.fit_transform(X_train)
#    X_test_scaled  = scaler.transform(X_test) 
#    data_all_scaled = scaler.transform(data_all) 
#    return X_train_scaled,X_test_scaled,data_all_scaled

def CA_summary(ca_each,sa_each,oa_class,oa_seg,num_classes):
    index1 = ['class_'+np.str(i) for i in range(1,1+num_classes)]
    index1.append('AA')
    index1.append('OA')
    df = pd.DataFrame(np.random.randn(num_classes+2, 2),
                      index=index1,
                  columns=['Classification', 'Segmentation'])
    ca_each = list(ca_each)
    sa_each = list(sa_each)
    ca_each.append(np.mean(ca_each))
    sa_each.append(np.mean(sa_each))
    ca_each.append(oa_class)
    sa_each.append(oa_seg)
    df['Classification'] = ca_each
    df['Segmentation'] = sa_each
    return df
    
def draw(GT_Label,ES_Label,Seg_Label,train_map,test_map):
    
    fig = plt.figure(figsize=(12,6))

    p = plt.subplot(1, 5, 1)
    v = spy.imshow(classes=GT_Label, fignum=fig.number)
    p.set_title('Ground Truth')
    p.set_xticklabels([])
    p.set_yticklabels([])

    p = plt.subplot(1, 5, 2)
    spy.imshow(classes = train_map , fignum=fig.number)
    p.set_title('Training Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
    p = plt.subplot(1, 5, 3)
    v = spy.imshow(classes = test_map, fignum=fig.number)
    p.set_title('Testing Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
    p = plt.subplot(1, 5, 4)
    v = spy.imshow(classes = ES_Label * (GT_Label != 0), fignum=fig.number)
    p.set_title('Classification Map')
    p.set_xticklabels([])
    p.set_yticklabels([])

    p = plt.subplot(1, 5, 5)
    v = spy.imshow(classes = Seg_Label * (GT_Label != 0), fignum=fig.number)
    p.set_title('Segmentation Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
def draw_part(GT_Label,ES_Label,train_map,test_map):

    fig = plt.figure(figsize=(12,6))

    p = plt.subplot(1, 4, 1)
    v = spy.imshow(classes=GT_Label, fignum=fig.number)
    p.set_title('Ground Truth')
    p.set_xticklabels([])
    p.set_yticklabels([])

    p = plt.subplot(1, 4, 2)
    spy.imshow(classes = train_map , fignum=fig.number)
    p.set_title('Training Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
    p = plt.subplot(1, 4, 3)
    v = spy.imshow(classes = test_map, fignum=fig.number)
    p.set_title('Testing Map')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
    p = plt.subplot(1, 4, 4)
    v = spy.imshow(classes = ES_Label * (GT_Label != 0), fignum=fig.number)
    p.set_title('Classification Map')
    p.set_xticklabels([])
    p.set_yticklabels([])

def unaries_reshape(unaries,height,width,num_classes):
    una = []
    for i in range(num_classes):
        temp = unaries[:,i].reshape(height,width).transpose(1,0)
        una.append(temp)
    return np.dstack(una).copy("C")

def Post_Processing(prob_map,height,width,num_classes,y_test,test_indexes):
    Gamma = [10,20,30,50,100,150,200]
#    Gamma = [20]
    SL = np.zeros([len(Gamma),height,width])
    SAE = np.zeros([len(Gamma),num_classes])
    SA = np.zeros([len(Gamma)])
    for j in range(len(Gamma)):
        gamma = Gamma[j]
        unaries = (-gamma*np.log(prob_map+1e-4)).astype(np.int32)      # 20   15
        una = unaries_reshape(unaries,width,height,num_classes)
        one_d_topology = (np.ones(num_classes)-np.eye(num_classes)).astype(np.int32).copy("C")
        Seg_Label = cut_simple(una, 100 * one_d_topology)# 30   200
        Seg_Label = Seg_Label + 1
        seg_Label = Seg_Label.transpose().flatten()
        test_indexes = test_indexes.astype(np.int32)
        cmat = confusion_matrix(y_test,seg_Label[test_indexes])
        cmat.astype(float)
        a = cmat.diagonal()
        a = np.array([float(a[i]) for i in range(0,len(a))])
        b = cmat.sum(axis=1) 
        seg_accuracy_each = a/b
        seg_accuracy = accuracy_score(y_test,seg_Label[test_indexes])
        SL[j,:,:] = Seg_Label
        SAE[j,:] = seg_accuracy_each
        SA[j] = seg_accuracy
    SA = SA.tolist()
    max_ind = SA.index(max(SA))
    seg_accuracy = SA[max_ind]
    Seg_Label = SL[max_ind,:,:]
    seg_accuracy_each = SAE[max_ind,:]
    return Seg_Label, seg_accuracy, seg_accuracy_each
    
def classification_pipeline(classifier,X_train,y_train,X_test,y_test,data_all,\
                            width,height,num_classes,test_indexes,\
                            num_train_each_class, model_selection=True):
    Classifiers = ["KNN","GaussNB","LDA","LR","KSVM","DT","RF","GB","MLR"]
    IsScale = [True,False,False,True,True,False,False,False,True]
    is_scale = IsScale[Classifiers.index(classifier)]
    if is_scale:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test) 
        data_all = scaler.transform(data_all) 
    
    if classifier=="KNN":
        start_time = time.time()
        if model_selection==True:
            Clf = KNeighborsClassifier()
            param_grid = {'n_neighbors':[3,5,7,9]}
            if np.sum(num_train_each_class<5)==len(num_train_each_class):
                nfolds = 3;
            else:
                nfolds = 5
            best_params = param_selection(Clf, X_train, y_train, param_grid, nfolds)
            print("KNN----------------------")
            print("The parameter grid is:")
            print(param_grid)
            print("The best parameter is:")
            print(best_params)
            KNN = KNeighborsClassifier(n_neighbors=best_params['n_neighbors']).fit(X_train,y_train)
#        KNN = KNeighborsClassifier(n_neighbors=7).fit(X_train,y_train)
        if model_selection==False:
            n_neighbors = 5
            KNN = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train,y_train)
        Cla_Map = KNN.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
        predict_prob = KNN.predict_proba(data_all)
        # Post-processing using Graph-Cut
        Seg_Map, seg_accuracy, seg_accuracy_each = Post_Processing(predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
        print('(KNN) Train_Acc=%.3f, Test_Cla_Acc=%.3f, Seg_Acc=%.3f (Time_cost=%.3f)'\
              % (KNN.score(X_train,y_train),KNN.score(X_test,y_test),\
                 seg_accuracy, (time.time()-start_time)))
        cla_accuracy = KNN.score(X_test,y_test)
#        time_cost = time.time()-start_time
    
    if classifier=="GaussNB":        
        start_time = time.time()
        GaussNB = GaussianNB().fit(X_train,y_train)
        Cla_Map = GaussNB.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
        predict_prob = GaussNB.predict_proba(data_all)
        # Post-processing using Graph-Cut
        Seg_Map, seg_accuracy, seg_accuracy_each = Post_Processing(predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
        print('(GaussNB) Train_Acc=%.3f, Test_Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
              % (GaussNB.score(X_train,y_train),GaussNB.score(X_test,y_test),\
                 seg_accuracy, (time.time()-start_time)))
        cla_accuracy = GaussNB.score(X_test,y_test)
#        time_cost = time.time()-start_time
        
    if classifier=="LDA":        
        start_time = time.time()
        LDA = LinearDiscriminantAnalysis().fit(X_train,y_train)
        Cla_Map = LDA.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
        predict_prob = LDA.predict_proba(data_all)
        # Post-processing using Graph-Cut
        Seg_Map, seg_accuracy, seg_accuracy_each = Post_Processing(predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
        print('(LDA) Train_Acc=%.3f, Test_Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
              % (LDA.score(X_train,y_train),LDA.score(X_test,y_test),\
                 seg_accuracy, (time.time()-start_time)))
        cla_accuracy = LDA.score(X_test,y_test)
#        time_cost = time.time()-start_time
        
    if classifier=="LR":        
        start_time = time.time()
        if model_selection==True:
            Clf = LogisticRegression(multi_class='multinomial',solver='lbfgs')
            param_grid = {'C':[0.1,1,10,20,30,50]}
            if np.sum(num_train_each_class<5)==len(num_train_each_class):
                nfolds = 3;
            else:
                nfolds = 5
            best_params = param_selection(Clf, X_train, y_train, param_grid, nfolds)
            print("LR----------------------")
            print("The parameter grid is:")
            print(param_grid)
            print("The best parameter is:")
            print(best_params)
            LR = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=best_params['C']).fit(X_train,y_train)
        if model_selection==False:
            LR = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=1).fit(X_train,y_train)       
        Cla_Map = LR.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
        predict_prob = LR.predict_proba(data_all)
        # Post-processing using Graph-Cut
        Seg_Map, seg_accuracy, seg_accuracy_each = Post_Processing(predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
        print('(LR) Train_Acc=%.3f, Test_Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
              % (LR.score(X_train,y_train),LR.score(X_test,y_test),\
                 seg_accuracy, (time.time()-start_time)))
        cla_accuracy = LR.score(X_test,y_test)
#        time_cost = time.time()-start_time
                
    if classifier=="KSVM":        
        start_time = time.time()
        if model_selection==True:
            Clf = SVC(probability=True)
            param_grid = {'C':[2**(-9),2**(-8),2**(-7),2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),\
                           2**(-1),2**(0),2**(1),2**(2),2**(3),2**(4),2**(5),2**(6),2**(7),2**(8),2**(9)]}
            if np.sum(num_train_each_class<5)==len(num_train_each_class):
                nfolds = 3;
            else:
                nfolds = 5
            best_params = param_selection(Clf, X_train, y_train, param_grid, nfolds)
            print("KSVM----------------------")
            print("The parameter grid is:")
            print(param_grid)
            print("The best parameter is:")
            print(best_params)
            SVM = SVC(C=best_params['C'],probability=True).fit(X_train, y_train)
        if model_selection==False:
            SVM = SVC(C=512,probability=True).fit(X_train, y_train)
        Cla_Map = SVM.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
        predict_prob = SVM.predict_proba(data_all)
        # Post-processing using Graph-Cut
        Seg_Map, seg_accuracy, seg_accuracy_each = Post_Processing(predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
        print('(Kernel SVM) Train_Acc=%.3f, Test_Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
              % (SVM.score(X_train,y_train),SVM.score(X_test,y_test),\
                 seg_accuracy, (time.time()-start_time)))
        cla_accuracy = SVM.score(X_test,y_test)
#        time_cost = time.time()-start_time
        
    if classifier=="DT":        
        start_time = time.time()
        if model_selection==True:
            Clf = DecisionTreeClassifier()
            param_grid = {'max_depth':[5,10,20,50,100,200,300]}
            if np.sum(num_train_each_class<5)==len(num_train_each_class):
                nfolds = 3;
            else:
                nfolds = 5
            best_params = param_selection(Clf, X_train, y_train, param_grid, nfolds)
            print("DT----------------------")
            print("The parameter grid is:")
            print(param_grid)
            print("The best parameter is:")
            print(best_params)
            DTree = DecisionTreeClassifier(max_depth=best_params['max_depth']).fit(X_train,y_train)
        if model_selection==False:
            DTree = DecisionTreeClassifier(max_depth=200).fit(X_train,y_train)
        Cla_Map = DTree.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
        predict_prob = DTree.predict_proba(data_all)
        # Post-processing using Graph-Cut
        Seg_Map, seg_accuracy, seg_accuracy_each = Post_Processing(predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
        print('(Decision Tree) Train_Acc=%.3f, Test_Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
              % (DTree.score(X_train,y_train),DTree.score(X_test,y_test),\
                 seg_accuracy, (time.time()-start_time)))
        cla_accuracy = DTree.score(X_test,y_test)
#        time_cost = time.time()-start_time
        
    if classifier=="RF":        
        start_time = time.time()
        if model_selection==True:
            Clf = RandomForestClassifier()
            param_grid = {'n_estimators':[5,10,20,50,100,200,300]}
            if np.sum(num_train_each_class<5)==len(num_train_each_class):
                nfolds = 3;
            else:
                nfolds = 5
            best_params = param_selection(Clf, X_train, y_train, param_grid, nfolds)
            print("RF----------------------")
            print("The parameter grid is:")
            print(param_grid)
            print("The best parameter is:")
            print(best_params)
            RF = RandomForestClassifier(n_estimators=best_params['n_estimators']).fit(X_train,y_train)
        if model_selection==False:
            RF = RandomForestClassifier(n_estimators=200).fit(X_train,y_train)
        Cla_Map = RF.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
        predict_prob = RF.predict_proba(data_all)
        # Post-processing using Graph-Cut
        Seg_Map, seg_accuracy, seg_accuracy_each = Post_Processing(predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
        print('(Random Forest) Train_Acc=%.3f, Test_Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
              % (RF.score(X_train,y_train),RF.score(X_test,y_test),\
                 seg_accuracy, (time.time()-start_time)))
        cla_accuracy = RF.score(X_test,y_test)
#        time_cost = time.time()-start_time
        
    if classifier=="GB":        
        start_time = time.time()
        if model_selection==True:
            Clf = GradientBoostingClassifier()
            param_grid = {'n_estimators':[10,50,100,200,300]}
            if np.sum(num_train_each_class<5)==len(num_train_each_class):
                nfolds = 3;
            else:
                nfolds = 5
            best_params = param_selection(Clf, X_train, y_train, param_grid, nfolds)
            print("GB----------------------")
            print("The parameter grid is:")
            print(param_grid)
            print("The best parameter is:")
            print(best_params)
            GB = GradientBoostingClassifier(n_estimators=best_params['n_estimators']).fit(X_train,y_train)
        if model_selection==False:
            GB = GradientBoostingClassifier(n_estimators=200).fit(X_train,y_train)
        Cla_Map = GB.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
        predict_prob = GB.predict_proba(data_all)
        # Post-processing using Graph-Cut
        Seg_Map, seg_accuracy, seg_accuracy_each = Post_Processing(predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
        print('(Gradient Boosting) Train_Acc=%.3f, Test_Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
              % (GB.score(X_train,y_train),GB.score(X_test,y_test),\
                 seg_accuracy, (time.time()-start_time)))
        cla_accuracy = GB.score(X_test,y_test)
#        time_cost = time.time()-start_time
        
    if classifier=="MLR":        
        start_time = time.time()
        if model_selection==True:
            Clf = MLPClassifier()
            param_grid = {'hidden_layer_sizes':[[50,50],[50,100],[50,200],[100,100],\
                                            [100,200],[200,100],[200,200],[200,300],\
                                            [200,500],[300,300],[300,400],[300,500],[400,500],[500,500]]}
            if np.sum(num_train_each_class<5)==len(num_train_each_class):
                nfolds = 3;
            else:
                nfolds = 5
            best_params = param_selection(Clf, X_train, y_train, param_grid, nfolds)
            print("MLR----------------------")
            print("The parameter grid is:")
            print(param_grid)
            print("The best parameter is:")
            print(best_params)
            MLP = MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes']).fit(X_train,y_train)
        if model_selection==False:
            MLP = MLPClassifier(hidden_layer_sizes=[300,400]).fit(X_train,y_train)
        Cla_Map = MLP.predict(data_all).reshape(width,height).astype(int).transpose(1,0)
        predict_prob = MLP.predict_proba(data_all)
        # Post-processing using Graph-Cut
        Seg_Map, seg_accuracy, seg_accuracy_each = Post_Processing(predict_prob,height,width,\
                                          num_classes,y_test,test_indexes)
        print('(MLP) Train_Acc=%.3f, Test_Cla_Acc=%.3f, Seg_Acc=%.3f(Time_cost=%.3f)'\
              % (MLP.score(X_train,y_train),MLP.score(X_test,y_test),\
                 seg_accuracy, (time.time()-start_time)))
        cla_accuracy = MLP.score(X_test,y_test)
#        time_cost = time.time()-start_time
        
    return Cla_Map,Seg_Map,cla_accuracy,seg_accuracy
    
    
def SHSIC(dataset,feature,classifiers,train_size,repeat_num=1,model_selection=True,isdraw=True):
    """
    ----Input----
    dataset: name of dataset ('indianpines' 'pavia' 'paviaU' 'salinas' 'salinasA' 'KSC' 'Botswana') 
    feature: name of feature ('raw'  lowrank'  '3ddwt'  '3dgabor'  'SAE'   'moglowrank')
    classifiers: ["KNN","GaussNB","LDA","LR","KSVM","DT","RF","GB","MLR"] or select some from these 
    train_size: number of training samples in each class 
                (train_size < 1 means the proportion for each class;   
                 train_size > 1 means the number for each class)
    repeated_num:  the repeated time of the experiment  
    model_selection: Flag of model selection for each classifier (True or False) 
    isdraw: Flag of showing classification map (True or False)
    
    ----Output----
    Cla_Acc_Mean: mean classification accuracy
    Cla_Acc_Std:  standard derivation of classification accuracy
    Seg_Acc_Mean: mean segmentation accuracy
    Seg_Acc_Std:  standard derivation of classification accuracy
    df_result:    result summary
"""
    print("--------------------------begin--------------------------------")
    print("Dataset: "+ dataset)
    print("Feature: "+ feature)
    print("CLassifier: "+ str(classifiers))
    ## Load data
    data_all, label_all, X, y, height, width, num_classes, GT_Map,ind,ind_each_class = load_data(dataset,feature)
    
    ## train-test-split
    if train_size>1: 
        train_size = train_size * num_classes
    
    X_train, X_test, y_train, y_test, train_indexes, test_indexes = \
           split_train_test(X, y, train_size, ind_each_class, random_state=0)
    ## train_test data summary
    print_data_summary(y_train,y_test,y,num_classes)
    
    Cla_accuracy = np.zeros((np.size(classifiers),repeat_num))
    Seg_accuracy = np.zeros((np.size(classifiers),repeat_num))  
    CLA_MAP = np.zeros((height,width,repeat_num,len(classifiers)))
    SEG_MAP = np.zeros((height,width,repeat_num,len(classifiers)))
    Train_Map = np.zeros((height,width,repeat_num))
    Test_Map = np.zeros((height,width,repeat_num))
    for j in range(repeat_num):
        print("-------------------------repeat %d-----------------------------" % (j+1))
        X_train, X_test, y_train, y_test, train_indexes, test_indexes = \
           split_train_test(X, y, train_size, ind_each_class, random_state=j)

        ## train map and test map (preparation for draw)
        train_map,test_map = train_test_map(data_all,train_indexes,test_indexes,label_all,GT_Map)  
        Train_Map[:,:,j] = train_map
        Test_Map[:,:,j] = test_map
        
        num_train_each_class = np.array([np.sum(y_train==i+1) for i in range(num_classes)])
 
    
        ## Classification
        for i in range(len(classifiers)):
            classifier = classifiers[i]
            Cla_Map,Seg_Map,cla_accuracy,seg_accuracy = classification_pipeline(classifier,X_train,y_train,X_test,y_test,data_all,\
                                                                                      width,height,num_classes,test_indexes,\
                                                                                      num_train_each_class,model_selection)
            Cla_accuracy[i,j] = cla_accuracy
            Seg_accuracy[i,j] = seg_accuracy
            
            CLA_MAP[:,:,j,i] = Cla_Map
            SEG_MAP[:,:,j,i] = Seg_Map
    
    # draw classification map
    Train_Map = Train_Map.astype(int)
    Test_Map = Test_Map.astype(int)
    CLA_MAP = CLA_MAP.astype(int)
    SEG_MAP = SEG_MAP.astype(int)
    DATA_PATH1 = os.path.join(os.getcwd(),"image_show")
    DATA_PATH = os.path.join(DATA_PATH1,dataset)
#    if os.path.exists(DATA_PATH)==True:
#        shutil.rmtree(DATA_PATH)
#    if os.path.exists(DATA_PATH)==False:
#        os.mkdir(DATA_PATH)
#    source_file = DATA_PATH1 + '/'+ dataset + '_gt.mat'
#    destination_file = DATA_PATH
#    shutil.copy(source_file,destination_file)
    for l in range(len(classifiers)):
        index_max = np.argmax(Seg_accuracy[l,:])
        seg_acc = max(Seg_accuracy[l,:])
        seg_acc = seg_acc.astype('str')
        Cla_Map_Max = CLA_MAP[:,:,index_max,l]
        Seg_Map_Max = SEG_MAP[:,:,index_max,l]
        train_map_j = Train_Map[:,:,index_max]
        test_map_j = Test_Map[:,:,index_max]
        if isdraw==True:
            Temp = {}
            Temp["Seg_Map"] = Seg_Map_Max
            feature1 = feature.split('_')
            file_name = dataset + "_" + feature1[0] + "_" + classifiers[l] + "_Seg_Map" + "_" + seg_acc + "_.mat"
            scipy.io.savemat(os.path.join(DATA_PATH, file_name),Temp)
#            draw(GT_Map,Cla_Map_Max,Seg_Map_Max,train_map_j,test_map_j)
    if isdraw==True:
        Comparison_draw(dataset)
    Cla_Acc_Mean = np.mean(Cla_accuracy,axis=1)
    Cla_Acc_Std = np.std(Cla_accuracy,axis=1)
    Seg_Acc_Mean = np.mean(Seg_accuracy,axis=1)
    Seg_Acc_Std = np.std(Seg_accuracy,axis=1)
        
    df_result = pd.DataFrame(np.random.randn(np.size(classifiers),4),index=classifiers,\
                                 columns=['Cla_Mean','Cla_Std','Seg_Mean','Seg_Std'])
    df_result['Cla_Mean'] = Cla_Acc_Mean
    df_result['Cla_Std'] = Cla_Acc_Std
    df_result['Seg_Mean'] = Seg_Acc_Mean
    df_result['Seg_Std'] = Seg_Acc_Std
        
    return Cla_Acc_Mean,Cla_Acc_Std,Seg_Acc_Mean,Seg_Acc_Std,df_result


def param_selection(Clf, X_train, y_train, param_grid, nfolds):
    grid_search = GridSearchCV(Clf, param_grid, cv=nfolds)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params


def Comparison_draw(dataset):
    DATA_PATH1 = os.path.join(os.getcwd(),"image_show")
    DATA_PATH = os.path.join(DATA_PATH1,dataset)
    filename_list = os.listdir(DATA_PATH)
    h = []
    for filename in filename_list:
        if filename.startswith(dataset):
            h.append(filename)
#    new_h = sorted(h,key=lambda i:len(i), reverse=False)
    new_h = sorted(h,key=lambda x: os.path.getmtime(os.path.join(DATA_PATH, x)),reverse=False)
#    new_h = sorted(h,key=lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(x))), reverse=False)
    GT_Label = scipy.io.loadmat(os.path.join(DATA_PATH, new_h[0]))[dataset+"_gt"]
    
    fig = plt.figure(figsize=(12,12))
    
    p = plt.subplot(3, 4, 1)
    v = spy.imshow(classes=GT_Label, fignum=fig.number)
    p.set_title('Ground Truth')
    p.set_xticklabels([])
    p.set_yticklabels([])
    
    for i in range(len(new_h)-1):
        file_name = new_h[i+1]
        file_name_split = file_name.split('_')
        ES_Label = scipy.io.loadmat(os.path.join(DATA_PATH, file_name))["Seg_Map"]
        title = file_name_split[1] + "+" + file_name_split[2]
        seg_acc = float('%.4f' % float(file_name_split[-2]))
        
        p = plt.subplot(3, 4, i+2)
        v = spy.imshow(classes = ES_Label * (GT_Label != 0), fignum=fig.number)
        p.set_title(title+"("+'%.2f' % (100*seg_acc)+ "%"+")")
        p.set_xticklabels([])
        p.set_yticklabels([])
    
    

def file_check(dataset):
    DATA_PATH1 = os.path.join(os.getcwd(),"image_show")
    DATA_PATH = os.path.join(DATA_PATH1,dataset)
    if os.path.exists(DATA_PATH)==True:
        shutil.rmtree(DATA_PATH)
    if os.path.exists(DATA_PATH)==False:
        os.mkdir(DATA_PATH)
    source_file = DATA_PATH1 + '/'+ dataset + '_gt.mat'
    destination_file = DATA_PATH
    shutil.copy(source_file,destination_file)    
    
    
    
    
