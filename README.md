﻿# Supervised Hyperspectral Image Classification Package (SHIP)

This is an open package (called **SHIP**) for supervised hyperspectral image classification task. Besides, this repository guarantees you to reproduce the results reported in the paper:
 - [Xiangyong Cao, Zongben Xu and Deyu Meng, Spectral-Spatial Hyperspectral Image Classification via Robust Low-Rank Feature Extraction and Markov Random Field, Remote Sens. 2019, 11(13), 1565.](https://www.mdpi.com/2072-4292/11/13/1565) 

If you use this code, pleae cite this paper in your work. 

## Setup
### Install Dependencies
If you were using Ubuntu, simply type the following commands in your terminal to install dependencies: 

        python setup.py

### Download and prepare the datasets
Download the datasets used in the paper from the following link:
 - [https://pan.baidu.com/s/1mtAJ73RU8pb38GuIfe7FLQ](https://pan.baidu.com/s/1mtAJ73RU8pb38GuIfe7FLQ)
 - Code: g8bg
 
After downloading the datasets file, put it in the main directory of SHIP file.

## Reproducing the results

1. To choose the best classifier for one given feature (also reproduce the result in Exp 4.1), execute

        python demo_Exp1.py

   According to your needs, many parameters can be set in demo_Exp1.py, such as dataset, feature, classifieris, train_size, repeat_num, model_selection, isdraw. More detailed comments of these parameters can be found in SHSIC function. 
   
   This package supports 7 datasets, 6 features (5 classical features and 1 deep feature), 9 classifiers, model selection for each classifier, post-processing classification map and drawing classification map for each method . 

2. To reproduce the results in Exp 4.3, execute

        python demo_Exp2.py


   According to your needs, many parameters can be set in demo_Exp2.py.


## Contact:
This package is still developing and this is the first version. In the next step, we prepare to embed the feature extraction methods into this package, thus it can implement the feature extraction (this package only provide some pre-extracted features by some offline code). 

For the package of this version, we hope more reserachers in this field can provide your extracted feature data to me. Welcome to contact me (Xiangyong Cao:   caoxiangyong45@gmail.com  /  caoxiangyong@mail.xjtu.edu.cn).
