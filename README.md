# Supervised Hyperspectral Image Classification Package (SHIP)

This is an open package (called **SHIP**) for supervised hyperspectral image classification task. Besides, this repository guarantees you to reproduce the results reported in the paper:
 - [Supervised Hyperspectral Image Classification: Benchmark and State of the Art](https://ieeexplore.ieee.org/abstract/document/8271995) (This work is reviewing now and so the link is wrong.)

If you think it helpful, we would appreciate if you cite the papers in your work. 

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

1. To choose the best classifier for one given feature (also reproduce the result in Exp.1), execute

        python demo_Exp1.py

   According to your needs, many parameters can be set in demo_Exp1.py, such as dataset, feature, classifieris, train_size, repeat_num, model_selection, isdraw. More detailed comments of these parameters can be found in SHSIC function. 
   
   This package supports 7 datasets, 6 features (5 classical features and 1 deep feature), 9 classifiers, model selection for each classifier, post-processing classification map and drawing classification map for each method . 

2. To reproduce the results in Exp.2, execute

        python demo_Exp2.py


   According to your needs, many parameters can be set in demo_Exp2.py.


## Contact:
This package is still developing and this is the first version. In the next step, we prepare to embed the feature extraction methods into this package, thus it can implement the feature extraction (this package only provide some pre-extracted features by some offline code). 

For the package of this version, we hope more reserachers in this field can provide your extracted feature data to me. Welcome to contact me (Xiangyong Cao:   caoxiangyong45@gmail.com  /  caoxiangyong@mail.xjtu.edu.cn).
