# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:28:37 2021

@author: talha
"""

import os
import numpy as np
import shutil

dataset = "COVID-19_Radiography_Dataset"

root_dir        = os.getcwd()
classes         = os.listdir(dataset)

test_ratio=0.1
validation_ratio = 0.1

shutil.rmtree(root_dir+'/train/', ignore_errors=True)
shutil.rmtree(root_dir+'/validation/', ignore_errors=True)
shutil.rmtree(root_dir+'/test/', ignore_errors=True)


for cls in classes:
    os.makedirs(root_dir+'/train/'+cls)
    os.makedirs(root_dir+'/validation/'+cls)
    os.makedirs(root_dir+'/test/'+cls)
    
for cls in classes:
    #ouputs the name of images inside the files.
    imgs=os.listdir(root_dir+"/"+dataset+"/"+cls)
    
    #shuffle names
    np.random.shuffle(imgs)
    
    #split the files
    train_FileNames=imgs[:int(len(imgs)*(1-test_ratio-validation_ratio))]
    validation_FileNames=imgs[int(len(imgs)*(1-test_ratio-validation_ratio)):int(len(imgs)*(1-test_ratio))]
    test_FileNames=imgs[int(len(imgs)*(1-test_ratio)):]
    
    #add path to image names
    train_FileNames=[root_dir+"/"+dataset+"/"+cls+"/"+name for name in train_FileNames]
    validation_FileNames=[root_dir+"/"+dataset+"/"+cls+"/"+name for name in validation_FileNames]
    test_FileNames=[root_dir+"/"+dataset+"/"+cls+"/"+name for name in test_FileNames]
    print(cls," Splitting")
    print("Total Images:",len(imgs))
    print("Train",len(train_FileNames))
    print("Validation",len(validation_FileNames))
    print("Test",len(test_FileNames))
    
    #copy images
    for name in train_FileNames:
        shutil.copy(name,root_dir+"/train/"+cls)
        
    for name in validation_FileNames:
        shutil.copy(name,root_dir+"/validation/"+cls)
        
    for name in test_FileNames:
        shutil.copy(name,root_dir+"/test/"+cls)

