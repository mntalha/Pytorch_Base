# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:40:59 2021

@author: talha
"""

#Dataloaders

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os
import math


class MyCustomDataset:
            
    def __init__(self,TRAIN_PATH,TEST_PATH,VALIDATION_PATH,BATH_SIZE):
        self.train_path = TRAIN_PATH
        self.test_path = TEST_PATH
        self.validation_path = VALIDATION_PATH
        self.batch_size = BATH_SIZE
        self.loader = {}
        self.dataset = {}
        
    def isValid(self):
        #if these path is exist return true
        if os.listdir(self.train_path):
            return True
    def getClasses(self):  
        return os.listdir(self.train_path)
    
    def execute(self):
        train_transforms = transforms.Compose([
            transforms.Resize([128,128]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Grayscale(num_output_channels=1)
        ])
        val_transforms = transforms.Compose([
            transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Grayscale(num_output_channels=1)
        ])
        test_transforms = transforms.Compose([
            transforms.Resize([128,128]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Grayscale(num_output_channels=1)
        ])
        
        
        train_dataset = ImageFolder(self.train_path, train_transforms)
        val_dataset = ImageFolder(self.validation_path, val_transforms)
        test_dataset = ImageFolder(self.test_path, test_transforms)
        
        
        #Add to the system

        self.dataset["train"] = train_dataset
        self.dataset["validation"] = val_dataset
        self.dataset["test"] = test_dataset
        
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=12, shuffle=True,pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, num_workers=12, shuffle=False,pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, num_workers=12, shuffle=False,pin_memory=True)
        
        self.loader["train"] = train_loader
        self.loader["validation"] = val_loader
        self.loader["test"] = test_loader

        
        return self.dataset,self.loader
        

if __name__ == "__main__":
    
    TRAIN_PATH="C:/Users/talha/Desktop/Dataset/train"
    TEST_PATH="C:/Users/talha/Desktop/Dataset/test"
    VALIDATION_PATH="C:/Users/talha/Desktop/Dataset/validation"
    batch_size=64
    mydataset=MyCustomDataset(TRAIN_PATH,TEST_PATH,VALIDATION_PATH,batch_size)
    mydataset.execute()
    print(mydataset)
    print(f"Dataset contains {len(mydataset.getClasses())} classes which are {mydataset.getClasses()}")
    print(f"  Train Image Number : {len(mydataset.dataset['train'])} ")
    print(f"  Batch size         : {batch_size} ")
    print(f"  1 epoch requires {math.ceil(len(mydataset.dataset['train'])/batch_size)} loop")
    

 
            
        
        