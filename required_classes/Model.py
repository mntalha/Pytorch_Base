# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 12:59:27 2021

@author: talha
"""

import torch.nn as nn
import torch.nn.functional as F
import torch 
class Model(nn.Module):
    
    #List of Modules
    conv = []
    dropout = []
    batch = []
    maxpooling = []
    
    
    def __init__(self):
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        #dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        super(Model, self).__init__()
        
        keep_rate=0.75
        
      
        #CNV
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=50,kernel_size=5,stride=1,padding="same",bias=False)
        self.conv.append(self.conv1)
        
        self.conv2 = nn.Conv2d(in_channels=50,out_channels=50,kernel_size=5,stride=1,padding="same",bias=False)
        self.conv.append(self.conv2)
        
        self.conv3 = nn.Conv2d(in_channels=50,out_channels=50,kernel_size=5,stride=1,padding="same",bias=False)
        self.conv.append(self.conv3)
        
        self.conv4 = nn.Conv2d(in_channels=50,out_channels=50,kernel_size=5,stride=1,padding="same",bias=False)
        self.conv.append(self.conv4)
        
        self.conv5 = nn.Conv2d(in_channels=50,out_channels=50,kernel_size=5,stride=1,padding="same",bias=False)
        self.conv.append(self.conv5)
        
        self.conv6 = nn.Conv2d(in_channels=50,out_channels=16,kernel_size=4,stride=1,padding="valid",bias=False)
        self.conv.append(self.conv6)


        #Dropout
        
        self.dropout1 = nn.Dropout2d(1-keep_rate)
        self.dropout.append(self.dropout1)
        
        self.dropout2 = nn.Dropout2d(1-keep_rate)
        self.dropout.append(self.dropout2)
        
        self.dropout3 = nn.Dropout2d(1-keep_rate)
        self.dropout.append(self.dropout3)
        
        self.dropout4 = nn.Dropout2d(1-keep_rate)
        self.dropout.append(self.dropout4)
        
        self.dropout5 = nn.Dropout2d(1-keep_rate)
        self.dropout.append(self.dropout5)
        
        self.dropout6 = nn.Dropout2d(1-keep_rate)
        self.dropout.append(self.dropout6)
        
        #BATCH 
        self.batch1=nn.BatchNorm2d(50) 
        self.batch.append(self.batch1)
        
        self.batch2=nn.BatchNorm2d(50)
        self.batch.append(self.batch2)


        self.batch3=nn.BatchNorm2d(50)
        self.batch.append(self.batch3)

                
        self.batch4=nn.BatchNorm2d(50)
        self.batch.append(self.batch4)
                        
        self.batch5=nn.BatchNorm2d(50)
        self.batch.append(self.batch5)
                               
        self.batch6=nn.BatchNorm2d(16)
        self.batch.append(self.batch6)



        #MAXPOOLING
        self.maxpooling1= nn.MaxPool2d(2)
        self.maxpooling.append(self.maxpooling1)

        self.maxpooling2= nn.MaxPool2d(2)
        self.maxpooling.append(self.maxpooling2)

        self.maxpooling3= nn.MaxPool2d(2)
        self.maxpooling.append(self.maxpooling3)

        self.maxpooling4= nn.MaxPool2d(2)
        self.maxpooling.append(self.maxpooling4)

        self.maxpooling5= nn.MaxPool2d(2)
        self.maxpooling.append(self.maxpooling5)


    def forward(self, x):
        
        
        for i in range(6):
            if i  == 5: # Last layer
                #cnv
                x = self.conv[i](x)
                #print(f"{i}**",x.shape)
                
                #relu
                x = F.relu(x)
                #print(f"{i}**",x.shape)
                
               
                #batch
                x = self.batch[i](x)
                #print(f"{i}**",x.shape)
                
                
               
                #Flatten
                out = torch.flatten(x,1)
                #print(f"{i}**",x.shape)
            else:
                #cnv
                x = self.conv[i](x)
                #print(f"{i}**",x.shape)
                
                 #relu
                x = F.relu(x)
                #print(f"{i}**",x.shape)
                
                 #batch
                x = self.batch[i](x)
                #print(f"{i}**",x.shape)
                
                 #pooling
                x = self.maxpooling[i](x)
                #print(f"{i}**",x.shape)
                
               
                #dropout
                x = self.dropout[i](x)
                #print(f"{i}**",x.shape)
            
        
        return out

            
if __name__ == "__main__":
    model = Model()
    
