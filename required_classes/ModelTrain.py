# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:43:09 2021

@author: talha
"""

#load MyCustomDataset class .
from MyCustomDataset import *
from Model import *
import torch
import time
import math
import numpy as np
import torch.optim as optim




class ModelTrain:

    loss_values = {
        'train_every_iteration' : [] ,
        'train_every_epoch' : [] ,
        'validation_every_iteration' : [] ,
        'validation_every_epoch' : []
        }
    accuracy_values = {
        'train_every_iteration' : [] ,
        'train_every_epoch' : [] ,
        'validation_every_iteration' : [] ,
        'validation_every_epoch' : []
        }

    use_gpu = None
    model = None
    criteria = None
    optimizer = None
    batch_size = None
    epoch = None
    dataset = None
    loader = None
    mydataset = None
    logger = None
    debugger = None


    def __init__(self):
        pass

    def dataset_load(self,TRAIN_PATH,TEST_PATH,VALIDATION_PATH,batch_size):
        
        self.batch_size = batch_size

        self.mydataset=MyCustomDataset(TRAIN_PATH,TEST_PATH,VALIDATION_PATH,batch_size)

        self.dataset,self.loader = self.mydataset.execute()
        
    def dataset_info(self,dataset_type):
        
        if  self.mydataset == None:
            assert False,"First run 'dataset_load(self,TRAIN_PATH,TEST_PATH,VALIDATION_PATH,batch_size)' function "
        #Train , Test and Validation Datasize
        print(f"Dataset contains {len(self.mydataset.getClasses())} classes which are {self.mydataset.getClasses()}")
        
        if dataset_type == "train" or  dataset_type == "all":
            #Train
            print("-"*30)
            print(f"  Train Image Number : {len(self.mydataset.dataset['train'])}")
            print(f"  Batch size         : {self.batch_size} ")
            print(f"  1 epoch requires {math.ceil(len(self.mydataset.dataset['train'])/self.batch_size)} loop")
            print("-"*30)

        if dataset_type == "validation" or  dataset_type == "all":
            #Train
            print(f"  Validation Image Number : {len(self.mydataset.dataset['validation'])}")
            print(f"  Batch size         : {self.batch_size} ")
            print(f"  1 epoch requires {math.ceil(len(self.mydataset.dataset['validation'])/self.batch_size)} loop")
            print("-"*30)

        if dataset_type == "test" or  dataset_type == "all":
            #Train
            print(f"  Test Image Number : {len(self.mydataset.dataset['test'])}")
            print(f"  Batch size         : {self.batch_size} ")
            print(f"  1 epoch requires {math.ceil(len(self.mydataset.dataset['test'])/self.batch_size)} loop")
            print("-"*30)

    def GPU_Usage(self,use=False):
        self.use_gpu=use

    def train(self,model_,criteria_,n_epoch_,optimizer_):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True # speed up

        if self.use_gpu == True and device.type == "cpu":
            print("GPU is not supported --->")

        self.model = model_
        self.criteria = criteria_
        self.epoch = n_epoch_
        self.optimizer = optimizer_

        #Model to cpu or cuda(GPU)
        self.model.to(device)

        start = time.time()



        for epoch in range(self.epoch):
            #Debug element
            Debug = f" ********** { {epoch} } EPOCH"
            self.logger.debug(Debug)
            self.debugger(Debug)
            # make it 0 in each epoch
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0

            #start train gradient calculation active
            self.model.train()


            for ix, data in enumerate(self.loader["train"]):
                
                img, label = data

                img = img.to(device)
                label = label.to(device)

                #gradient refresh ,clear the gradients of all optimized variables
                for param in model.parameters(): # Daha hızlı lduğu söyleniyor
                    param.grad = None 
                #self.optimizer.zero_grad()


                #y_pred is form of probability
                with torch.cuda.amp.autocast(): # hız artıyor aa basarı dusuyır
                    y_pred = self.model(img)
                    # output is float16 because linear layers autocast to float16.
                    assert y_pred.dtype is torch.float16
                    #Training Loss calculation part
                    loss = self.criteria(y_pred, label)
                    # loss is float32 because mse_loss layers autocast to float32.
                    assert loss.dtype is torch.float32
                    loss.backward()
                    
                self.optimizer.step()

                #On each batch it sum up.
                train_loss += loss.item()* img.size(0)

                #Accuracy calculation part
                label = label.cpu()
                y_pred = y_pred.cpu()
                acc = bul1(y_pred,label) * img.size(0)
                
                #On each batch it sum up.
                train_acc += acc
                
            
                #Iteration loss and accuracy
                self.loss_values['train_every_iteration'].append(loss.item()* img.size(0))
                self.accuracy_values['train_every_iteration'].append(acc)

            #Epoch losses and accuracy
            train_loss = train_loss / len(self.loader["train"].sampler)
            self.loss_values['train_every_epoch'].append(train_loss)

            train_acc = train_acc / (len(self.loader["train"].dataset))
            self.accuracy_values['train_every_epoch'].append(train_acc)
            
            #Debug
            Debug = "train_acc= "+str(train_acc)+" train_loss= "+str(train_loss)
            self.logger.debug(Debug)
            self.debugger(Debug)

            # stopp = time.time()
            # print(f' ********** { {epoch} } epoch training time is {(stopp-startt)} seconds.')


            if epoch % 3 == 0:
                #start evaluation gradient calculation passive
                #"model.train()" and "model.eval()" activates and deactivates Dropout and BatchNorm, so it is quite important. "with torch.no_grad()" only deactivates gradient calculations, but doesn't turn off Dropout and BatchNorm. Your model accuracy will therefore be lower if you don't use model.eval() when evaluating the model.
                self.model.eval()

                with torch.no_grad():
                    # Measure the performance in validation set.
                    for ix2, data2 in enumerate(self.loader["test"]):
                                                
                        img, label = data2
                        if self.use_gpu:
                            img = img.cuda()
                            label = label.cuda()


                        label = ww[label].float()
                        #y_pred is form of probability
                        y_pred = self.model(img)

                        #Validation Loss calculation part
                        loss = self.criteria(y_pred, label)

                        #On each batch it sum up.
                        valid_loss += loss.item()* img.size(0)

                        #Accuracy calculation part
                        label = label.cpu()
                        y_pred = y_pred.cpu()
                        acc = bul2(y_pred,label) * img.size(0)
                        

                        #On each batch it sum up.
                        valid_acc += acc
                        

                        #Iteration loss and accuracy
                        self.loss_values['validation_every_iteration'].append(loss.item()* img.size(0))
                        self.accuracy_values['validation_every_iteration'].append(acc)


                #Epoch losses and accuracy
                valid_loss = valid_loss / (len(self.loader["test"].sampler))
                self.loss_values['validation_every_epoch'].append(valid_loss)

                valid_acc = valid_acc / (len(self.loader["test"].dataset))
                self.accuracy_values['validation_every_epoch'].append(valid_acc)

                #Debug
                Debug = "valid_acc= "+str(valid_acc)+" valid_loss= "+str(valid_loss)
                self.logger.debug(Debug)
                self.debugger(Debug)

                self.logger.debug ("confusion Matrix:")
                self.debugger("confusion Matrix:")

                Debug = str(con)
                self.logger.debug (Debug)
                self.debugger (Debug)
                
                Debug = "---------------------------------------------------------------------"
                self.logger.debug (Debug)
                self.debugger (Debug)
                
               
                
        end = time.time()
        
        Debug = 'Total Elapsed time is %f seconds.' % (end - start)
        self.logger.debug(Debug)
        self.debugger(Debug)

        return self.loss_values,self.accuracy_values

    def result(self):
        pass
    def show_weights(self):
        pass
    
  

if __name__ == "__main__":
    #Dataset
    TRAIN_PATH="C:/Users/talha/Desktop/Dataset/train"
    TEST_PATH="C:/Users/talha/Desktop/Dataset/test"
    VALIDATION_PATH="C:/Users/talha/Desktop/Dataset/validation"

    model = Model()
    my=ModelTrain()

    batch_size = 64 #en iyisi
    my.dataset_load(TRAIN_PATH, TEST_PATH, VALIDATION_PATH, batch_size)
    my.GPU_Usage(True)
    print(my.use_gpu)
    
    #Dataset info
    my.dataset_info("all")
    
    #logging
    import sys
    sys.path.insert(1, 'C:/Users/talha/Desktop/Github_sources/pycodes')
    import tk_logging
    tk = tk_logging.Tk_logging("ModelTrain","log.txt","DEBUG")
    loggerr = tk.logger
    my.logger = loggerr
    
    #Debugger
    from icecream import ic
    icdebugger = ic
    icdebugger.configureOutput(prefix='') # writes Debug before each printing
    icdebugger.enable()
    my.debugger = icdebugger
    

    learning_rate = 1e-3
    n_epochs =  21
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate,weight_decay=1e-5) #,,weight_decay=1e-5
    criteria = nn.MSELoss()
    
    loss_values,accuracy_values=my.train(model, criteria, n_epochs, optimizer)
