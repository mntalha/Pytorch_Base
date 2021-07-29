# Pytorch_Base__
The purpose is to build base code for user using Pytorch library 

### How To
Since I am coming from C and C++ background , to see "main" files make me happy :)

Therefore , I seperate the related class and files into folders such as 

[required_classes](https://github.com/mntalha/Pytorch_Base/blob/main/dataset) auxiliry classses 

[dataset](https://github.com/mntalha/Pytorch_Base/blob/main/main_function)  contains raw and processed images.

[main_function](https://github.com/mntalha/Pytorch_Base/blob/main/required_classes) main python file that will be runned.


# Requirements

- python3.x
- torch
- numpy
- time
- torch.utils.data
- torchvision.datasets
- shutil

# Example Usage
```python
    #Dataset
    TRAIN_PATH="C:/Users/talha/Desktop/Dataset/train"
    TEST_PATH="C:/Users/talha/Desktop/Dataset/test"
    VALIDATION_PATH="C:/Users/talha/Desktop/Dataset/validation"

    model = Model()
    my=ModelTrain()

    batch_size = 64 
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
```

**for any problem , don't hesitate to contact me from** [Linkedin](https://www.linkedin.com/in/mntalhakilic/) :+1: :+1:



