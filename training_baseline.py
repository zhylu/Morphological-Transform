# -*- coding: utf-8 -*-
import torch
torch.cuda.empty_cache()
import time
start = time.time()
import os
os.chdir('D:/gdfn')
print(os.getcwd())
import torch
torch.manual_seed(1)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import h5py



FILENAME = "STL10.mat"
# use the pre-processed dataset
mean = [0.4467,0.4398,0.4067]
std = [0.2586,0.2548,0.2696]
c = 10
ep = 80

bs = 64
test_bs = 500
p = 0.1
p_dilation = 0.5*p
mb = 78



data = h5py.File(FILENAME,"r")

print(data.keys())

#%%
import torch
import numpy
traindatan,trainlabel,testdatan,testlabel = numpy.array(data['traindata']),\
                                            numpy.array(data['trainlabel']),\
                                            numpy.array(data['testdata']),\
                                            numpy.array(data['testlabel'])


            

traindata,testdata,trainlabel,testlabel = torch.from_numpy(traindatan).permute(0,1,3,2),\
                                            torch.from_numpy(testdatan).permute(0,1,3,2),\
                                            torch.from_numpy(trainlabel).squeeze().long(),\
                                            torch.from_numpy(testlabel).squeeze().long()
                                              



print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)

traindata_size = int(traindata.size(0))
testdata_size = int(testdata.size(0))
#%%

import matplotlib.pyplot as plt
n = torch.randint(0,traindata_size,(1,1)).squeeze()
print(n)
test = traindata[n].permute(1,2,0).float()/255
plt.imshow(test)
plt.show()
#%%
sd = 224

from torchvision import transforms as tf
import torch.nn as nn

tfmtest = nn.Sequential(
                     
                      tf.Resize(sd),                     
                      tf.ConvertImageDtype(torch.float),
                      tf.Normalize(mean, std)
                      )
tfmtrain = nn.Sequential( 

                       
                       tf.Resize(sd),
                       tf.RandomCrop(sd,12),   
                       tf.RandomHorizontalFlip(),
                       tf.ConvertImageDtype(torch.float),
                       tf.Normalize(mean, std),
                        )



#%%
acc_sum = 0
for t in range(3):
# training for 3 times 
    print(t)
    from torchvision import models
    d = 512
    
    
    net = models.resnet18()
    net.fc = nn.Linear(d, c)
    
    
    device = torch.device("cuda:0")
    net.to(device)
    import torch.optim as optim
    
    optimizer = optim.SGD(net.parameters(), lr = 0.1,weight_decay=0.0001,momentum=0.9)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = ep, eta_min = 1e-4)
    
    lossfn = nn.CrossEntropyLoss()
    #%%
    
    '''
    PATH = "stl10_ck"
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #loss = checkpoint['loss']
    print("checkpoint loaded")
    '''
    #%%
    from torch.utils.data import TensorDataset,DataLoader
    
    train = TensorDataset(traindata, trainlabel)
    test = TensorDataset(testdata, testlabel)
     

    trainld = DataLoader(train, batch_size=bs, shuffle=True, pin_memory=True, drop_last=True)
    testld = DataLoader(test, batch_size=test_bs, pin_memory=True)
    
    
    #%%
    from torch.cuda.amp import autocast,GradScaler
    scaler = GradScaler()



    for epoch in range(ep):
    
    
        
        
        net.train()
        loss_sum = 0
        for  data in trainld:
            
            inputs, labels = data
   
            inputs_aug = tfmtrain(inputs.cuda(non_blocking=True))
            optimizer.zero_grad()
            
            with autocast():
    
                outputs = net(inputs_aug)
                loss = lossfn(outputs, labels.cuda(non_blocking=True))
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            # print(loss.item())
      
        scheduler.step()
        del inputs,inputs_aug,outputs,labels
        torch.cuda.empty_cache()
        
        print(epoch,"%f"%(loss_sum/mb),'%f'%optimizer.param_groups[0]['lr'])
    
    
    

        if (epoch+1)%10 == 0 :
              
            correct = 0
            net.eval()
            with torch.no_grad():
                for data in testld:
                    inputs, labels = data
        
                    inputs_aug = tfmtest(inputs.cuda(non_blocking=True))
        
                    outputs = net(inputs_aug)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    correct += (predicted == labels.cuda(non_blocking=True)).sum().item()
                acc = correct/testdata_size
                print(acc)
                del inputs,inputs_aug,outputs,predicted
                torch.cuda.empty_cache()
            if epoch == ep-1:
                acc_sum += acc
          

      

print(acc_sum/3)

#%%


#%%

  
  
#%%   
'''
PATH = "stl10_ck"

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
            }, PATH)
'''

print(time.ctime(start))
print(time.ctime())
