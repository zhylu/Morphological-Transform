# -*- coding: utf-8 -*-
import os
#os.chdir('')
print(os.getcwd())
#%%
import h5py
import numpy

FILENAME = "stl10.mat"
data = h5py.File(FILENAME,"r")

print(data.keys())
#%%
import torch

unlabeledata,traindata,testdata =  torch.from_numpy(numpy.array(data['unlabeled'],dtype=numpy.float32)).permute(0,1,3,2),\
                                   torch.from_numpy(numpy.array(data['traindata'],dtype=numpy.float32)).permute(0,1,3,2),\
                                   torch.from_numpy(numpy.array(data['testdata'],dtype=numpy.float32)).permute(0,1,3,2) 
                     
trainlabels,testlabels = numpy.array(data['trainlabel']).T,\
                         numpy.array(data['testlabel']).T
                                                
print(unlabeledata.shape,traindata.shape,testdata.shape,trainlabels.shape,testlabels.shape)
#%%
sd = 96
from torchvision import transforms as tf
tfm = tf.Compose([tf.ToPILImage(),
                  tf.ToTensor(),                    
                  tf.Normalize([0.4406, 0.4273, 0.3858], [0.2687, 0.2613, 0.2685])])
tfm_o = tf.Compose([tf.ToPILImage(),
                  tf.ToTensor()])
tfmaug_a = tf.Compose([tf.ToPILImage(),
                  tf.RandomCrop(sd,int(sd/8)),
                  tf.ColorJitter(0.4,0.4,0.4,0.1),
                  tf.RandomGrayscale(p=0.2),
                  tf.ToTensor(),
                  tf.Normalize([0.4406, 0.4273, 0.3858], [0.2687, 0.2613, 0.2685])])
tfmaug_p = tf.Compose([tf.ToPILImage(),
                  tf.RandomCrop(int(sd/2),int(sd/16)),
                  tf.ColorJitter(0.4,0.4,0.4,0.1),
                  tf.RandomGrayscale(p=0.2),
                  tf.ToTensor(),
                  tf.Normalize([0.4406, 0.4273, 0.3858], [0.2687, 0.2613, 0.2685])])
#%%

import torch.nn as nn
from resnet_FMA import resnet34
from resnet_FMA_FC import resnet18
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,model):
        
        super(Net, self).__init__()
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        self.fc1 = nn.Linear(512,1024)
        self.fc2 = nn.Linear(1024,512)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
     
    def forward(self, x):

        x = self.resnet(x)
        x = x.squeeze()
        y = F.normalize(x)
        x = self.fc1(y)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.normalize(x)
        return x,y

net = Net(resnet18(pretrained=False)).cuda()

import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.001)
#%%
samplek = 64
mb = 100000//samplek
n = 3
a = 2

for epoch in range(60):
   
    print(epoch)
    
    loss_sum = 0
    ids=torch.randperm(100000)

    for i in range(mb):

        batch_ids = ids[i*samplek:(i+1)*samplek]
        batch_data = unlabeledata[batch_ids]

        anchor = torch.zeros(samplek,3,sd,sd)
        positive = torch.zeros(samplek*n,3,int(sd/2),int(sd/2))


        for k in range(samplek):

            anchor[k] = tfmaug_a(batch_data[k])
            
            for j in range(n):
                positive[j*samplek+k] = tfmaug_p(batch_data[k])

        
        net.train()        
        
        gf_a,_= net(anchor.cuda())
        gf_p,_= net(positive.cuda())      
        gf_products = gf_a.mm(gf_p.t())
        gf_p_d = gf_products.repeat(n,1).diag().view(n,samplek).t().repeat_interleave(samplek,dim=1)
        loss = torch.log1p(((gf_products - gf_p_d)*a).exp().sum(1)-n).mean()
        
        net.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(loss_sum/mb)
    
    if epoch > 50 :

        bs = 250
        d = 512
        traindatafe = numpy.zeros((5000,d))
        testdatafe = numpy.zeros((8000,d))
        with torch.no_grad():
            
            
            for i in range(20):
         
                inputsaug = torch.zeros(bs,3,sd,sd) 
                for k in range(bs):
                    inputsaug[k]=tfm(traindata[i*bs+k])
                inputsaug = inputsaug.cuda()
                net.eval()
                _,outputs = net(inputsaug)
                traindatafe[i*bs:bs+i*bs] = outputs.cpu().numpy()
                
            for i in range(32):
  
                inputsaug = torch.zeros(bs,3,sd,sd) 
                for k in range(bs):
                    inputsaug[k]=tfm(testdata[i*bs+k])
                inputsaug = inputsaug.cuda()
        
                net.eval()
                _,outputs = net(inputsaug)
                testdatafe[i*bs:bs+i*bs] = outputs.cpu().numpy()

    
            from sklearn.linear_model import LogisticRegression 
            from sklearn.preprocessing import MinMaxScaler
            
            classifier= LogisticRegression(solver='saga', multi_class='multinomial', tol=.01, C=10., max_iter=100)    
            scaler = MinMaxScaler()        
            traindatafe = scaler.fit_transform(traindatafe)
            testdatafe = scaler.transform(testdatafe)
            classifier.fit(traindatafe, trainlabels.ravel())
            acc= classifier.score(testdatafe, testlabels.ravel())
            print(acc)
