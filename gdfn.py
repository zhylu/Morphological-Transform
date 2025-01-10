# -*- coding: utf-8 -*-
import torchvision.transforms.functional as F
import torchvision.transforms as tf
import torch

re = tf.RandomErasing(p=1)


class gdfn_img(torch.nn.Module):
    
# this method is used for single image processing in GPU
    
    def __init__(self,kernel_size = 3,dilation='True'):
        # 'True' for dilation and 'False' for erosion 
        super().__init__()
        self.ks = kernel_size
        self.dilation = dilation
    def forward(self, img):

        ks2 = self.ks**2
        offset = int((self.ks-1)/2)     
        channel,img_size,_ = img.size()

        img_p = F.pad(img.float(), [offset,offset,offset,offset])

        img_p = img_p + torch.rand_like(img_p)
 
        img_e = torch.zeros(channel,img_size,img_size,self.ks,self.ks)
        for i in range(ks2):
            x,y = divmod(i,self.ks)

            img_e[:,:,:,x,y] = img_p[:,x:x+img_size,y:y+img_size];

        Std = torch.std(img_e,[3,4],unbiased=True,keepdim=True)

        Mean = torch.mean(img_e,[3,4],keepdim=True)

        img_3d = (-(img_e -Mean).pow(2).div(2*Std.pow(2))).exp().min(0,keepdim=True)[0] #compute the 3d fuzzy number

  
        img_r = img_e.mul(img_3d).sum([0,3,4],keepdim=True).div(img_3d.sum([3,4],keepdim=True)).squeeze()
  
        img_r_p = F.pad(img_r, [offset,offset,offset,offset])
  
        img_r_e = torch.zeros(img_size,img_size,ks2).cuda();
        for i in range(ks2):
            x,y = divmod(i,self.ks)
            img_r_e[:,:,i] = img_r_p[x:x+img_size,y:y+img_size];

        ids = torch.sort(img_r_e,dim=2)[1]
  
        #ids = ids[:,:,0] #erosion
        ids = ids[:,:,-1] if self.dilation else ids[:,:,0] #dilation
        
      
        img_3d_s = img_3d.reshape([img_size,img_size,ks2])
  
        img_3d_mod = img_3d_s.gather(2,ids.unsqueeze(2)).squeeze()  #sorting
     
        img_3d_mod_p = F.pad(img_3d_mod, [offset,offset,offset,offset])

        img_3d_mod_e = torch.zeros(img_size,img_size,self.ks,self.ks)
        for i in range(ks2):
            x,y = divmod(i,self.ks)
            img_3d_mod_e[:,:,x,y] = img_3d_mod_p[x:x+img_size,y:y+img_size];
        
        img_3d_mod_e = img_3d_mod_e.unsqueeze(0)
 
        img_mod = img_e.mul(img_3d_mod_e).sum([3,4],keepdim=True).div(img_3d_mod_e.sum([3,4],keepdim=True)).squeeze().int()  # compute the original image
     
         
        return img_mod
     
class gdfn_batch_mix(torch.nn.Module):
    
# this method is used for the mini-batch in model training.
    
    def __init__(self,p = 0.1,p_mix = 0.5,kernel_size = 3,device = 'cuda'):
        
        super().__init__()
        self.p = p
        self.p_mix = p_mix
        self.ks = kernel_size
        self.device = device
    def forward(self, img):
              
        ks2 = self.ks**2
        offset = int((self.ks-1)/2)     
        batch_size,channel,img_size,_ = img.size()
        
        ids_sel = torch.rand(batch_size,device=self.device).lt(self.p)
  
        batch_size = ids_sel.sum()
        

      
        img = img.float()
        img_p = F.pad(img[ids_sel], [offset,offset,offset,offset])
        
 

        
        img_p = img_p + torch.rand_like(img_p)
 
        img_e = torch.zeros(batch_size,channel,img_size,img_size,self.ks,self.ks,device=self.device)
        for i in range(ks2):
            x,y = divmod(i,self.ks)
         
            img_e[:,:,:,:,x,y] = img_p[:,:,x:x+img_size,y:y+img_size];
        Std = torch.std(img_e,[4,5],unbiased=True,keepdim=True)
  
        Mean = torch.mean(img_e,[4,5],keepdim=True)

        img_3d = (-(img_e -Mean).pow(2).div(2*Std.pow(2))).exp().min(1,keepdim=True)[0]
      
  
        img_r = img_e.mul(img_3d).sum([1,4,5],keepdim=True).div(img_3d.sum([4,5],keepdim=True)).squeeze(1,4,5)
        
        img_r_p = F.pad(img_r, [offset,offset,offset,offset])
  
        img_r_e = torch.zeros(batch_size,img_size,img_size,ks2,device=self.device);
  
        for i in range(ks2):
            x,y = divmod(i,self.ks)

            img_r_e[:,:,:,i] = img_r_p[:,x:x+img_size,y:y+img_size];

        ids = torch.sort(img_r_e,dim=3)[1]
  
        ids_mix = ids[:,:,:,0]
        ids_dilation = ids[:,:,:,-1]
        ids_dilation_sel = torch.rand(batch_size,device=self.device).lt(self.p_mix)
        ids_mix[ids_dilation_sel] = ids_dilation[ids_dilation_sel]
        
        
        img_3d_s = img_3d.reshape([batch_size,img_size,img_size,ks2])
        
        img_3d_mod = img_3d_s.gather(3,ids_mix.unsqueeze(3)).squeeze(3)
      
        img_3d_mod_p = F.pad(img_3d_mod, [offset,offset,offset,offset])
   
        img_3d_mod_e = torch.zeros(batch_size,img_size,img_size,self.ks,self.ks,device=self.device)
        for i in range(ks2):
            x,y = divmod(i,self.ks)
            img_3d_mod_e[:,:,:,x,y] = img_3d_mod_p[:,x:x+img_size,y:y+img_size];
        
        img_3d_mod_e = img_3d_mod_e.unsqueeze(1)
     
        img_mod = img_e.mul(img_3d_mod_e).sum([4,5],keepdim=True).div(img_3d_mod_e.sum([4,5],keepdim=True)).squeeze()
        img[ids_sel] = img_mod 
         
        return img.byte()
            
class gdfn_region_batch(torch.nn.Module):
    
# This method is depreicated
    
    def __init__(self,p = 0.5,p_mix = 0.5,kernel_size = 3):
        
        super().__init__()
        self.p = p
        self.p_mix = p_mix
        self.ks = kernel_size
    def forward(self, img):
        torch.set_default_device('cuda')
        ks2 = self.ks**2
        offset = int((self.ks-1)/2)     
        batch_size,channel,img_size,_ = img.size()
        
        ids_sel = torch.rand(batch_size).lt(self.p)
  
        batch_size = ids_sel.sum()
    
      
        img_sel = img[ids_sel].float()
        img_p = F.pad(img_sel, [offset,offset,offset,offset])

        img_p = img_p + torch.rand_like(img_p)
 
        img_e = torch.zeros(batch_size,channel,img_size,img_size,self.ks,self.ks)
        for i in range(ks2):
            x,y = divmod(i,self.ks)
            img_e[:,:,:,:,x,y] = img_p[:,:,x:x+img_size,y:y+img_size];
        Std = torch.std(img_e,[4,5],unbiased=True,keepdim=True)
  
        Mean = torch.mean(img_e,[4,5],keepdim=True)

        img_3d = (-(img_e -Mean).pow(2).div(2*Std.pow(2))).exp().min(1,keepdim=True)[0]
      
  
        img_r = img_e.mul(img_3d).sum([1,4,5],keepdim=True).div(img_3d.sum([4,5],keepdim=True)).squeeze()
        
        img_r_p = F.pad(img_r, [offset,offset,offset,offset])
  
        img_r_e = torch.zeros(batch_size,img_size,img_size,ks2);
        for i in range(ks2):
            x,y = divmod(i,self.ks)
            img_r_e[:,:,:,i] = img_r_p[:,x:x+img_size,y:y+img_size];
     
        ids = torch.sort(img_r_e,dim=3)[1]
  
        ids_mix = ids[:,:,:,0]
        ids_dilation = ids[:,:,:,-1]
        ids_dilation_sel = torch.rand(batch_size).lt(self.p_mix)
      
        ids_mix[ids_dilation_sel] = ids_dilation[ids_dilation_sel]
        img_3d_s = img_3d.reshape([batch_size,img_size,img_size,ks2])
  
        img_3d_mod = img_3d_s.gather(3,ids_mix.unsqueeze(3)).squeeze()
        
        img_3d_mod_p = F.pad(img_3d_mod, [offset,offset,offset,offset])
   
        img_3d_mod_e = torch.zeros(batch_size,img_size,img_size,self.ks,self.ks)
        for i in range(ks2):
            x,y = divmod(i,self.ks)
            img_3d_mod_e[:,:,:,x,y] = img_3d_mod_p[:,x:x+img_size,y:y+img_size];
        
        img_3d_mod_e = img_3d_mod_e.unsqueeze(1)
     
        img_mod = img_e.mul(img_3d_mod_e).sum([4,5],keepdim=True).div(img_3d_mod_e.sum([4,5],keepdim=True)).squeeze()
     
        
                
        re_mask = re(torch.ones_like(img_mod).float())
        re_mask_reverse = 1-re_mask
    
        
        img[ids_sel] = (img_mod.mul(re_mask_reverse)+ img_sel.mul(re_mask)).byte()
        
        
        return img     
           
class gdfn_batch_random(torch.nn.Module):
    
# This method is simplifed version of the gdfn_batch with less time cost
    
    def __init__(self,p = 0.1,kernel_size = 3,device = 'cuda'):
        
        super().__init__()
        self.p = p
        self.ks = kernel_size
        self.device = device
    def forward(self, img):
        

        
      
        ks2 = self.ks**2
        offset = int((self.ks-1)/2)     
        batch_size,channel,img_size,_ = img.size()
        
        ids_sel = torch.rand(batch_size,device=self.device).lt(self.p)
  
        batch_size = ids_sel.sum()
        

      
        img = img.float()
        img_p = F.pad(img[ids_sel], [offset,offset,offset,offset])
        
 

        
        img_p = img_p + torch.rand_like(img_p)
 
        img_e = torch.zeros(batch_size,channel,img_size,img_size,self.ks,self.ks,device=self.device)
        for i in range(ks2):
            x,y = divmod(i,self.ks)
         
            img_e[:,:,:,:,x,y] = img_p[:,:,x:x+img_size,y:y+img_size];
        Std = torch.std(img_e,[4,5],unbiased=True,keepdim=True)
  
        Mean = torch.mean(img_e,[4,5],keepdim=True)

        img_3d = (-(img_e -Mean).pow(2).div(2*Std.pow(2))).exp().min(1,keepdim=True)[0]
      
  
        img_r = img_e.mul(img_3d).sum([1,4,5],keepdim=True).div(img_3d.sum([4,5],keepdim=True)).squeeze(1,4,5)
        
        img_r_p = F.pad(img_r, [offset,offset,offset,offset])
  
        img_r_e = torch.zeros(batch_size,img_size,img_size,ks2,device=self.device);
  
        for i in range(ks2):
            x,y = divmod(i,self.ks)

            img_r_e[:,:,:,i] = img_r_p[:,x:x+img_size,y:y+img_size];

        
        ids_mix = torch.randint(0,ks2,[batch_size,img_size,img_size],device=self.device)
        
        img_3d_s = img_3d.reshape([batch_size,img_size,img_size,ks2])
        
        img_3d_mod = img_3d_s.gather(3,ids_mix.unsqueeze(3)).squeeze(3)
      
        img_3d_mod_p = F.pad(img_3d_mod, [offset,offset,offset,offset])
   
        img_3d_mod_e = torch.zeros(batch_size,img_size,img_size,self.ks,self.ks,device=self.device)
        for i in range(ks2):
            x,y = divmod(i,self.ks)
            img_3d_mod_e[:,:,:,x,y] = img_3d_mod_p[:,x:x+img_size,y:y+img_size];
        
        img_3d_mod_e = img_3d_mod_e.unsqueeze(1)
     
        img_mod = img_e.mul(img_3d_mod_e).sum([4,5],keepdim=True).div(img_3d_mod_e.sum([4,5],keepdim=True)).squeeze()
        img[ids_sel] = img_mod 
         
        return img.byte()
