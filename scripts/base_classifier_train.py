import sys
sys.path.append('/home/zeyu/fht/diffusion/guided-diffusion')
from guided_diffusion.image_datasets import load_data
from torch.utils.data import DataLoader,Dataset
import torch as torch
from torch import nn
from torchvision import models
from torchvision.datasets import ImageFolder
from guided_diffusion import dist_util, logger
from typing import Any,Tuple
import os
import torchvision
from PIL import Image
from torchvision import transforms




torch.cuda.set_device(2)

dist_util.setup_dist()
defaults = dict(
        data_dir="/home/zeyu/fht/Data/imagenet-10",
        image_size=64,
        iterations=100, #epoch
        lr=1e-4,
        weight_decay=0.1,
        batch_size=8,
        num_class = 10
    )

def data_load(data_dir,batch_size, image_size):
    transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.485, 0.456, 0.406])
    ])
    dataset = ImageFolder(data_dir,transform)
    data_loader = DataLoader(dataset,batch_size,shuffle=True, num_workers=1, drop_last=True)
    return data_loader

data_loader = data_load(defaults['data_dir'],defaults['batch_size'],defaults['image_size'])



model = models.resnet50(pretrained=True)




model =  nn.Sequential(model,nn.Linear(1000,defaults['num_class']))

def remove_batchnorm_recursive(module):
    """
    递归地删除模型中的批归一化（Batch Normalization）模块
    Args:
        module (nn.Module): 需要删除批归一化模块的模块

    Returns:
        nn.Module: 不包含批归一化模块的新模块
    """
    if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
        for i, child in enumerate(module.children()):
            module[i] = remove_batchnorm_recursive(child)
        return module
    
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        return nn.Identity()

    elif isinstance(module, nn.Module):
        for name, child in module.named_children():
            module.__setattr__(name, remove_batchnorm_recursive(child))
        return module

    
    else:
        return module



# model = remove_batchnorm_recursive(model)
model.load_state_dict(torch.load('guided-diffusion/models/base_class/res50_epoch99_0.97054.pt'))



model.to(dist_util.dev())



loss_fn = nn.CrossEntropyLoss()

optim = torch.optim.AdamW(model.parameters(),lr=defaults['lr'],weight_decay=defaults['weight_decay'])

for epoch in range(defaults['iterations']):
    num = 0
    for i ,(imgs,label) in enumerate(data_loader):

        print(imgs.max(),imgs.min())
        imgs= imgs.to(dist_util.dev())
        imgs.requires_grad=True
        label = label.to(dist_util.dev())

        pred = model(imgs)
        # print(label == pred.argmax(dim=-1))

        loss = loss_fn(pred,label)

        grad = torch.autograd.grad(loss,imgs,torch.ones_like(loss),retain_graph=True,create_graph=True)[0]
        grad_2 = torch.autograd.grad(grad,imgs,torch.ones_like(grad),retain_graph=True,create_graph=True)[0]

        # optim.zero_grad()
        # loss.backward()
        # optim.step()
        num+= (pred.argmax(dim=-1)==label).sum()
        if i % 10 ==9:
            print(num/(defaults['batch_size']*(i+1)))
    
    print(f'epoch:{epoch}')
    # if epoch%5==4:
    #     torch.save(model.state_dict(),f'guided-diffusion/models/base_class/res50_epoch{epoch}_{num/13000:.5f}.pt')



