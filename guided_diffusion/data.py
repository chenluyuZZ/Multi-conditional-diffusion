from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn


def data_load(data_dir,batch_size, image_size):
    transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.485, 0.456, 0.406])
    ])
    dataset = datasets.ImageFolder(data_dir,transform)
    data_loader = DataLoader(dataset,batch_size,shuffle=True, num_workers=1, drop_last=True)
    return data_loader



import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

def visualize_gradients(gradient, save_path):
    # 归一化梯度值到0-1范围
    gradient = gradient - gradient.min()
    gradient = gradient / gradient.max()

    # 转换为RGB图像
    gradient = transforms.ToPILImage()(gradient)
    gradient = transforms.Resize((224, 224))(gradient)
    gradient = transforms.ToTensor()(gradient)

    # 保存图像
    save_image(gradient, save_path)



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