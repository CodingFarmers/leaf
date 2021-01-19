import os
import pandas as pd
import albumentations as albu
import matplotlib.pyplot as plt
import json
import seaborn as sns
import cv2
import albumentations as albu
import numpy as np
import random
import torch
from loss import BiTemperedLoss
import fitlog

fitlog.set_log_dir('vit/')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(719)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
BASE_DIR="../cassava/cassava-leaf-disease-classification/"
TRAIN_IMAGES_DIR=os.path.join(BASE_DIR,'train_images')

train_df=pd.read_csv(os.path.join(BASE_DIR,'origin_train.csv'))

with open(f'{BASE_DIR}/label_num_to_disease_map.json', 'r') as f:
    name_mapping = json.load(f)
    
name_mapping = {int(k): v for k, v in name_mapping.items()}
train_df["class_id"]=train_df["label"].map(name_mapping)

def visualize_images(image_ids,labels):
    plt.figure(figsize=(16,12))
    
    for ind,(image_id,label) in enumerate(zip(image_ids,labels)):
        plt.subplot(3,3,ind+1)
        
        image=cv2.imread(os.path.join(TRAIN_IMAGES_DIR,image_id))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        plt.imshow(image)
        plt.title(f"Class: {label}",fontsize=12)
        
        plt.axis("off")
    plt.show()
    

def plot_augmentation(image_id,transform):
    plt.figure(figsize=(16,4))
    
    img=cv2.imread(os.path.join(TRAIN_IMAGES_DIR,image_id))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.axis("off")
    
    plt.subplot(1,3,2)
    x=transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    
    plt.subplot(1,3,3)
    x=transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    
    plt.show()
    
    
def visualize(images, transform):
    """
    Plot images and their transformations
    """
    fig = plt.figure(figsize=(32, 16))
    
    for i, im in enumerate(images):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        plt.imshow(im)
        
    for i, im in enumerate(images):
        ax = fig.add_subplot(2, 5, i + 6, xticks=[], yticks=[])
        plt.imshow(transform(image=im)['image'])

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split
from albumentations.pytorch import ToTensorV2
# from efficientnet_pytorch import EfficientNet
import time
import datetime
import copy

# DataSet class

class CassavaDataset(Dataset):
    def __init__(self,df:pd.DataFrame,imfolder:str,train:bool = True, transforms=None):
        self.df=df
        self.imfolder=imfolder
        self.train=train
        self.transforms=transforms
        
    def __getitem__(self,index):
        im_path=os.path.join(self.imfolder,self.df.iloc[index]['image_id'])
        x=cv2.imread(im_path,cv2.IMREAD_COLOR)
        x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
        
        if(self.transforms):
            x=self.transforms(image=x)['image']
        
        if(self.train):
            y=self.df.iloc[index]['label']
            return x,y
        else:
            return x
        
    def __len__(self):
        return len(self.df)

train_augs = albu.Compose([
    albu.RandomResizedCrop(height=384, width=384, p=1.0),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.RandomBrightnessContrast(p=0.5),
    albu.ShiftScaleRotate(p=0.5),
    albu.Normalize(    
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
    ToTensorV2(),
])

valid_augs = albu.Compose([
    albu.Resize(height=384, width=384, p=1.0),
    albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
    ToTensorV2(),
])

train, valid = train_test_split(
    train_df, 
    test_size=0.1, 
    random_state=719,
    stratify=train_df.label.values
)


# reset index on both dataframes
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

train_targets = train.label.values

# targets for validation
valid_targets = valid.label.values

train_dataset=CassavaDataset(
    df=train,
    imfolder=TRAIN_IMAGES_DIR,
    train=True,
    transforms=train_augs
)

valid_dataset=CassavaDataset(
    df=valid,
    imfolder=TRAIN_IMAGES_DIR,
    train=True,
    transforms=valid_augs
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    num_workers=4,
    shuffle=False,
)

def train_model(datasets, dataloaders, model, criterion, optimizer, scheduler, num_epochs, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels=labels.to(device)

                # Zero out the grads
                optimizer.zero_grad()

                # Forward
                # Track history in train mode
                with torch.set_grad_enabled(phase == 'train'):
                    model=model.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Statistics
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss/len(datasets[phase])
            epoch_acc = running_corrects.double()/len(datasets[phase])

            torch.save(model.state_dict(), f'ViT-B_16_fold_{epoch}.pt')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

from vision_transformer_pytorch import VisionTransformer

# model_name = 'efficientnet-b7'
datasets={'train':train_dataset,'valid':valid_dataset}
dataloaders={'train':train_loader,'valid':valid_loader}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model=models.(pretrained=True)
# model.fc=nn.Linear(512,5)
# model = EfficientNet.from_pretrained(model_name, num_classes=5) 
# model=models.resnext50_32x4d()#Add Pretrained=True to use pretrained with internet enabled
# model.fc=nn.Linear(model.fc.in_features,5)
model = VisionTransformer.from_pretrained('ViT-B_16', num_classes=5) 

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
criterion = BiTemperedLoss(t1=CFG['t1'], t2=CFG['t2'], label_smoothing=CFG['label_smoothing'])
# criterion=nn.CrossEntropyLoss()
num_epochs=10
trained_model=train_model(datasets,dataloaders,model,criterion,optimizer,scheduler,num_epochs,device)
fitlog.finish()
