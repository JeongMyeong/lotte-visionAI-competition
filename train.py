
save_dir = 'b2-adamp'
# Library Load
from adamp import AdamP
import warnings
warnings.filterwarnings('ignore')
# import wandb
import torch
from glob import glob
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import KFold
import time
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from torch_poly_lr_decay import PolynomialLRDecay
import random
import timm
from torchvision import models
# from sam import SAM
import albumentations as A
from skimage import measure
from sklearn.metrics import accuracy_score
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
torch.set_num_threads(8)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import math



# seed everything
def seed_everything(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
random_state=2925
seed_everything(random_state)




# path load
sample_submit = pd.read_csv('dataset/sample.csv')[:]
train_imgs_path = np.array(glob('dataset/train/*/*'))[:]


# train imgs load
imgs=[]
labels=[]
for n, path in tqdm(enumerate(train_imgs_path), total=len(train_imgs_path)):
    label = int(train_imgs_path[n].split('/')[2])
    labels.append(label)
    img=cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)    
imgs=np.array(imgs)
labels=np.array(labels)



# test imgs load
test_imgs=[]
for path in tqdm(sample_submit['filename'].values):
    img=cv2.imread('dataset/test/'+path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_imgs.append(img)
    
test_imgs=np.array(test_imgs)



# DataLoader
class LotteDataset(Dataset):
    def __init__(self, imgs, labels, transform, train=True, trainset=False, s_range=None,e_range=None,weight=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.train = train
        self.trainset = trainset
        self.s_range=s_range
        self.e_range=e_range
        self.weight=weight
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        
        cutoff = random.randint(0,2)
        if self.trainset:
            aug = A.Compose([
                        A.ShiftScaleRotate(border_mode=1),
                        A.GridDistortion(border_mode=1),
                        A.Blur(blur_limit=1),
                        A.HorizontalFlip(p=0.5),
                        A.Rotate(border_mode=1),
                        A.OneOf([
                                A.CLAHE(),
                                A.RandomBrightnessContrast(),]),
                            ])
            img = aug(image=img)['image']

            # mixup
            if cutoff == 1:
                alpha=random.randint(0,33333)*1e-5
                random_idx = random.randint(0, len(self.imgs)-1)           # random index
                random_img = self.imgs[random_idx]               # random image 선택
                random_label = self.labels[random_idx]                     # random label 선택
                random_soft = np.zeros(1000)                               # label 생성
                random_soft[random_label] = alpha
                random_img = aug(image=random_img)['image']
#                 img[:256,:256] = img*alpha + random_img*(1-alpha)
                
                alpha2=random.randint(0,33333)*1e-5
                random_idx2 = random.randint(0, len(self.imgs)-1)           # random index
                random_img2 = self.imgs[random_idx2]               # random image 선택
                random_label2 = self.labels[random_idx2]                     # random label 선택
                random_soft2 = np.zeros(1000)                               # label 생성
                random_soft2[random_label2] = alpha2
                random_img2 = aug(image=random_img2)['image']
                
                img[:256,:256] = img*(1-alpha-alpha2) + random_img*alpha + random_img2*alpha2
                
            # Cutmix
            elif cutoff == 2:

                width = random.randint(32, 150)
                height = random.randint(32, 150)

                if (self.s_range!=None) & (self.e_range !=None):
                    width = random.randint(self.s_range, self.e_range)
                    height = random.randint(self.s_range, self.e_range)

                random_x = random.randint(0, 256-width)
                random_y = random.randint(0, 256-height)
                random_position_x, random_position_y = random.randint(0,256-width), random.randint(0,256-height)
                random_idx = random.randint(0, len(self.imgs)-1)           # random index
                random_img = self.imgs[random_idx]               # random image 선택
                random_label = self.labels[random_idx]                     # random label 선택
                random_soft = np.zeros(1000)                               # label 생성
                alpha = (width*height)/(256*256)

                random_soft[random_label] = alpha
                random_img = aug(image=random_img)['image']
                img[random_position_y:random_position_y+height,random_position_x:random_position_x+width] = random_img[random_y:random_y+height, random_x:random_x+width]


                
        img = self.transform(img)   
        if self.train:
            label = self.labels[idx]
            soft_label = np.zeros(1000)
            
            if (self.trainset) & (cutoff==1):
                soft_label[label] = 1-alpha-alpha2
                label = random_soft+soft_label+random_soft2
            elif (self.trainset) & (cutoff==2):
                soft_label[label]=1-alpha
                label = random_soft+soft_label
            elif (self.trainset) & (cutoff==3):
                soft_label[label]=1-alpha
                label = random_soft+soft_label
            else:
                soft_label[label] = 1
                label = soft_label
            return img, label
        else:
            return img
    

# model load
class Network_Efficientnet(nn.Module):
    def __init__(self, b=None):
        super(Network_Efficientnet, self).__init__()
        self.pretrained_net = EfficientNet.from_pretrained(f'efficientnet-b{b}', in_channels=3)
        self.FC = nn.Linear(1000, 1000)

    def forward(self, x):
        x = F.relu(self.pretrained_net(x))
        x = self.FC(x)
        x = F.softmax(x)
        return x




# KFold
skf = StratifiedKFold(n_splits=7, random_state=random_state, shuffle=True)
folds=[]
for train_idx, valid_idx in skf.split(imgs, labels):
    folds.append((train_idx, valid_idx))



# Defined Transforms 
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
])


test_dataset = LotteDataset(test_imgs, labels=None, transform=test_transform, train=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=2)


# test dataset prediction & probability save
def output_submit(model_path, b=None, rot_k=0):
    model = Network_Efficientnet(b=b).to(device)
    model = nn.DataParallel(model, device_ids=[0,1,2])
    weights = torch.load(f'{save_dir}/{model_path}')
    model.load_state_dict(weights['model'])
    model.eval()
    
    preds=[]
    for n, img in tqdm(enumerate(test_loader), total=len(test_loader)):
        if (rot_k!=0) & (rot_k<5):
            img = torch.rot90(img, k=rot_k, dims=(2,3))
        if rot_k==5:
            img = torch.flip(img, (3,))
        img = torch.tensor(img, dtype=torch.float32, device=device)
        pred = model(img)
        pred = pred.detach().cpu().numpy()
        preds += list(pred)

    return np.array(preds)


# model save
def model_save(model, optimizer, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)




# Config Setting
config={}
config['epochs'] = 55
config['batch_size'] = 128
config['init_learning_rate'] = 1e-3
config['end_learning_rate'] = 1e-6
config['scheduler'] = 'polynomial'
config['architecutre'] = 'efficinetnet-adamp'
config['b'] = 2
config['random_state'] = random_state
config['save_dir'] = save_dir


# 7-folds Training Start
for fold in range(7):
    config['fold'] = fold

    b=config['b']
    architecture = config['architecture']
    random_state = config['random_state']


    model_name = f'{architecture}-b{b}-adamp-seed({random_state})-fold({fold})'
    
    train_idx, valid_idx = folds[fold]
    epochs=config['epochs']
    batch_size=config['batch_size']


    # DataLoader
    train_dataset = LotteDataset(imgs[train_idx], labels=labels[train_idx], transform=train_transform, train=True, trainset=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    valid_dataset = LotteDataset(imgs[valid_idx], labels=labels[valid_idx], transform=valid_transform, train=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


    # Model Load
    model = Network_Efficientnet(b=b).to(device)
    model = nn.DataParallel(model, device_ids=[0,1,2])



    # Optimizer & Scheduler
    # optimizer = torch.optim.Adam(model.parameters(), lr =1e-3)
    # Q = math.floor(len(train_dataset)/batch_size+1)*epochs/7
    # lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = Q)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    optimizer = AdamP(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)
    decay_steps = (len(train_dataset)//batch_size+1)*(epochs-2)
    plr = PolynomialLRDecay(optimizer, max_decay_steps=decay_steps, end_learning_rate=1e-6, power=0.9)


    # Loss
    criterion = nn.BCELoss()


    # Training
    best=0
    save=0
    for epoch in range(epochs):
        model.train()
        start = time.time()
        train_accuracy=0
        train_loss=0
        valid_accuracy=0
        valid_loss=0


        pred_list=[]
        label_list=[]
        epoch_loss=0

        for img, label in (train_loader):
            img = torch.tensor(img, device=device, dtype=torch.float32)
            label = torch.tensor(label, device=device, dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            plr.step()

            pred = pred.argmax(1).detach().cpu().numpy()
            label = label.detach().cpu().numpy().argmax(1)

            pred_list += list(pred)
            label_list += list(label)
            epoch_loss+=loss.item()


        train_epoch_accuracy = accuracy_score(label_list, pred_list)
        train_epoch_loss = epoch_loss/len(train_loader)


        pred_list=[]
        label_list=[]
        epoch_loss=0
        model.eval()
        with torch.no_grad():
            for img, label in (valid_loader):
                img = torch.tensor(img, device=device, dtype=torch.float32)
                label = torch.tensor(label, device=device, dtype=torch.float32)

                pred = model(img)
                loss = criterion(pred, label)


                pred = pred.argmax(1).detach().cpu().numpy()
                label = label.detach().cpu().numpy().argmax(1)

                pred_list += list(pred)
                label_list += list(label)
                epoch_loss+=loss.item()


        valid_epoch_accuracy = accuracy_score(label_list, pred_list)
        valid_epoch_loss = epoch_loss/len(valid_loader)
        end = time.time()-start
        
        log=f'fold : {fold+1}\tEpoch : {epoch+1}/{epochs}\ttrain_accuracy : {train_epoch_accuracy:.5f}\ttrain_loss : {train_epoch_loss:.5f}\tvalid_accuracy : {valid_epoch_accuracy:.5f}\tvalid_loss : {valid_epoch_loss:.5f}\ttime : {end:.0f}s/{(epochs-(epoch+1))*end:.0f}s left'
        print(log)
        
        if valid_epoch_accuracy>best:
            best = valid_epoch_accuracy
            model_save(model=model, optimizer=optimizer, path=f'{save_dir}/{model_name}.pt')
        # wandb.log({'train_accuracy':train_epoch_accuracy, 'train_loss':train_epoch_loss,'valid_accuracy':valid_epoch_accuracy,'valid_loss':valid_epoch_loss,'learning_rate':optimizer.param_groups[0]["lr"]})


    submit = output_submit(f'{model_name}.pt', b=b)
    np.save(f'{save_dir}/{model_name}.npy', submit)
    submit = output_submit(f'{model_name}.pt', b=b, rot_k=5)
    np.save(f'{save_dir}/{model_name}-TTAflip.npy', submit)
    telegram_send(f'fold{fold+1} end !')
    run.finish()
