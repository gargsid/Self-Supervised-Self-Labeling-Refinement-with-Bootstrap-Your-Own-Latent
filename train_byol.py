import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import transforms as tfms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import STL10

import matplotlib.pyplot as plt
import os, argparse, copy, time

from stl10_dataloader import STLUnlabeledDataloader, STLLabeledDataloader 

from byol import BYOL, ProjectorHead 


def logprint(log):
    print(log, end='')
    with open(logs_file, 'a') as f:
        f.write(log)

def run_classification_model(classifier_model, eval_optim, dataLoader, phase='train'):
    
    if phase=='train':
        classifier_model.train()
    elif phase=='val':
        classifier_model.eval()

    running_acc = 0. 
    running_loss = 0.
            
    for idx, item in enumerate(dataLoader):
            
        inputs = item[0].to(device)
        labels = item[1].to(device)

        if phase =='train':
            eval_optim.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = classifier_model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels) 
                loss.backward()
                eval_optim.step()
        else:
            with torch.no_grad():
                outputs = classifier_model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels) 

        running_loss += loss.item() 
        running_acc += (torch.sum(preds == labels.data).float() / preds.shape[0]).item()

        log = "{} {}/{} loss:{:.4f} acc:{:.4f}\n".format(phase, idx, len(dataLoader), running_loss / (idx+1), running_acc / (idx+1))
        logprint(log)

        # break

        # if idx == 50:
        #     break

    return classifier_model, running_loss / len(dataLoader), running_acc / len(dataLoader)

def eval_byol():

    eval_model = nn.Sequential(*[copy.deepcopy(model.online_network), ProjectorHead(512, 512, 10)]).to(device)
    eval_model[0].requires_grad = False
    
    eval_params = list(eval_model.parameters())
    eval_optim = torch.optim.SGD(eval_params, lr=0.001, momentum=0.9)

    for eval_epoch in range(1):
        eval_model, train_loss, train_acc = run_classification_model(eval_model, eval_optim, trainDataloader, 'train')
        eval_model, val_loss, val_acc = run_classification_model(eval_model, eval_optim, valDataloader, 'val')
        
        logprint('EvalEpoch:{}/10 train loss:{:.4f} acc:{:.4f}\n'.format(eval_epoch, train_loss, train_acc))
        logprint('EvalEpoch:{}/10 val loss:{:.4f} acc:{:.4f}\n'.format(eval_epoch, val_loss, val_acc))

        # break

    return train_loss, train_acc, val_loss, val_acc



learning_rate = 0.008
weight_decay = 4e-4
scheduler_step = 10
save_model_dir = "."
logs_file = os.path.join(save_model_dir, 'logs.txt')
batch_size = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classification_tfms = tfms.Compose([
            tfms.ToTensor(),
            tfms.RandomHorizontalFlip(),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
val_tfms = tfms.Compose([
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
dataroot = 'stl10/stl10_binary/unlabeled_X.bin'
dataset = STLUnlabeledDataloader(dataroot)
byolDataLoader = DataLoader(dataset, batch_size, shuffle=True)

train_ds = STL10(root="../"+dataroot, split='train', transform=classification_tfms)
trainDataloader = DataLoader(train_ds, batch_size, shuffle=True)

val_ds = STL10(root="../"+dataroot, split='test', transform=val_tfms)
valDataloader = DataLoader(val_ds, batch_size, shuffle=True)

model = BYOL().to(device)
model.train()
epochs = 100
params = list(model.parameters())
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.99)
criterion = nn.CrossEntropyLoss()

best_loss = np.inf
graphs = {'feature_loss':[], 'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}


for epoch in range(epochs):

    running_loss = 0.
    # Iterate over data.
    for idx, item in enumerate(byolDataLoader):
        online_input = item['online_input'].to(device)
        target_input = item['target_input'].to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            loss = model(online_input, target_input)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() 

        log = "{}/{} loss:{:.4f}\n".format(idx, len(byolDataLoader), running_loss / (idx+1))
        # logprint(log)
        print(log)

        model.update_target_network()

        # if idx % 100==0:
        #     graphs['feature_loss'].append(running_loss / (idx+1))

        # break

    epoch_loss = running_loss / len(byolDataLoader)
    print("Epoch loss:",epoch_loss)
    graphs['feature_loss'].append(epoch_loss)

    if epoch_loss < best_loss:
        logprint('saving a new best model!\n')

        best_loss = epoch_loss 
        torch.save({
            'byol': model.state_dict(),
            'best_loss': best_loss,
            'graphs' : graphs
        }, os.path.join(save_model_dir, 'byol_withoutBN.pth'))

    scheduler.step()

    # evaluate byol
    train_loss, train_acc, val_loss, val_acc = eval_byol()
    
    graphs['train_loss'].append(train_loss)
    graphs['train_acc'].append(train_acc)

    graphs['val_loss'].append(val_loss)
    graphs['val_acc'].append(val_acc)