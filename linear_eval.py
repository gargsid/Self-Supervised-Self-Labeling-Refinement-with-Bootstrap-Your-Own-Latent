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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def plot_graphs(plots_dir, ssl_graphs, vanilla_graphs):
    
    statistics = {'ssl':ssl_graphs, 'vanilla':vanilla_graphs}

    for typ in ['loss', 'accuracy']:
        plt.figure()

        for phase in ['train', 'val']:
            for gnames, graphs in statistics.items():

                tasksummary = graphs[typ][phase]
                x = len(tasksummary)
                plt.plot(np.arange(x), tasksummary, label='{}_{}'.format(phase, gnames))

        plt.legend(loc='best')
        plt.title('{}'.format(typ))
        plt.xlabel('runs')
        plt.ylabel('{}'.format(typ))

        path = os.path.join(plots_dir, '{}.jpg'.format(typ))
        plt.savefig(path)
        plt.close()

classification_tfms = tfms.Compose([
            tfms.ToTensor(),
            tfms.RandomHorizontalFlip(),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
val_tfms = tfms.Compose([
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

parser = argparse.ArgumentParser()
parser.add_argument('--finetune_encoder', type=bool, default=False)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--learning_rate', type=float, default=3e-2)
parser.add_argument('--scheduler_step', type=int, default=10)

parser.add_argument('--weight_decay', type=float, default=4e-4)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--save_model_dir', type=str, default="")
parser.add_argument('--byol_model_path', type=str, default="")

parser.add_argument('--ssl_classifier_path', type=str, default="")
parser.add_argument('--vanilla_classifier_path', type=str, default="")

parser.add_argument('--jobid', type=int, default=0)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.save_model_dir):
    os.mkdir(args.save_model_dir)
    os.mkdir(os.path.join(args.save_model_dir, 'plots'))
plots_dir = os.path.join(args.save_model_dir, 'plots')

ssl_saved_model_path = os.path.join(args.save_model_dir, 'ssl_classifier.pth')
vanilla_saved_model_path = os.path.join(args.save_model_dir, 'vanilla_classifier.pth')

logs_file = os.path.join(args.save_model_dir, 'logs.txt')

def logprint(log):
    print(log, end='')
    with open(logs_file, 'a') as f:
        f.write(log)

logprint('**********Linear Eval JobID:{}************\n'.format(args.jobid))

train_ds = STL10(root='stl10/', split='train', transform=classification_tfms)
trainDataloader = DataLoader(train_ds, args.batch_size, shuffle=True)

val_ds = STL10(root='stl10/', split='test', transform=val_tfms)
valDataloader = DataLoader(val_ds, args.batch_size, shuffle=True)

byol_model = BYOL().to(device)
byol_ckpt = torch.load(args.byol_model_path)
byol_model.load_state_dict(byol_ckpt['byol'])
logprint('loaded trained byol model from {}\n'.format(args.byol_model_path))

ssl_classifier = nn.Sequential(*[copy.deepcopy(byol_model.online_network), ProjectorHead(512, 512, 10)]).to(device)
if args.finetune_encoder:
    ssl_classifier[0].requires_grad = True 
else:
    ssl_classifier[0].requires_grad = False 

if args.ssl_classifier_path != "":
    checkpoint = torch.load(args.ssl_classifier_path)
    ssl_classifier.load_state_dict(checkpoint['best_model_wts'])
    ssl_best_acc = checkpoint['best_val_acc']
    ssl_graphs = checkpoint['graphs']
    logprint('loaded pretrained model from {}\n'.format(args.ssl_classifier_path))
else:
    ssl_best_acc = 0
    ssl_graphs = {'loss':{'train':[],'val':[]}, 'accuracy':{'train':[],'val':[]}}

ssl_params = list(ssl_classifier.parameters())
ssl_optim = torch.optim.SGD(ssl_params, lr=0.009, momentum=0.9)

resnet18 = models.resnet18(pretrained=True)
modules = list(resnet18.children())[:-2] # delete the layer4, avgpool and fc layer.
backbone = nn.Sequential(*modules)
vanilla_classifier = nn.Sequential(*[backbone, ProjectorHead(512, 512, 10)]).to(device)
if args.finetune_encoder:
    vanilla_classifier[0].requires_grad = True 
else:
    vanilla_classifier[0].requires_grad = False 

if args.vanilla_classifier_path != "":
    checkpoint = torch.load(args.vanilla_classifier_path)
    vanilla_classifier.load_state_dict(checkpoint['best_model_wts'])
    vanilla_best_acc = checkpoint['best_val_acc']
    vanilla_graphs = checkpoint['graphs']
    logprint('loaded pretrained model from {}\n'.format(args.vanilla_classifier_path))
else:
    vanilla_best_acc = 0
    vanilla_graphs = {'loss':{'train':[],'val':[]}, 'accuracy':{'train':[],'val':[]}}


vanilla_params = list(vanilla_classifier.parameters())
vanilla_optim = torch.optim.SGD(vanilla_params, lr=0.009, momentum=0.9)

criterion = nn.CrossEntropyLoss()

epochs = args.epochs 

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

        if phase=='train':
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
    return classifier_model, running_loss / len(dataLoader), running_acc / len(dataLoader)


for epoch in range(args.epochs):
    logprint('Epoch:{}/{}'.format(epoch, args.epochs))

    ssl_classifier, ssl_train_loss, ssl_train_acc = run_classification_model(ssl_classifier, ssl_optim, trainDataloader, 'train')
    ssl_classifier, ssl_val_loss, ssl_val_acc = run_classification_model(ssl_classifier, ssl_optim, valDataloader, 'val')

    vanilla_classifier, vanilla_train_loss, vanilla_train_acc = run_classification_model(vanilla_classifier, vanilla_optim, trainDataloader, 'train')
    vanilla_classifier, vanilla_val_loss, vanilla_val_acc = run_classification_model(vanilla_classifier, vanilla_optim, valDataloader, 'val')
    
    logprint('SSL LinearEvalEpoch:{}/{} train loss:{:.4f} acc:{:.4f}\n'.format(epoch, args.epochs, ssl_train_loss, ssl_train_acc))
    logprint('SSL LinearEvalEpoch:{}/{} val loss:{:.4f} acc:{:.4f}\n'.format(epoch, args.epochs, ssl_val_loss, ssl_val_acc))

    logprint('Vanilla LinearEvalEpoch:{}/{} train loss:{:.4f} acc:{:.4f}\n'.format(epoch, args.epochs, vanilla_train_loss, vanilla_train_acc))
    logprint('Vanilla LinearEvalEpoch:{}/{} val loss:{:.4f} acc:{:.4f}\n'.format(epoch, args.epochs, vanilla_val_loss, vanilla_val_acc))

    ssl_graphs['loss']['train'].append(ssl_train_loss)
    ssl_graphs['loss']['val'].append(ssl_val_loss)
    ssl_graphs['accuracy']['train'].append(ssl_train_acc)
    ssl_graphs['accuracy']['val'].append(ssl_val_acc)

    vanilla_graphs['loss']['train'].append(vanilla_train_loss)
    vanilla_graphs['loss']['val'].append(vanilla_val_loss)
    vanilla_graphs['accuracy']['train'].append(vanilla_train_acc)
    vanilla_graphs['accuracy']['val'].append(vanilla_val_acc)

    plot_graphs(plots_dir, ssl_graphs, vanilla_graphs)

    if ssl_best_acc < ssl_val_acc:
        ssl_best_acc = ssl_val_acc 
        
        logprint('saving new best ssl classifier\n')

        torch.save({
            'best_model_wts':ssl_classifier.state_dict(),
            'best_val_acc':ssl_best_acc,
            'graphs' : ssl_graphs,
        }, ssl_saved_model_path)


    if vanilla_best_acc < vanilla_val_acc:
        vanilla_best_acc = vanilla_val_acc 
        
        logprint('saving new best vanilla classifier\n')

        torch.save({
            'best_model_wts':vanilla_classifier.state_dict(),
            'best_val_acc':vanilla_best_acc,
            'graphs' : vanilla_graphs,
        }, vanilla_saved_model_path)

