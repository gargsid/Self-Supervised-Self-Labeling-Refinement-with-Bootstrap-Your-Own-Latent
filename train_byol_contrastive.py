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

from byol import BYOL, ProjectorHead, BYOLContrastive

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
parser.add_argument('--model_path', type=str, default=None)

parser.add_argument('--theta_positive', type=float, default=0.8)
parser.add_argument('--theta_negative', type=float, default=0.2)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--eval_epochs', type=int, default=100)

parser.add_argument('--learning_rate', type=float, default=3e-2)
parser.add_argument('--scheduler_step', type=int, default=10)
parser.add_argument('--scheduler_gamma', type=float, default=10)

parser.add_argument('--eval_learning_rate', type=float, default=1e-4)

parser.add_argument('--save_model_dir', type=str, default="")
parser.add_argument('--trained_model_path', type=str, default="")

parser.add_argument('--optimizer', type=str, default="sgd")
parser.add_argument('--weight_decay', type=float, default=4e-4)
parser.add_argument('--dropout', type=float, default=0.01)

parser.add_argument('--jobid', type=int, default=0)

args = parser.parse_args()

if not os.path.exists(args.save_model_dir):
    os.mkdir(args.save_model_dir)
    # os.mkdir(os.path.join(args.save_model_dir, 'plots'))
    # logprint('created {} and ./plots dir!\n'.format(args.save_model_dir))


# logs_file = 'stl10_models/logs.txt'
logs_file = os.path.join(args.save_model_dir, 'logs.txt')

def logprint(log):
    print(log, end='')
    with open(logs_file, 'a') as f:
        f.write(log) 


logprint('*** JobID:{} ***\n'.format(args.jobid))

hyperparameters = vars(args) 
log = ''
for k, v in hyperparameters.items():
    log += '{}:{} '.format(k, v)
log += '\n'
logprint(log)

dataroot = 'stl10/stl10_binary/unlabeled_X.bin' # change to your data root
dataset = STLUnlabeledDataloader(dataroot)
byolDataLoader = DataLoader(dataset, args.batch_size, shuffle=True)

# dataset = STLLabeledDataloader(dataroot='stl10/stl10_binary/', mode='train')
train_ds = STL10(root='stl10/', split='train', transform=classification_tfms)
trainDataloader = DataLoader(train_ds, args.batch_size, shuffle=True)

# dataset = STLLabeledDataloader(dataroot='stl10/stl10_binary/', mode='test')
val_ds = STL10(root='stl10/', split='test', transform=val_tfms)
valDataloader = DataLoader(val_ds, args.batch_size, shuffle=True)

model = BYOLContrastive(theta_positive=args.theta_positive, theta_negative=args.theta_negative).to(device)
model.train()
# model.initialize_target_network()

params = list(model.parameters())
# optimizer = torch.optim.RMSprop(params, lr=args.learning_rate)
optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

criterion = nn.CrossEntropyLoss()

if args.trained_model_path != "":
    checkpoint = torch.load(args.trained_model_path)
    model.load_state_dict(checkpoint['byol'])
    # best_loss = checkpoint['best_loss']
    # graphs = checkpoint['graphs']
    best_loss = np.inf
    graphs = {'feature_loss':[], 'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]} 

    logprint('loaded pretrained model from {}\n'.format(args.trained_model_path))
else:
    best_loss = np.inf
    graphs = {'feature_loss':[], 'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]} 

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

        # if idx == 50:
        #     break

    return classifier_model, running_loss / len(dataLoader), running_acc / len(dataLoader)

def eval_byol():

    eval_model = nn.Sequential(*[copy.deepcopy(model.online_network), ProjectorHead(512, 512, 10)]).to(device)
    eval_model[0].requires_grad = False
    
    eval_params = list(eval_model.parameters())
    eval_optim = torch.optim.SGD(eval_params, lr=args.eval_learning_rate, momentum=0.9)

    for eval_epoch in range(args.eval_epochs):
        eval_model, train_loss, train_acc = run_classification_model(eval_model, eval_optim, trainDataloader, 'train')
        eval_model, val_loss, val_acc = run_classification_model(eval_model, eval_optim, valDataloader, 'val')
        
        logprint('EvalEpoch:{}/10 train loss:{:.4f} acc:{:.4f}\n'.format(eval_epoch, train_loss, train_acc))
        logprint('EvalEpoch:{}/10 val loss:{:.4f} acc:{:.4f}\n'.format(eval_epoch, val_loss, val_acc))

        # break

    return train_loss, train_acc, val_loss, val_acc
    

for epoch in range(epochs):
    logprint('ByolEpoch:{}/{}\n'.format(epoch+1, epochs))

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

        log = "{}/{} loss:{:.7f}\n".format(idx, len(byolDataLoader), running_loss / (idx+1))
        logprint(log)

        model.update_target_network()

        # if idx % 100==0:
        #     graphs['feature_loss'].append(running_loss / (idx+1))

        # break

    epoch_loss = running_loss / len(byolDataLoader)
    graphs['feature_loss'].append(epoch_loss)

    if epoch_loss < best_loss:
        logprint('saving a new best model!\n')

        best_loss = epoch_loss 
        torch.save({
            'byol': model.state_dict(),
            'best_loss': best_loss,
            'graphs' : graphs
        }, os.path.join(args.save_model_dir, 'byol.pth'))

    scheduler.step()

    plt.figure()
    x = len(graphs['feature_loss'])
    plt.plot(np.arange(x), graphs['feature_loss'], label='feature_loss')
    # x = len(graphs['val_acc'])
    # plt.plot(np.arange(x), graphs['val_acc'], label='val_acc')

    plt.legend(loc='best')
    plt.title('byol')
    plt.xlabel('runs')
    plt.ylabel('byol')

    path = os.path.join(args.save_model_dir, 'byol_loss.jpg')
    plt.savefig(path)
    plt.close()


    if epoch % 100 == 0:
        # evaluate byol
        train_loss, train_acc, val_loss, val_acc = eval_byol()
        
        graphs['train_loss'].append(train_loss)
        graphs['train_acc'].append(train_acc)

        graphs['val_loss'].append(val_loss)
        graphs['val_acc'].append(val_acc)


        plt.figure()
        x = len(graphs['val_loss'])
        plt.plot(np.arange(x), graphs['val_loss'], label='val_loss')
        x = len(graphs['train_loss'])
        plt.plot(np.arange(x), graphs['train_loss'], label='train_loss')

        plt.legend(loc='best')
        plt.title('loss')
        plt.xlabel('runs')
        plt.ylabel('loss')

        path = os.path.join(args.save_model_dir, 'loss.jpg')
        plt.savefig(path)
        plt.close()

        plt.figure()
        x = len(graphs['val_acc'])
        plt.plot(np.arange(x), graphs['val_acc'], label='val_acc')
        x = len(graphs['train_acc'])
        plt.plot(np.arange(x), graphs['train_acc'], label='train_acc')

        plt.legend(loc='best')
        plt.title('accuracy')
        plt.xlabel('runs')
        plt.ylabel('accuracy')

        path = os.path.join(args.save_model_dir, 'accuracy.jpg')
        plt.savefig(path)
        plt.close()


    # break