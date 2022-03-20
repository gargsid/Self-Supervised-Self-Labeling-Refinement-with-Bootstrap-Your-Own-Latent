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


g1 = torch.load('stl10_models/m40/contrastive_loss_from_scratch/run_1/byol.pth')['graphs']
train_acc_1 = g1['train_acc']
val_acc_1 = g1['val_acc']
print(len(train_acc_1))
print(len(val_acc_1))

g2 = torch.load('stl10_models/m40/contrastive_loss_from_scratch/run_2/byol.pth')['graphs']
train_acc_2 = g2['train_acc']
val_acc_2 = g2['val_acc']
print(len(train_acc_2))
print(len(val_acc_2))

train_acc = [train_acc_1[i] for i in range(0, len(train_acc_1), 100)] + [train_acc_1[-1]] + train_acc_2

val_acc = [val_acc_1[i] for i in range(0, len(val_acc_1), 100)] + [val_acc_1[-1]] + val_acc_2

print(len(train_acc))
print(len(val_acc))

plt.figure()
x = len(train_acc)
plt.plot(np.arange(x), train_acc, label='train_acc')
plt.plot(np.arange(x), val_acc, label='val_acc')

plt.legend(loc='best')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('CCSL BYOL feature loss')

plt.savefig('stl10_models/m40/contrastive_loss_from_scratch/run_2/accuracy_2.jpg')
plt.close()

# plt.figure()
# x = len(graphs['val_acc'][:350])
# plt.plot(np.arange(x), graphs['val_acc'][:350], label='val_acc')
# x = len(graphs['train_acc'][:350])
# plt.plot(np.arange(x), graphs['train_acc'][:350], label='train_acc')

# plt.legend(loc='best')
# plt.title('accuracy')
# plt.xlabel('runs')
# plt.ylabel('accuracy')

# path = os.path.join(save_model_dir, 'plots', 'accuracy_2.jpg')
# plt.savefig(path)
# plt.close()

# save_model_dir = 'stl10_models/m40/sgd_run_4'
# trained_model_path = 'stl10_models/m40/vanilla_byol/sgd_run_5/byol.pth'
# trained_model_path = 'stl10_models/m40/contrastive_loss_from_scratch/run_2/byol.pth'
# trained_model_path = 'stl10_models/m40/sigmoid_pos_neg/run_2/byol.pth'

# checkpoint = torch.load(trained_model_path)
# graphs = checkpoint['graphs']

# val_acc = graphs['val_acc']

# max_val_acc = max(val_acc)
# print('max_val_acc:', max_val_acc)

# plt.figure()
# x = len(graphs['feature_loss'])
# plt.plot(np.arange(x), graphs['feature_loss'], label='feature_loss')
# # x = len(graphs['val_acc'])
# # plt.plot(np.arange(x), graphs['val_acc'], label='val_acc')

# plt.legend(loc='best')
# plt.title('byol')
# plt.xlabel('epochs')
# plt.ylabel('loss')

# path = os.path.join(save_model_dir, 'plots', 'byol_loss_2.jpg')
# plt.savefig(path)
# plt.close()