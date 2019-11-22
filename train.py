#### Main script to train SRGAN ####
# Authors: Zainab Khan, Jonas Messner

# Packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Model Files, Utils, etc.
from utils import *


# Parser
### TASK: Think about which input parameters should be sweepable, e.g. scale_factor, epochs. Write parser.


# Load data
print("Data Loading...")
data_train = DatasetFromFolder('../data/tiny-imagenet-200/train/all_images', crop_size=64, upscale_factor=4)
data_val = DatasetFromFolder('../data/tiny-imagenet-200/temp/images', upscale_factor=4)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=256, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
print("Done.")


# Instantiate networks
G = 0 # Generator model
D = 0 # Discriminator model

# Instantiate loss functions
loss_func_G = 0
loss_func_D = 0

# Instantiate optimizer
optim_G = optim.Adam(G.parameters())
optim_D = optim.Adam(D.parameters())




# Train network for defined number of epochs
epochs = 10
for epoch in range(1, epochs + 1):

    #### TRAINING ####
    # Set models into training mode
    G.train()
    D.train()

    # Train discriminator
    D.zero_grad()
    dis_loss = 0


    dis_loss.backward(retain_graph=True)
    optim_D.step()

    # Train generator
    G.zero_grad()
    gen_loss = loss_func_G()
    gen_loss.backward()
    optim_G.step()


    # Print losses
    print('\rEpoch: %03d/%03d - Step: %03d/%03d  - GenLoss: %.4f - DisLoss: %.4f'%(
        epoch+1, epochs, batch_idx+1, batches, gen_loss, dis_loss), end='')


    #### VALIDATION ####
    # Set generator into evaluation mode
    G.eval()







