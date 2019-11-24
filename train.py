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
from models.model import *
from loss import *
from utils import *

# GPU setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")




# Parser
### TASK: Think about which input parameters should be sweepable, e.g. scale_factor, epochs. Write parser.


# Load data
print("Data Loading...")
data_train = DatasetFromFolder('data/tiny-imagenet-200/train/all_images', crop_size=64, upscale_factor=4)
data_val = DatasetFromFolder('data/tiny-imagenet-200/temp/images', crop_size=64, upscale_factor=4)
train_loader = DataLoader(dataset=data_train, num_workers=4, batch_size=256, shuffle=True)
val_loader = DataLoader(dataset=data_val, num_workers=4, batch_size=1, shuffle=False)
print("Done.")


print("Instantiation of model, loss functions and optimizer...")
# Instantiate networks
G = Generator(num_residual_blocks=8, scale_factor=4) # Generator model
D = Discriminator() # Discriminator model
G.cuda()
D.cuda()

# Instantiate loss functions
loss_weights_G = torch.tensor([1,1,1,1],device=device)
loss_func_G = GeneratorLoss(loss_weights_G).cuda()
loss_func_D = DiscriminatorLoss().cuda()

# Instantiate optimizer
optim_G = optim.Adam(G.parameters())
optim_D = optim.Adam(D.parameters())
print("Done.")

# Train network for defined number of epochs
print("Training...")
num_batches = len(train_loader)
epochs = 10
for epoch in range(1, epochs + 1):
    for batch_idx, data in enumerate(train_loader):
        img_LR = data[0] # Low resolution image (input to generator)
        img_HR = data[1] # High resolution image (ground truth)

        # Transfer to GPU
        img_LR, img_HR = img_LR.to(device), img_HR.to(device)

        #### TRAINING ####
        # Set models into training mode
        G.train()
        D.train()

        # Generate an image using G
        img_SR = G(img_LR)

        # Train discriminator
        D.zero_grad()
        labels_real = D(img_HR)
        labels_gen = D(img_SR)
        dis_loss = loss_func_D(labels_real, labels_gen)
        dis_loss.backward(retain_graph=True)
        optim_D.step()

        # Train generator
        G.zero_grad()
        gen_loss = loss_func_G(img_HR, img_SR, labels_gen)
        gen_loss.backward()
        optim_G.step()


        # Print losses
        print('\rEpoch: %03d/%03d - Step: %03d/%03d  - GenLoss: %.4f - DisLoss: %.4f'%(
            epoch+1, epochs, batch_idx+1, num_batches, gen_loss, dis_loss), end='')


    #### VALIDATION ####
    # Set generator into evaluation mode
    G.eval()







