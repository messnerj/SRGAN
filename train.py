#### Main script to train SRGAN ####
# Authors: Zainab Khan, Jonas Messner

# Packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import pytorch_ssim
import argparse


# Model Files, Utils, etc.
from models.model import *
from loss import *
from utils import *

# GPU setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parser
### TASK: Think about which input parameters should be sweepable, e.g. scale_factor, epochs. Write parser.
parser = argparse.ArgumentParser(description='Parameters for training SRGAN.')
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of epochs to train both models')
#parser.add_argument('pretrain_epochs', nargs='?', default=200,
#		    help='number of epochs to pretrain discriminator')
parser.add_argument('--upscale_factor', default=4, type=int,
		    help='how much to super resolve image by')
parser.add_argument('--lr_g', default=1e-3, type=float,
                    help='learning rate for generator')
parser.add_argument('--lr_d', default=2e-3, type=float,
                    help='learning rate for discriminator')
parser.add_argument('--residual_blocks', default=8, type=int,
		    help='number of residual blocks')
parser.add_argument('--crop_size', default=200, type=int,
		    help='crop size of training/val images')
parser.add_argument('--training_batch_size', default=16, type=int,
		    help='batch size of training images')

args = parser.parse_args()

# Load data
print("Data Loading...")
data_train = DatasetFromFolder('data/BSDS200', crop_size=args.crop_size, upscale_factor=args.upscale_factor)
data_val = DatasetFromFolder('data/Set14', crop_size=args.crop_size, upscale_factor=args.upscale_factor)
train_loader = DataLoader(dataset=data_train, num_workers=4, batch_size=args.training_batch_size, shuffle=True)
val_loader = DataLoader(dataset=data_val, num_workers=4, batch_size=1, shuffle=False)
print("Done.")


print("Instantiation of model, loss functions and optimizer...")
# Instantiate networks
G = Generator(num_residual_blocks=args.residual_blocks, scale_factor=args.upscale_factor) # Generator model
D = Discriminator() # Discriminator model
G.cuda()
D.cuda()

# Instantiate loss functions
#loss_weights_G = torch.tensor([0,1,0,0],device=device)
loss_weights_G = torch.tensor([1e-2,1,6e-2,0],device=device) # Order: adversarial, image, perceptual, tv
loss_func_G = GeneratorLoss(loss_weights_G).cuda()
loss_func_D = DiscriminatorLoss().cuda()

# Instantiate optimizer
optim_G = optim.Adam(G.parameters(), lr=args.lr_g)
optim_D = optim.Adam(D.parameters(), lr=args.lr_d)
print("Done.")

# Train network for defined number of epochs
print("Training...")
num_batches = len(train_loader)
epochs = args.epochs
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
            epoch, epochs, batch_idx+1, num_batches, gen_loss, dis_loss), end='')


    #### VALIDATION ####
    # Set generator into evaluation mode
    G.eval()
    print('')

    with torch.no_grad():
        scores = [0,0]
        for batch_idx, data in enumerate(val_loader):
            val_img_LR = data[0] # Low resolution image (input to generator)
            val_img_HR = data[1] # High resolution image (ground truth)

            # Transfer to GPU
            val_img_LR, val_img_HR = val_img_LR.to(device), val_img_HR.to(device)

            # Generate an image using G
            val_img_SR = G(val_img_LR)  

            # Compute scores
            scores[0] += ((val_img_SR - val_img_HR)**2).data.mean()/len(val_loader) # MSE
            scores[1] += pytorch_ssim.ssim(val_img_SR, val_img_HR).item()/len(val_loader) # SSIM

            if batch_idx == 1:
                temp = val_img_LR[0]
                save_image(temp, 'results/val_LR.png')
                temp = val_img_HR[0]
                save_image(temp, 'results/val_HR.png')
                temp = val_img_SR[0]
                save_image(temp, 'results/val_SR.png')

        # Print scores
        print('Validation | MSE score: %f, SSIM score:%f'%(scores[0],scores[1]))

