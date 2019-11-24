import torch
import torch.nn as nn
import torch.nn.functional as F

def sandwich_lrelu(x, conv, bn):
    return bn(nn.LeakyReLU()(conv(x)))

# todo discriminator
# fix prelu everywhere
# maybe add modules? for dynamic size
# check padding
# inplace relu/sigmoid

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=(2,2), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=(2,2), padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=(2,2), padding=2)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=(2,2), padding=2)
        self.bn8 = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        self.fc1 = nn.Linear(512*2*2, 1024) 
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = sandwich_lrelu(x, self.conv1, self.bn1)
        x = sandwich_lrelu(x, self.conv2, self.bn2)
        x = sandwich_lrelu(x, self.conv3, self.bn3)
        x = sandwich_lrelu(x, self.conv4, self.bn4)
        x = sandwich_lrelu(x, self.conv5, self.bn5)
        x = sandwich_lrelu(x, self.conv6, self.bn6)
        x = sandwich_lrelu(x, self.conv7, self.bn7)
        x = sandwich_lrelu(x, self.conv8, self.bn8)
        x = self.avgpool(x)
        x = torch.reshape(x,(-1,))
        x = sandwich_lrelu(x, self.fc1,   self.fc2)
        x = nn.Sigmoid()(x) 

        return x

def create_residual_blocks(B):
    layers = []
    for i in range(B):
        block = nn.Module()
        block.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        block.bn1 = nn.BatchNorm2d(64)
        block.prelu = nn.PReLU()
        block.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        block.bn2 = nn.BatchNorm2d(64)
        layers.append(block)
    return layers

def create_upscaler_blocks(scale_factor):
    B = scale_factor//2 # number of repeated modules
    # always scale by 2 at a time
    layers = []
    for i in range(B):
        block = nn.Module()
        block.conv = nn.Conv2d(64, 64 * 4, 3, padding=1)
        block.shuffle = nn.modules.PixelShuffle(2)
        block.prelu = nn.PReLU()
        layers.append(block)
    return layers

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=8, scale_factor=4):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=1)
        self.prelu1 = nn.PReLU()
        self.blocks = create_residual_blocks(num_residual_blocks)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.upscalers = create_upscaler_blocks(scale_factor)
        self.conv3 = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        skip_residual_blocks = x
        residual = x

        for i in range(len(self.blocks)):
            x = self.blocks[i].conv1(x)
            x = self.blocks[i].bn1(x)
            x = self.blocks[i].prelu(x)
            x = self.blocks[i].conv2(x)
            x = self.blocks[i].bn2(x)
            x = residual + x
            residual = x
        
        # after residual blocks
        x = self.conv2(x)
        x = self.bn2(x)
        # last skip connection
        x = skip_residual_blocks + x
       
        # grow by 2x at a time 
        for i in range(len(self.upscalers)):
            x = self.upscalers[i].conv(x)
            x = self.upscalers[i].shuffle(x)
            x = self.upscalers[i].prelu(x)
        
        x = self.conv3(x)
        
        return x

    def cuda(self, device=None):
        for i in range(len(self.blocks)):
            self.blocks[i] = self.blocks[i].cuda()
        for i in range(len(self.upscalers)):
            self.upscalers[i] = self.upscalers[i].cuda()
        return super(Generator,self).cuda()


