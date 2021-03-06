import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sandwich_lrelu(x, conv, bn):
    return nn.LeakyReLU(0.2)(bn(conv(x)))

# todo discriminator
# fix prelu everywhere
# maybe add modules? for dynamic size
# check padding
# inplace relu/sigmoid

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        #self.bn1 = nn.BatchNorm2d(64) # no batchnorm in first layer
        self.conv2 = nn.Conv2d(64, 64, 3, stride=(2,2), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=(2,2), padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=(2,2), padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=(2,2), padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(512, 1024, 1) #nn.Linear(512*1*1, 1024) 
        self.fc2 = nn.Conv2d(1024,   1, 1) #nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = sandwich_lrelu(x, self.conv2, self.bn2)
        x = sandwich_lrelu(x, self.conv3, self.bn3)
        x = sandwich_lrelu(x, self.conv4, self.bn4)
        x = sandwich_lrelu(x, self.conv5, self.bn5)
        x = sandwich_lrelu(x, self.conv6, self.bn6)
        x = sandwich_lrelu(x, self.conv7, self.bn7)
        x = sandwich_lrelu(x, self.conv8, self.bn8)
        #old_shape = x.shape
        #x = x.reshape((-1,x.shape[2],x.shape[3]))
        x = self.avgpool(x)
        #x = x.reshape((old_shape[0], -1))
        x = self.fc1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.fc2(x)
        #x = nn.Sigmoid()(x) # Sigmoid is inherently computed in loss function - better stability
        return x



class Generator(nn.Module):
    def __init__(self, num_residual_blocks, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.x1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.x2 = ResidualBlock(64)
        self.x3 = ResidualBlock(64)
        self.x4 = ResidualBlock(64)
        self.x5 = ResidualBlock(64)
        self.x6 = ResidualBlock(64)
        self.x7 = ResidualBlock(64)
        self.x8 = ResidualBlock(64)
        self.x9 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        x10 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        x10.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.x10 = nn.Sequential(*x10)

    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        x6 = self.x6(x5)
        x7 = self.x7(x6)
        x8 = self.x8(x7)
        x9 = self.x9(x8)
        x10 = self.x10(x1 + x9)

        return (torch.tanh(x10) + 1) / 2



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

