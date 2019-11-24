import torch
from torch import nn
from torchvision.models.vgg import vgg16

# Copied from CS231N Github
def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


class GeneratorLoss(nn.Module):
    def __init__(self, weights):
        super(GeneratorLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss() # Adversarial loss
        self.mse_loss = nn.MSELoss() # MSE loss for img_loss and percept_loss

        vgg = vgg16(pretrained=True)
        self.feature_net = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.feature_net.parameters():
            param.requires_grad = False

        self.w = weights/torch.sum(weights) # Weights of losses (order: adv, img, percept, tv)

    def forward(self, img_real, img_gen, labels_gen):
        # Adversarial loss
        labels_target = torch.ones_like(labels_gen) #  - torch.rand_like(labels_gen)*0.2 # smoothened labels (make generated imgs look like real imgs)
        adv_loss = self.bce_loss(labels_gen, labels_target)

        # Image loss (Enforcing images to look similar)
        img_loss = self.mse_loss(img_gen, img_real)

        # Perceptual loss (Enforcing images to have similar features)
        features_gen = self.feature_net(img_gen)
        features_real = self.feature_net(img_real)
        percept_loss = self.mse_loss(features_gen, features_real)

        # Total variation loss (Enforcing smoothness)
        totalvar_loss = tv_loss(img_gen,1)

        # Overall loss
        gen_loss = self.w[0]*adv_loss + self.w[1]*img_loss + self.w[2]*percept_loss + self.w[3]*totalvar_loss
        return gen_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss() # Adversarial loss

    def forward(self, labels_real, labels_gen):
        # Loss real images
        labels_target_real = torch.ones_like(labels_real) - torch.rand_like(labels_real)*0.2 # smoothened label
        loss_real = self.bce_loss(labels_real, labels_target_real)

        # Loss generated images
        labels_target_gen = torch.zeros_like(labels_gen)
        loss_gen = self.bce_loss(labels_gen, labels_target_gen)
        
        # Overall loss
        dis_loss = loss_real/2 + loss_gen/2
        return dis_loss


