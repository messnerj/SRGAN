from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from natsort import natsorted


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, dataset_type, crop_size=0, upscale_factor=4):
        super(DatasetFromFolder, self).__init__()
        self.dataset_type = dataset_type
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        if dataset_type == 'pirm': # Sort pirm data, since these images will be compared to a ground truth image in Matlab
            self.image_filenames = natsorted(self.image_filenames)
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = hr_transform(crop_size)
        self.lr_transform = lr_transform(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
            lr_image = self.lr_transform(hr_image)
            if lr_image.shape[0] == 1: # If image is black/white, increase dimensions
                hr_image = hr_image.repeat(3,1,1)
                lr_image = lr_image.repeat(3,1,1)
        elif self.dataset_type == 'val': # no crop
            hr_image = Image.open(self.image_filenames[index])
            w, h = hr_image.size
            hr_image = CenterCrop((w//self.upscale_factor*self.upscale_factor,h//self.upscale_factor*self.upscale_factor))(hr_image)
            lr_scale = Resize((w//self.upscale_factor,h//self.upscale_factor), interpolation=Image.BICUBIC)
            lr_image = lr_scale(hr_image)
            lr_image = ToTensor()(lr_image)
            hr_image = ToTensor()(hr_image)
            if lr_image.shape[0] == 1: # If image is black/white, increase dimensions
                hr_image = hr_image.repeat(3,1,1)
                lr_image = lr_image.repeat(3,1,1)
        elif self.dataset_type == 'pirm':
            lr_image = Image.open(self.image_filenames[index])
            lr_image = ToTensor()(lr_image)
            hr_image = lr_image # doesn't matter, unused
            if lr_image.shape[0] == 1: # If image is black/white, increase dimensions
                hr_image = hr_image.repeat(3,1,1)
                lr_image = lr_image.repeat(3,1,1)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

