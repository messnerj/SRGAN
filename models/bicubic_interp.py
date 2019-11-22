# Perform bicubic interpolation on low resolution images to scale them up

import scipy.misc
import numpy as np

# Image IDs from PRIM test set
img_ids = np.arange(201,301)

# Scaling factor
scale = 4

for i in img_ids:
    # Read image
    path = '../evaluation/PIRM_testset/4x_downsampled/'+str(i)+'.png'
    img = scipy.misc.imread(path)
    shape = img.shape

    # Define new shape
    shape_SR = (shape[0]*scale, shape[1]*scale, shape[2])

    # Generate and save SR image
    img_SR = scipy.misc.imresize(img,shape_SR,interp='bicubic')
    fileName = '../results/bicubic_interp/'+str(i)+'.png'
    scipy.misc.imsave(fileName, img_SR)

