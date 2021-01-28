import sys
import scipy
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

class DataLoader():
    def __init__(self, input_training_dir, input_test_dir, img_res, nb_channels_images, nb_channels_masks):
        self.input_training_dir = input_training_dir
        self.input_test_dir = input_test_dir
        self.img_res = img_res
        self.nb_channels_images = nb_channels_images
        self.nb_channels_masks = nb_channels_masks

    def load_batch(self):
        image_dir = os.path.join(self.input_training_dir, "images")
        mask_dir = os.path.join(self.input_training_dir, "masks")
        
        imageFileList = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        for imageFile in imageFileList:

            imgs_A, imgs_B = [], []            
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(image_dir, imageFile)
            if os.path.exists(os.path.join(mask_dir, baseName + ".png")):
                maskPath = os.path.join(mask_dir, baseName + ".png")
            elif os.path.exists(os.path.join(mask_dir, baseName + ".tif")):
                maskPath = os.path.join(mask_dir, baseName + ".tif")
            elif os.path.exists(os.path.join(mask_dir, baseName + ".tiff")):
                maskPath = os.path.join(mask_dir, baseName + ".tiff")
            else:
                sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
            img_A = self.imread(imagePath)
            img_B = self.imread(maskPath)

            nb_channels_A = 1
            if len(img_A.shape) > 2:
                if img_A.shape[0] < img_A.shape[2]:
                    nb_channels_A = img_A.shape[0]
                else:
                    nb_channels_A = img_A.shape[2]
            if nb_channels_A!=self.nb_channels_images:
                    sys.exit("The image " + baseName + " has " + str(nb_channels_A) + " channels instead of " + str(self.nb_channels_images) + ".")
                    
            img_A = img_A.astype('float64')
            new_img_A = np.zeros([img_A.shape[1], img_A.shape[2], nb_channels_A], dtype=np.float64)
            if len(img_A.shape) > 2:
                if img_A.shape[0] < img_A.shape[2]:
                    for c in range(nb_channels_A):
                        new_img_A[:, :, c] = img_A[c, :, :]/(np.max(img_A[c, :, :])/2.) - 1.
                else:
                    for c in range(nb_channels_A):
                        new_img_A[:, :, c] = img_A[:, :, c]/(np.max(img_A[:, :, c])/2.) - 1.
            else:
                new_img_A[:, :, 0] = img_A[:, :]/(np.max(img_A[:, :])/2.) - 1.
            img_A = new_img_A
            
            nb_channels_B = 1
            img_dim_x = 0
            img_dim_y = 1
            if len(img_B.shape) > 2:
                if img_B.shape[0] < img_B.shape[2]:
                    nb_channels_B = img_B.shape[0]
                    img_dim_x = 1
                    img_dim_y = 2
                else:
                    nb_channels_B = img_B.shape[2]
            if nb_channels_B!=self.nb_channels_masks:
                    sys.exit("The mask " + baseName + " has " + str(nb_channels_B) + " channels instead of " + str(self.nb_channels_masks) + ".")
            
            img_B_max = np.max(img_B)
            new_img_B_indices = np.zeros([nb_channels_B, int(img_B_max)], dtype=np.uint32)
            if len(img_B.shape) > 2:
                if img_B.shape[0] < img_B.shape[2]:
                    for c in range(nb_channels_B):
                        for y in range(img_B.shape[1]):
                            for x in range(img_B.shape[2]):
                                index = int(img_B[c,y,x]) - 1
                                if index >= 0:
                                    new_img_B_indices[c, index] = 1
                else:
                    for c in range(nb_channels_B):
                        for y in range(img_B.shape[0]):
                            for x in range(img_B.shape[1]):
                                index = int(img_B[y,x,c]) - 1
                                if index >= 0:
                                    new_img_B_indices[c, index] = 1
            else:
                for y in range(img_B.shape[0]):
                    for x in range(img_B.shape[1]):
                        index = int(img_B[y,x]) - 1
                        if index >= 0:
                            new_img_B_indices[0, index] = 1
                
            for c in range(nb_channels_B):
                count = 0
                for i in range(int(img_B_max)):
                    if new_img_B_indices[c, i] > 0:
                        new_img_B_indices[c, i] = count
                        count += 1
        
            new_img_B = np.zeros([img_B.shape[img_dim_x], img_B.shape[img_dim_y], nb_channels_B], dtype=np.uint32)
            if len(img_B.shape) > 2:
                if img_B.shape[0] < img_B.shape[2]:
                    for c in range(nb_channels_B):
                        for y in range(img_B.shape[1]):
                            for x in range(img_B.shape[2]):
                                index = int(img_B[c,y,x]) - 1
                                if index >= 0:
                                    new_img_B[y, x, c] = new_img_B_indices[c, index] + 1
                else:
                    for c in range(nb_channels_B):
                        for y in range(img_B.shape[0]):
                            for x in range(img_B.shape[1]):
                                index = int(img_B[y,x,c]) - 1
                                if index >= 0:
                                    new_img_B[y, x, c] = new_img_B_indices[c, index] + 1
            else:
                for y in range(img_B.shape[0]):
                    for x in range(img_B.shape[1]):
                        index = int(img_B[y,x]) - 1
                        if index >= 0:
                            new_img_B[y, x, 0] = new_img_B_indices[0, index] + 1

            img_B = new_img_B.astype('float64')
            for c in range(nb_channels_B):
                img_B[:, :, c] = 2.*img_B[:, :, c]/(np.max(img_B[:, :, c])+1) - 1.

            # If training => do random flip
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

            if img_A.shape[2]!=img_B.shape[2]:
                if img_A.shape[2]<img_B.shape[2]:
                    converted_img_A = np.zeros([img_B.shape[0], img_B.shape[1], img_B.shape[2]], dtype=np.float64)
                    for c in range(img_A.shape[2]):
                        converted_img_A[:, :,  c] = img_A[:, :, c]
                    for c in range(img_A.shape[2], img_B.shape[2]):
                        converted_img_A[:, :,  c] = img_A[:, :, img_A.shape[2]-1]
                    img_A = converted_img_A
                else:
                    converted_img_B = np.zeros([img_A.shape[0], img_A.shape[1], img_A.shape[2]], dtype=np.float64)
                    for c in range(img_B.shape[2]):
                        converted_img_B[:, :,  c] = img_B[:, :, c]
                    for c in range(img_B.shape[2], img_A.shape[2]):
                        converted_img_B[:, :,  c] = img_B[:, :, img_B.shape[2]-1]
                    img_B = converted_img_B
            
            imgs_A.append(np.array(img_A).astype('float64'))
            imgs_B.append(np.array(img_B).astype('float64'))

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield imgs_A, imgs_B

    def load_image_test(self):

        image_dir = os.path.join(self.input_test_dir, "images")
        mask_dir = os.path.join(self.input_test_dir, "masks")
        
        imageFileList = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        for imageFile in imageFileList:
            imgs_A, imgs_B = [], []            
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(image_dir, imageFile)
            if os.path.exists(os.path.join(mask_dir, baseName + ".png")):
                maskPath = os.path.join(mask_dir, baseName + ".png")
            elif os.path.exists(os.path.join(mask_dir, baseName + ".tif")):
                maskPath = os.path.join(mask_dir, baseName + ".tif")
            elif os.path.exists(os.path.join(mask_dir, baseName + ".tiff")):
                maskPath = os.path.join(mask_dir, baseName + ".tiff")
            else:
                sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
            img_A = self.imread(imagePath)
            img_B = self.imread(maskPath)

            nb_channels_A = 1
            if len(img_A.shape) > 2:
                if img_A.shape[0] < img_A.shape[2]:
                    nb_channels_A = img_A.shape[0]
                else:
                    nb_channels_A = img_A.shape[2]
            if nb_channels_A!=self.nb_channels_images:
                    sys.exit("The image " + baseName + " has " + str(nb_channels_A) + " channels instead of " + str(self.nb_channels_images) + ".")
            
            img_A = img_A.astype('float64')
            new_img_A = np.zeros([img_A.shape[1], img_A.shape[2], nb_channels_A], dtype=np.float64)
            if len(img_A.shape) > 2:
                if img_A.shape[0] < img_A.shape[2]:
                    for c in range(nb_channels_A):
                        new_img_A[:, :, c] = img_A[c, :, :]/(np.max(img_A[c, :, :])/2.) - 1.
                else:
                    for c in range(nb_channels_A):
                        new_img_A[:, :, c] = img_A[:, :, c]/(np.max(img_A[:, :, c])/2.) - 1.
            else:
                new_img_A[:, :, 0] = img_A[:, :]/(np.max(img_A[:, :])/2.) - 1.
            img_A = new_img_A
            
            img_B_max = np.max(img_B)
            nb_channels_B = 1
            img_dim_x = 0
            img_dim_y = 1
            if len(img_B.shape) > 2:
                if img_B.shape[0] < img_B.shape[2]:
                    nb_channels_B = img_B.shape[0]
                    img_dim_x = 1
                    img_dim_y = 2
                else:
                    nb_channels_B = img_B.shape[2]
            if nb_channels_B!=self.nb_channels_masks:
                    sys.exit("The mask " + baseName + " has " + str(nb_channels_B) + " channels instead of " + str(self.nb_channels_masks) + ".")
            
            new_img_B_indices = np.zeros([nb_channels_B, int(img_B_max)], dtype=np.uint32)
            if len(img_B.shape) > 2:
                if img_B.shape[0] < img_B.shape[2]:
                    for c in range(nb_channels_B):
                        for y in range(img_B.shape[1]):
                            for x in range(img_B.shape[2]):
                                index = int(img_B[c,y,x]) - 1
                                if index >= 0:
                                    new_img_B_indices[c, index] = 1
                else:
                    for c in range(nb_channels_B):
                        for y in range(img_B.shape[0]):
                            for x in range(img_B.shape[1]):
                                index = int(img_B[y,x,c]) - 1
                                if index >= 0:
                                    new_img_B_indices[c, index] = 1
            else:
                for y in range(img_B.shape[0]):
                    for x in range(img_B.shape[1]):
                        index = int(img_B[y,x]) - 1
                        if index >= 0:
                            new_img_B_indices[0, index] = 1

            for c in range(nb_channels_B):
                count = 0
                for i in range(int(img_B_max)):
                    if new_img_B_indices[c, i] > 0:
                        new_img_B_indices[c, i] = count
                        count += 1
        
            new_img_B = np.zeros([img_B.shape[img_dim_x], img_B.shape[img_dim_y], nb_channels_B], dtype=np.uint32)
            if len(img_B.shape) > 2:
                if img_B.shape[0] < img_B.shape[2]:
                    for c in range(nb_channels_B):
                        for y in range(img_B.shape[1]):
                            for x in range(img_B.shape[2]):
                                index = int(img_B[c,y,x]) - 1
                                if index >= 0:
                                    new_img_B[y, x, c] = new_img_B_indices[c, index] + 1
                else:
                    for c in range(nb_channels_B):
                        for y in range(img_B.shape[0]):
                            for x in range(img_B.shape[1]):
                                index = int(img_B[y,x,c]) - 1
                                if index >= 0:
                                    new_img_B[y, x, c] = new_img_B_indices[c, index] + 1
            else:
                for y in range(img_B.shape[0]):
                    for x in range(img_B.shape[1]):
                        index = int(img_B[y,x]) - 1
                        if index >= 0:
                            new_img_B[y, x, 0] = new_img_B_indices[0, index] + 1

            img_B = new_img_B.astype('float64')
            for c in range(nb_channels_B):
                img_B[:, :, c] = 2.*img_B[:, :, c]/(np.max(img_B[:, :, c])+1.) - 1.

            if img_A.shape[2]!=img_B.shape[2]:
                if img_A.shape[2]<img_B.shape[2]:
                    converted_img_A = np.zeros([img_B.shape[0], img_B.shape[1], img_B.shape[2]], dtype=np.float64)
                    for c in range(img_A.shape[2]):
                        converted_img_A[:, :,  c] = img_A[:, :, c]
                    for c in range(img_A.shape[2], img_B.shape[2]):
                        converted_img_A[:, :,  c] = img_A[:, :, img_A.shape[2]-1]
                    img_A = converted_img_A
                else:
                    converted_img_B = np.zeros([img_A.shape[0], img_A.shape[1], img_A.shape[2]], dtype=np.float64)
                    for c in range(img_B.shape[2]):
                        converted_img_B[:, :,  c] = img_B[:, :, c]
                    for c in range(img_B.shape[2], img_A.shape[2]):
                        converted_img_B[:, :,  c] = img_B[:, :, img_B.shape[2]-1]
                    img_B = converted_img_B
            
            imgs_A.append(np.array(img_A).astype('float64'))
            imgs_B.append(np.array(img_B).astype('float64'))

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

        return imgs_A, imgs_B
            

    def imread(self, path):
        return tiff.imread(path).astype(np.float)
