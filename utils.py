# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

"""
Import python packages
"""

import os
from tensorflow import keras
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from math import pi, sqrt
import skimage
from skimage.draw import (ellipse)
from scipy.misc import bytescale
import sys

def get_image(file_name):
    if ('.tif' in file_name) or ('tiff' in file_name):
        im = tiff.imread(file_name)
        im = bytescale(im)
        im = np.float32(im)
    else:
        im = np.float32(imread(file_name))
        
    if len(im.shape) < 3:
        output_im = np.zeros((im.shape[0], im.shape[1], 1))
        output_im[:, :, 0] = im
        im = output_im
    else:
        if im.shape[0]<im.shape[2]:
            output_im = np.zeros((im.shape[1], im.shape[2], im.shape[0]))
            for i in range(im.shape[0]):
                output_im[:, :, i] = im[i, :, :]
            im = output_im
    
    return im

def generate_images(input_masks_dir, nb_channels_images, nb_channels_masks, imaging_field,
               generator_path, output_dir_images, output_dir_masks):
    
    generator = keras.models.load_model(generator_path, compile=False)

    imageFiles = [f for f in os.listdir(input_masks_dir) if os.path.isfile(os.path.join(input_masks_dir, f))]
    
    input_data = []
    mask_data = []
    for index, imageFile in enumerate(imageFiles):
        imagePath = os.path.join(input_masks_dir, imageFile)
        baseName = os.path.splitext(os.path.basename(imageFile))[0]
        image = tiff.imread(imagePath)
        if len(image.shape)>2:
            if image.shape[0] < image.shape[2]:
                if image.shape[0]!=nb_channels_masks:
                    sys.exit("The mask " + baseName + " has " + str(image.shape[0]) + " channels instead of " + str(nb_channels_masks) + ".")
                if image.shape[1]!=imaging_field:
                    sys.exit("The mask " + baseName + " has " + str(image.shape[1]) + " for dimension x instead of " + str(imaging_field) + ".")
                if image.shape[2]!=imaging_field:
                    sys.exit("The mask " + baseName + " has " + str(image.shape[2]) + " for dimension y instead of " + str(imaging_field) + ".")
            else:
                if image.shape[2]!=nb_channels_masks:
                    sys.exit("The mask " + baseName + " has " + str(image.shape[2]) + " channels instead of " + str(nb_channels_masks) + ".")
                if image.shape[0]!=imaging_field:
                    sys.exit("The mask " + baseName + " has " + str(image.shape[0]) + " for dimension x instead of " + str(imaging_field) + ".")
                if image.shape[1]!=imaging_field:
                    sys.exit("The mask " + baseName + " has " + str(image.shape[1]) + " for dimension y instead of " + str(imaging_field) + ".")
        else:
            if nb_channels_masks>1:
                sys.exit("The mask " + baseName + " has 1 channel instead of " + str(nb_channels_masks) + ".")
            if image.shape[0]!=imaging_field:
                sys.exit("The mask " + baseName + " has " + str(image.shape[0]) + " for dimension x instead of " + str(imaging_field) + ".")
            if image.shape[1]!=imaging_field:
                sys.exit("The mask " + baseName + " has " + str(image.shape[1]) + " for dimension y instead of " + str(imaging_field) + ".")
                    

        mask_data.append(image)
        
        nb_channels = max(nb_channels_images, nb_channels_masks)
        input_image = np.zeros((imaging_field, imaging_field, nb_channels), np.uint32)
        if len(image.shape)>2:
            if image.shape[0] < image.shape[2]:
                for c in range(nb_channels_masks):
                    input_image[:, :, c] = image[c, :, :]
            else:
                for c in range(nb_channels_masks):
                    input_image[:, :, c] = image[:, :, c]
        else:
            input_image[:, :, 0] = image[:, :]
        if nb_channels_masks<nb_channels_images:
            for c in range(nb_channels_masks, nb_channels_images):
                input_image[:, :, c] = input_image[:, :, nb_channels_masks-1]

        input_image = input_image.astype('float64')
        for c in range(nb_channels_masks):
            input_image[:, :, c] = 2.*input_image[:, :, c]/(np.max(input_image[:, :, c])+1.) - 1.

        input_data.append(np.array(input_image).astype('float64'))

    generated_input_image = generator.predict(np.array(input_data))
    
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)

    for i in range(len(generated_input_image)):
        output_generated_image = np.zeros([imaging_field, imaging_field, nb_channels_images])
        for k in range(nb_channels_images):
            output_generated_image[:, :, k] = 4095 * (0.5 * generated_input_image[i][:, :, 0] + 0.5)
        tiff.imsave(os.path.join(output_dir_images, "image_" + str(i) + ".tiff"), output_generated_image.astype('uint16'))
        tiff.imsave(os.path.join(output_dir_masks, "image_" + str(i) + ".tiff"), mask_data[i].astype('uint16'))

        
def split_images_and_extract_channels(input_image_dir, output_image_dir, channels, imaging_field):        

    imageFiles = [f for f in os.listdir(input_image_dir) if os.path.isfile(os.path.join(input_image_dir, f))]
    image_test = get_image(os.path.join(input_image_dir, imageFiles[0]))
    for k in range(len(channels)):
        if channels[k]>=image_test.shape[2]:
            sys.exit("The channels to be extracted are superior to the actual number of channels in the input images.")
    if imaging_field>image_test.shape[0]:
        sys.exit("The imaging field is superior to the x dimension of the input images.")
    if imaging_field>image_test.shape[1]:
        sys.exit("The imaging field is superior to the y dimension of the input images.")
    nb_x_bins = int(image_test.shape[0]/imaging_field)
    nb_y_bins = int(image_test.shape[1]/imaging_field)
    os.makedirs(output_image_dir, exist_ok=True)
    

    for index, imageFile in enumerate(imageFiles):
        imagePath = os.path.join(input_image_dir, imageFile)
        baseName = os.path.splitext(os.path.basename(imageFile))[0]
        image = tiff.imread(imagePath)
        
        i = 0
        j = 0
        for i in range(nb_x_bins):
            for j in range(nb_y_bins):
                x_init = i*imaging_field
                x_end = x_init + imaging_field
                y_init = j*imaging_field
                y_end = y_init + imaging_field
            
                if len(image.shape)>2:
                    output_image = np.zeros((len(channels), imaging_field, imaging_field), np.uint32)
                    if image.shape[0] < image.shape[-1]:
                        for k in range(len(channels)):
                            output_image[k, :, :] = (image[channels[k], x_init:x_end, y_init:y_end]).astype('uint32')
                            tiff.imsave(os.path.join(output_image_dir, baseName + "_" + str(i) + "_" + str(j) + ".tiff"), output_image)
                    else:
                        for k in range(len(channels)):
                            output_image[k, :, :] = (image[x_init:x_end, y_init:y_end, channels[k]]).astype('uint32')
                            tiff.imsave(os.path.join(output_image_dir, baseName + "_" + str(i) + "_" + str(j) + ".tiff"), output_image)
                else:
                    output_image = np.zeros((1, imaging_field, imaging_field), np.uint32)
                    output_image = (image[x_init:x_end, y_init:y_end]).astype('uint32')
                    tiff.imsave(os.path.join(output_image_dir, baseName + "_" + str(i) + "_" + str(j) + ".tiff"), output_image)
                    
                    
def estimate_distributions(input_mask_dir):
    
    imageFiles = [f for f in os.listdir(input_mask_dir) if os.path.isfile(os.path.join(input_mask_dir, f))]
    
    nb_nuclei = []
    size_distribution_total = []
    for index, imageFile in enumerate(imageFiles):
        maskPath = os.path.join(input_mask_dir, imageFile)
        mask = tiff.imread(maskPath)
            
        mask_max = np.max(mask)
        mask_indices = np.zeros([int(mask_max)], dtype=np.uint32)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                index = int(mask[y,x]) - 1
                if index >= 0:
                    mask_indices[index] += 1
                        
        count = 0
        for k in range(int(mask_max)):
            if mask_indices[k] > 0:
                count += 1
                size_distribution_total.append(mask_indices[k])

        nb_nuclei.append(count)
            
    plt.rcdefaults()
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(nb_nuclei, bins=np.arange(-200,800,50))
    axs[0, 0].set_title("Number of nuclei per image")
    axs[0, 0].set_xlabel("Number of nuclei")
    axs[0, 0].set_ylabel("Frequency")

    s = np.random.normal(np.mean(nb_nuclei), np.std(nb_nuclei), 1000)
    axs[1, 0].hist(s, bins=np.arange(-200,800,50), facecolor="orange")
    axs[1, 0].set_title("Gaussian distribution for\nnumber of nuclei per image")
    axs[1, 0].set_xlabel("Number of nuclei")
    axs[1, 0].set_ylabel("Frequency")

    axs[0, 1].hist(size_distribution_total, bins=np.arange(10,500,10))
    axs[0, 1].set_title("Size of nuclei distribution")
    axs[0, 1].set_xlabel("Size of nuclei")
    axs[0, 1].set_ylabel("Frequency")

    s = np.random.gumbel(np.mean(size_distribution_total), 65, 10000)
    axs[1, 1].hist(s, bins=np.arange(0,500,10), facecolor="orange")
    axs[1, 1].set_title("Gumbel distribution for\nsize of nuclei")
    axs[1, 1].set_xlabel("Size of nuclei")
    axs[1, 1].set_ylabel("Frequency")
    plt.tight_layout()
    
    return np.mean(nb_nuclei), np.std(nb_nuclei), np.mean(size_distribution_total)


def generate_images_with_ellipses(input_dim_x, input_dim_y, total_nb_images, output_masks_dir, avg_nb_nuclei, std_nb_nuclei, avg_nuclei_size):
    os.makedirs(output_masks_dir, exist_ok=True)
    approximate_nb_objects = np.random.normal(avg_nb_nuclei, std_nb_nuclei, total_nb_images)
    image_index = 1
    for i in range(len(approximate_nb_objects)):
        if int(approximate_nb_objects[0]) > 0:
            output = np.zeros((input_dim_x, input_dim_y), dtype=np.uint16)
            random_image = np.random.rand(input_dim_x,input_dim_y)
            output_centers = np.where(random_image < float(approximate_nb_objects[i])/float(input_dim_x*input_dim_y), 1, 0)
            current_index = 1
            s = np.random.gumbel(avg_nuclei_size, 65, np.sum(output_centers))
            for y in range(input_dim_y):
                for x in range(input_dim_x):
                    if output_centers[x,y]>0:
                        size = s[current_index-1]
                        if size<10.:
                            size = 10.
                        r_x = np.random.normal(sqrt(size / (pi)), .2*sqrt(size / (pi)))
                        r_y = size / (pi * r_x)
                        rotation = np.random.uniform(low=-pi, high=+pi)
                        rr, cc = ellipse(x, y, r_x, r_y, output.shape, rotation)
                        output[rr, cc] = current_index
                        current_index += 1

            tiff.imsave(os.path.join(output_masks_dir, "image_%d.tiff" % image_index), output.astype('uint16'))
            image_index += 1
