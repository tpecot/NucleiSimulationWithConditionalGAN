from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import tifffile as tiff

class Pix2Pix():
    def __init__(self, input_training_dir = None, input_test_dir = None, nb_rows = 256, nb_channels_A = 1, nb_channels_B = 1, learning_rate = 1e-4, beta = 0.5, output_simulations_dir = "simulations", output_generator_dir = "trainedClassifiers", generator_name = "generator"):
        self.input_training_dir = input_training_dir
        self.input_test_dir = input_test_dir
        self.img_rows = nb_rows
        self.img_cols = nb_rows
        self.channels_A = nb_channels_A
        self.channels_B = nb_channels_B
        if self.channels_A>self.channels_B:
            self.channels = self.channels_A
        else:
            self.channels = self.channels_B
        self.lr = learning_rate
        self.beta = beta
        self.output_simulations_dir = output_simulations_dir
        self.output_generator_dir = output_generator_dir
        self.generator_name = generator_name
        self.initialize()
        
    def initialize(self):
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.data_loader = DataLoader(input_training_dir = self.input_training_dir, input_test_dir = self.input_test_dir, img_res=(self.img_rows, self.img_cols), nb_channels_images = self.channels_A, nb_channels_masks = self.channels_B)


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(self.lr, self.beta)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])
                    
        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs):

        start_time = datetime.datetime.now()
        batch_size=1

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch()):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                
            # If at save interval => save generated image samples                
            self.sample_images(epoch)
            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch+1, epochs, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time))
            
        # save generator network
        os.makedirs(self.output_generator_dir, exist_ok=True)
        self.generator.save("%s/%s.h5" % (self.output_generator_dir, self.generator_name))

    def sample_images(self, epoch):
        os.makedirs(self.output_simulations_dir, exist_ok=True)
        imgs_A, imgs_B = self.data_loader.load_image_test()
        
        fake_A = self.generator.predict(imgs_B)
        fake_A = (0.5 * fake_A + 0.5) * 255
        
        
        if fake_A.shape[3]!=self.channels_A:
            if self.channels_A==2:
                new_fake_A = np.zeros([fake_A.shape[1], fake_A.shape[2], 3], dtype=np.float64)
                for c in range(2):
                    new_fake_A[:, :, c] = fake_A[0, :, :, c]
                fake_A = new_fake_A
            else:
                new_fake_A = np.zeros([fake_A.shape[1], fake_A.shape[2], self.channels_A], dtype=np.float64)
            for c in range(self.channels_A):
                new_fake_A[:, :, c] = fake_A[0, :, :, c]
                fake_A = new_fake_A
        elif fake_A.shape[3]==2:
            new_fake_A = np.zeros([fake_A.shape[1], fake_A.shape[2], 3], dtype=np.float64)
            for c in range(2):
                new_fake_A[:, :, c] = fake_A[0, :, :, c]
            fake_A = new_fake_A
        
        tiff.imsave("%s/simulation_%d.tif" % (self.output_simulations_dir, epoch), fake_A.astype('uint8'))
        