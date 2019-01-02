# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:48:43 2018

@author: shen1994
"""

import cv2
import random
import numpy as np

class Generator(object):
    
    def __init__(self,
        train_full_path,
        valid_full_path,
        image_size = 128,
        batch_size = 64,
        saturation_var=0.5,
        brightness_var=0.5,
        contrast_var=0.5,
        lighting_std=0.5,
        hflip_prob=0.0):
        train_useful_data = []
        with open(train_full_path, 'r') as f:
            line = f.readline().replace(',', ' ').strip().split()
            while(line):
                train_useful_data.append(line)
                line = f.readline().replace(',', ' ').strip().split()
        valid_useful_data = []
        with open(valid_full_path, 'r') as f:
            line = f.readline().replace(',', ' ').strip().split()
            while(line):
                valid_useful_data.append(line)
                line = f.readline().replace(',', ' ').strip().split()
        self.train_numbers = len(train_useful_data)
        self.valid_numbers = len(valid_useful_data)
        self.train_data = train_useful_data
        self.valid_data = valid_useful_data
        self.image_size = image_size
        self.batch_size = batch_size
        self.saturation_var = saturation_var
        self.brightness_var = brightness_var
        self.contrast_var = contrast_var
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])
        
    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
        
    def horizontal_flip(self, img):
        rand = np.random.random()
        if  rand < self.hflip_prob:
            img = img[:, ::-1]
        return img
    
    def generate(self, train=True):
        while(True):
            if train:
                random.shuffle(self.train_data)
                data = self.train_data
            else:
                random.shuffle(self.valid_data)
                data = self.valid_data
                
            inputs = []
            age_targets = []
            gender_targets = []
            for one_data in data:
                
                image_path = one_data[0]
                image = cv2.imread(image_path, 1)
                image = cv2.resize(image, (self.image_size, self.image_size))
                image = image.astype('float32')
            
                if train:
                    if self.saturation_var > 0:
                        image = self.saturation(image)
                    if self.brightness_var > 0:
                        image = self.brightness(image)
                    if self.contrast_var > 0:
                        image = self.contrast(image)
                    if self.lighting_std > 0:
                        image = self.lighting(image)
                    if self.hflip_prob > 0:
                        image = self.horizontal_flip(image)
                inputs.append(image)
                
                age = int(one_data[1])
                age_onehot = np.zeros(101)
                age_onehot[age] = 1
                age_targets.append(age_onehot)
                
                gender = int(one_data[2])
                gender_onehot = np.zeros(2)
                gender_onehot[gender] = 1
                gender_targets.append(gender_onehot)
                
                if len(inputs) == self.batch_size:
                    inputs_array = np.array(inputs)
                    targets = [np.array(gender_targets), np.array(age_targets)]
                    inputs = []
                    age_targets = []
                    gender_targets = []
                    yield (inputs_array, targets)
                             