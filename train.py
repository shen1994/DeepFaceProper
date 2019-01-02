# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:34:10 2018

@author: shen1994
"""
import os

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from wide_resnet import WideResNet
from data_generate import Generator

class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return 0.1 # 0.002
        elif epoch_idx < self.epochs * 0.5:
            return 0.02 # 0.0004
        elif epoch_idx < self.epochs * 0.75:
            return 0.0004
        return 0.0008

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    image_size = 128
    batch_size = 64
    nb_epochs = 30
    
    # depth of network (should be 10, 16, 22, 28, ...)
    # width of network (should be 2, 4, 6, 8, 10, ...)
    model = WideResNet(image_size, depth=16, k=4)()
    model.load_weights('model/weights.24-5.84.hdf5', by_name=True)
    sgd = SGD(lr=0.002, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=["categorical_crossentropy", "categorical_crossentropy"])

    gen = Generator(train_full_path="images/imdb_detect/imdb.csv", \
                    valid_full_path="images/wiki_detect/wiki.csv", \
                    image_size=image_size, \
                    batch_size=batch_size)

    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs)),
                 ModelCheckpoint("model/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 verbose=1,
                                 save_weights_only=True)]                           
                
    history = model.fit_generator(generator=gen.generate(True),
                                  steps_per_epoch=gen.train_numbers // batch_size,
                                  validation_data=gen.generate(False),
                                  validation_steps=gen.valid_numbers // batch_size,
                                  epochs=nb_epochs,
                                  verbose=1,
                                  callbacks=callbacks,
                                  workers=2)

    
    
    
    
    
    
    
    