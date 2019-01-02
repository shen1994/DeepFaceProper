# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:45:24 2018

@author: shen1994
"""

import os
import cv2
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    output_graph_def = tf.GraphDef()
    output_graph_def.ParseFromString(open("model/pico_FaceProper_model.pb", "rb").read())
    tensors = tf.import_graph_def(output_graph_def, name="")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    opt = sess.graph.get_operations()
    input_x = sess.graph.get_tensor_by_name("proper_input:0")
    gender_y = sess.graph.get_tensor_by_name("pred_gender/Softmax:0")
    age_y = sess.graph.get_tensor_by_name("pred_age/Softmax:0")
    
    image_path = "test.jpg"
    image_size = 128
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype('float32')
    
    gender_list, age_list = sess.run([gender_y, age_y], \
                           feed_dict={input_x: [image]})

    g_gender_list = np.arange(0, 2).reshape(2,)
    g_age_list = np.arange(0, 101).reshape(101,)
    
    gender = gender_list.dot(g_gender_list)[0]
    age = age_list.dot(g_age_list)[0]
    
    print(gender, age)
    