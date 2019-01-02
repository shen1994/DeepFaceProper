# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:34:27 2018

@author: shen1994
"""

import os
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework import graph_io

from wide_resnet import WideResNet

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    image_size = 128

    model = WideResNet(image_size, depth=16, k=4)()
    model.load_weights('model/weights.24-5.84.hdf5', by_name=True)

    # boxes = tf.placeholder(dtype='float32', shape=(None, 4))
    # scores = tf.placeholder(dtype='float32', shape=(None,))
    # nms = tf.image.non_max_suppression(boxes, scores, 100, iou_threshold=0.45)
    # model.input.name = 'proper_input'
    # model.input.rename
    print('input name is: ', model.input.name)
    print('output name is: ', model.output[0].name)
    print('output name is: ', model.output[1].name)
    
    K.set_learning_phase(0)
    frozen_graph = freeze_session(K.get_session(), \
                                  output_names=[model.output[0].op.name, model.output[1].op.name])
    graph_io.write_graph(frozen_graph, "model/", "pico_FaceProper_model.pb", as_text=False)
    
    